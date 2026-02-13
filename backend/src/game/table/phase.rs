use super::*;

impl PokerTable {
    fn fold_win_delay_ms(&self) -> u64 {
        let winner_is_bot = self
            .players
            .iter()
            .any(|p| p.is_winner && is_bot_identity(&p.user_id, &p.username));

        if winner_is_bot {
            BOT_FOLD_WIN_DELAY_MS
        } else {
            DEFAULT_FOLD_WIN_DELAY_MS
        }
    }

    /// Attempt a phase transition. Silently ignores invalid transitions.
    pub(crate) fn try_transition(&mut self, target: GamePhase) {
        if let Ok(next) = self.phase.transition_to(target) {
            self.phase = next;
        }
    }

    pub(crate) fn advance_phase(&mut self) {
        // Return uncallable bets BEFORE advancing and resetting
        // This handles cases where a player has chips that nobody can match because
        // others are all-in for less. Standard poker etiquette.
        let active_in_hand: Vec<usize> = self
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_active_in_hand())
            .map(|(idx, _)| idx)
            .collect();
        
        // Only return uncallable bets on PreFlop -> Flop transition
        if active_in_hand.len() > 1 && self.phase == GamePhase::PreFlop {
            let pot_before = self.pot.total();
            let uncontested = self.pot.calculate_side_pots(&self.player_bets());
            let pot_after = self.pot.total();
            
            for (player_idx, amount) in uncontested {
                self.players[player_idx].add_chips(amount);
                tracing::info!(
                    "Returned ${} uncallable bet to {} (pot was ${}, now ${})",
                    amount,
                    self.players[player_idx].username,
                    pot_before,
                    pot_after
                );
            }
        }

        // Record when phase changed
        self.last_phase_change_time = Some(current_timestamp_ms());

        // Reset current bets for new betting round
        for player in &mut self.players {
            player.reset_for_new_round();
        }
        self.pot.end_betting_round();
        
        self.current_bet = 0;
        self.raises_this_round = 0;

        // If only one player is active in hand, they win immediately (no showdown)

        if active_in_hand.len() == 1 {
            let winner_idx = active_in_hand[0];
            let total = self.pot.total();
            self.players[winner_idx].add_chips(total);
            self.players[winner_idx].is_winner = true;
            self.players[winner_idx].pot_won = total;
            self.won_without_showdown = true;
            // Initialize shown_cards to all false (winner can choose to reveal)
            let num_cards = self.players[winner_idx].hole_cards.len();
            self.players[winner_idx].shown_cards = vec![false; num_cards];
            self.last_winner_message = Some(format!(
                "{} wins ${}",
                self.players[winner_idx].username, total
            ));
            // Pot reset happens in start_new_hand(), not here
            self.try_transition(GamePhase::Showdown);
            tracing::info!(
                "Hand over: {} wins ${} (all others folded)",
                self.players[winner_idx].username,
                total
            );
            return;
        }

        match self.phase {
            GamePhase::PreFlop => {
                // Burn a card before dealing the flop
                self.deck.deal();
                // Deal flop
                self.community_cards = self.deck.deal_multiple(3);
                self.try_transition(GamePhase::Flop);
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::Flop => {
                // Burn a card before dealing the turn
                self.deck.deal();
                // Deal turn
                if let Some(card) = self.deck.deal() {
                    self.community_cards.push(card);
                } else {
                    tracing::error!("Deck exhausted when dealing turn card");
                }
                self.try_transition(GamePhase::Turn);
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::Turn => {
                // Burn a card before dealing the river
                self.deck.deal();
                // Deal river
                if let Some(card) = self.deck.deal() {
                    self.community_cards.push(card);
                } else {
                    tracing::error!("Deck exhausted when dealing river card");
                }
                self.try_transition(GamePhase::River);
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::River => {
                // Go to showdown
                if let Ok(next) = self.phase.transition_to(GamePhase::Showdown) {
                    self.phase = next;
                }
                self.resolve_showdown();
            }
            _ => {}
        }
    }

    /// Tournament Waiting phase: wait for the rebalance window then start next hand.
    /// Returns Some(true/false) if handled, None if this isn't a tournament waiting scenario.
    fn check_tournament_waiting(&mut self, now: u64) -> Option<bool> {
        let is_tournament_table = self.tournament_id.is_some() && self.format.eliminates_players();
        if self.phase != GamePhase::Waiting || !is_tournament_table {
            return None;
        }

        let playable_count = self.players.iter().filter(|p| p.stack > 0).count();
        if playable_count < MIN_PLAYERS_TO_START {
            return Some(false);
        }

        let waiting_since = match self.last_phase_change_time {
            Some(ts) => ts,
            None => {
                self.last_phase_change_time = Some(now);
                return Some(false);
            }
        };

        let elapsed = now.saturating_sub(waiting_since);
        if elapsed < MTT_WAITING_REBALANCE_MS {
            return Some(false);
        }

        tracing::info!(
            "Tournament Waiting window elapsed ({}ms), starting next hand",
            elapsed
        );
        self.start_new_hand();
        Some(true)
    }

    /// After showdown delay expires, transition to next hand or Waiting.
    fn advance_after_showdown(&mut self, now: u64) {
        let is_tournament_table = self.tournament_id.is_some() && self.format.eliminates_players();

        if is_tournament_table {
            tracing::info!("Showdown delay complete on tournament table, transitioning to Waiting");
            self.try_transition(GamePhase::Waiting);
            self.check_eliminations();
            self.last_phase_change_time = Some(now);
        } else if self.players.len() >= MIN_PLAYERS_TO_START {
            tracing::info!("Starting new hand after showdown delay");
            self.start_new_hand();
        } else {
            tracing::info!("Not enough players, going to Waiting phase");
            self.try_transition(GamePhase::Waiting);
            self.last_phase_change_time = Some(now);
        }
    }

    /// Recover from turn-pointer desync in betting phases.
    ///
    /// Under heavy tournament churn (table moves/sit-outs/disconnects), the
    /// current_player index can occasionally point at a player who cannot act.
    /// If at least one player can act, rotate to an actionable player so the
    /// hand cannot deadlock waiting on an impossible turn.
    fn recover_stuck_current_player(&mut self) -> bool {
        if !matches!(
            self.phase,
            GamePhase::PreFlop | GamePhase::Flop | GamePhase::Turn | GamePhase::River
        ) {
            return false;
        }

        if self.players.is_empty() || self.players.iter().all(|p| !p.can_act()) {
            return false;
        }

        if self.current_player >= self.players.len() {
            if let Some(next) = self.first_player_index_by_seat(|p| p.can_act()) {
                tracing::warn!(
                    "Recovered out-of-bounds current_player {} -> {} on table {}",
                    self.current_player,
                    next,
                    self.table_id
                );
                self.current_player = next;
                return true;
            }
            return false;
        }

        if self.players[self.current_player].can_act() {
            return false;
        }

        let original_idx = self.current_player;
        for _ in 0..self.players.len() {
            let next = self.next_active_player(self.current_player);
            if next == self.current_player {
                break;
            }
            self.current_player = next;
            if self.players[self.current_player].can_act() {
                tracing::warn!(
                    "Recovered non-actionable turn {} -> {} on table {}",
                    original_idx,
                    self.current_player,
                    self.table_id
                );
                return true;
            }
        }

        if let Some(next) = self.first_player_index_by_seat(|p| p.can_act()) {
            if next != self.current_player {
                tracing::warn!(
                    "Recovered turn using seat scan {} -> {} on table {}",
                    self.current_player,
                    next,
                    self.table_id
                );
                self.current_player = next;
                return true;
            }
        }

        false
    }

    /// Check if enough time has passed to auto-advance to the next phase.
    pub fn check_auto_advance(&mut self) -> bool {
        let now = current_timestamp_ms();
        let mut recovered_turn = false;

        // Tournament Waiting window (rebalance before next hand)
        if let Some(result) = self.check_tournament_waiting(now) {
            return result;
        }

        // During Showdown, always allow auto-advance (start new hand after delay).
        // For betting phases, do not auto-advance while any active player still
        // has a pending decision (for example: one covered player facing an all-in).
        // Only fully-closed action (no active decision makers) or a completed
        // betting round may progress automatically.
        if self.phase != GamePhase::Showdown {
            recovered_turn = self.recover_stuck_current_player();
            let betting_round_complete = self.is_betting_round_complete();
            let can_act_count = self.players.iter().filter(|p| p.can_act()).count();
            if can_act_count > 0 && !betting_round_complete {
                return recovered_turn;
            }
        }

        // Determine delay for current phase
        let delay_ms = match self.phase {
            GamePhase::PreFlop | GamePhase::Flop | GamePhase::Turn | GamePhase::River => {
                self.street_delay_ms
            }
            GamePhase::Showdown => {
                if self.won_without_showdown {
                    self.fold_win_delay_ms()
                } else {
                    self.showdown_delay_ms
                }
            }
            _ => return recovered_turn,
        };

        // Check if enough time has passed
        let last_change = match self.last_phase_change_time {
            Some(ts) => ts,
            None => return recovered_turn,
        };
        let elapsed = now.saturating_sub(last_change);
        if elapsed < delay_ms {
            return recovered_turn;
        }

        tracing::info!(
            "Auto-advancing from {:?} after {}ms delay",
            self.phase,
            elapsed
        );

        if self.phase == GamePhase::Showdown {
            self.advance_after_showdown(now);
        } else {
            self.advance_phase();
        }

        true
    }
}
