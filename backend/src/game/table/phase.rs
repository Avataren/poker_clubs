use super::*;

impl PokerTable {
    pub(crate) fn advance_phase(&mut self) {
        // Record when phase changed
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_phase_change_time = Some(now);

        // Reset current bets for new betting round
        for player in &mut self.players {
            player.reset_for_new_round();
        }
        self.pot.end_betting_round();
        self.current_bet = 0;
        self.raises_this_round = 0;

        // If only one player is active in hand, they win immediately (no showdown)
        let active_in_hand: Vec<usize> = self
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_active_in_hand())
            .map(|(idx, _)| idx)
            .collect();

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
            if let Ok(next) = self.phase.transition_to(GamePhase::Showdown) {
                self.phase = next;
            }
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
                if let Ok(next) = self.phase.transition_to(GamePhase::Flop) {
                    self.phase = next;
                }
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
                if let Ok(next) = self.phase.transition_to(GamePhase::Turn) {
                    self.phase = next;
                }
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
                if let Ok(next) = self.phase.transition_to(GamePhase::River) {
                    self.phase = next;
                }
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

    // Check if enough time has passed to auto-advance to next phase
    pub fn check_auto_advance(&mut self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // SNG and MTT should share tournament table behavior.
        // SNG differs only in how the tournament starts (auto-start when full).
        let is_tournament_table = self.tournament_id.is_some() && self.format.eliminates_players();

        // Tournament tables need a deterministic Waiting window so periodic balancing
        // can move players between hands.
        if self.phase == GamePhase::Waiting && is_tournament_table {
            let playable_count = self.players.iter().filter(|p| p.stack > 0).count();
            if playable_count < MIN_PLAYERS_TO_START {
                return false;
            }

            let waiting_since = match self.last_phase_change_time {
                Some(ts) => ts,
                None => {
                    self.last_phase_change_time = Some(now);
                    return false;
                }
            };

            let elapsed = now.saturating_sub(waiting_since);
            if elapsed < MTT_WAITING_REBALANCE_MS {
                return false;
            }

            tracing::info!(
                "Tournament Waiting window elapsed ({}ms), starting next hand",
                elapsed
            );
            self.start_new_hand();
            return true;
        }

        // During Showdown, always allow auto-advance (start new hand after delay).
        // For other phases, auto-advance only if fewer than 2 players can act
        // (everyone all-in or folded, no meaningful betting possible).
        if self.phase != GamePhase::Showdown {
            let can_act_count = self.players.iter().filter(|p| p.can_act()).count();
            if can_act_count >= 2 {
                return false;
            }
        }

        // Check if we're in a phase that needs delay
        let (needs_delay, delay_ms) = match self.phase {
            GamePhase::Flop | GamePhase::Turn | GamePhase::River => (true, self.street_delay_ms),
            GamePhase::Showdown => (true, self.showdown_delay_ms),
            _ => (false, 0),
        };

        if !needs_delay {
            return false;
        }

        // Check if enough time has passed
        if let Some(last_change) = self.last_phase_change_time {
            let elapsed = now.saturating_sub(last_change);

            if elapsed >= delay_ms {
                tracing::info!(
                    "Auto-advancing from {:?} after {}ms delay",
                    self.phase,
                    elapsed
                );

                if self.phase == GamePhase::Showdown {
                    if is_tournament_table {
                        // Tournament tables must pass through Waiting so balancing can run.
                        tracing::info!(
                            "Showdown delay complete on tournament table, transitioning to Waiting"
                        );
                        if let Ok(next) = self.phase.transition_to(GamePhase::Waiting) {
                            self.phase = next;
                        }

                        // Remove busted players once showdown has been visible long enough.
                        self.check_eliminations();
                        self.last_phase_change_time = Some(now);
                    } else if self.players.len() >= MIN_PLAYERS_TO_START {
                        // Cash games continue immediately between hands.
                        tracing::info!("Starting new hand after showdown delay");
                        self.start_new_hand();
                    } else {
                        tracing::info!("Not enough players, going to Waiting phase");
                        if let Ok(next) = self.phase.transition_to(GamePhase::Waiting) {
                            self.phase = next;
                        }
                        self.last_phase_change_time = Some(now);
                    }
                } else {
                    // Advance to next street
                    self.advance_phase();
                }

                return true;
            }
        }

        false
    }
}
