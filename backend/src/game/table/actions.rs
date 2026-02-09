use super::*;
use crate::game::variant::BettingStructure;

impl PokerTable {
    pub fn handle_action(&mut self, user_id: &str, action: PlayerAction) -> GameResult<()> {
        tracing::debug!(
            "handle_action: user_id={}, action={:?}, current_player={}",
            user_id,
            action,
            self.current_player
        );

        // Verify it's the player's turn
        let current = &self.players[self.current_player];
        if current.user_id != user_id {
            tracing::debug!("Not your turn: current player is {}", current.user_id);
            return Err(GameError::NotYourTurn);
        }

        if !current.can_act() {
            tracing::debug!("Cannot act: state={:?}", current.state);
            return Err(GameError::CannotAct);
        }
        let acted_by_bot = is_bot_identity(&current.user_id, &current.username);

        tracing::debug!("Processing action: {:?}", action);
        let betting_structure = self.variant.betting_structure();
        // Process action
        match action {
            PlayerAction::Fold => {
                self.players[self.current_player].fold();
            }
            PlayerAction::Check => {
                if self.current_bet > self.players[self.current_player].current_bet {
                    return Err(GameError::CannotCheck {
                        current_bet: self.current_bet,
                    });
                }
            }
            PlayerAction::Call => {
                let to_call = self.current_bet - self.players[self.current_player].current_bet;
                let actual = self.players[self.current_player].place_bet(to_call);
                self.pot.add_bet(self.current_player, actual);
            }
            PlayerAction::Raise(amount) => {
                let current_bet = self.players[self.current_player].current_bet;
                let raise_to = self.current_bet + amount;

                // Ensure min_raise is never below big_blind
                let effective_min_raise = self.min_raise.max(self.big_blind);
                if amount < effective_min_raise {
                    return Err(GameError::RaiseTooSmall {
                        min_raise: effective_min_raise,
                        attempted: amount,
                    });
                }

                match betting_structure {
                    BettingStructure::PotLimit => {
                        let to_call = self.current_bet - current_bet;
                        let max_raise = self.pot.total() + to_call;
                        if amount > max_raise {
                            return Err(GameError::RaiseTooLarge {
                                max_raise,
                                attempted: amount,
                            });
                        }
                    }
                    BettingStructure::FixedLimit { small_bet, big_bet } => {
                        if self.raises_this_round >= MAX_RAISES_PER_ROUND {
                            return Err(GameError::MaxRaisesReached {
                                max_raises: MAX_RAISES_PER_ROUND,
                            });
                        }
                        // Early streets (PreFlop, Flop) use small_bet; later streets use big_bet
                        let required = match self.phase {
                            GamePhase::PreFlop | GamePhase::Flop => small_bet,
                            _ => big_bet,
                        };
                        if amount != required {
                            return Err(GameError::RaiseNotExact {
                                required,
                                attempted: amount,
                            });
                        }
                    }
                    BettingStructure::NoLimit => {}
                }

                let total_to_bet = raise_to - current_bet;
                let actual = self.players[self.current_player].place_bet(total_to_bet);
                self.pot.add_bet(self.current_player, actual);

                self.current_bet = current_bet + actual;
                // min_raise is the raise size, but never below big_blind
                self.min_raise = amount.max(self.big_blind);
                self.raises_this_round += 1;

                // Reset has_acted for all other players since there's a new bet
                for (idx, player) in self.players.iter_mut().enumerate() {
                    if idx != self.current_player {
                        player.has_acted_this_round = false;
                    }
                }
            }
            PlayerAction::AllIn => {
                let stack = self.players[self.current_player].stack;

                match betting_structure {
                    BettingStructure::PotLimit => {
                        let current_bet = self.players[self.current_player].current_bet;
                        let to_call = self.current_bet - current_bet;
                        let max_raise = self.pot.total() + to_call;
                        let max_total = self.current_bet + max_raise;
                        let desired_total = current_bet + stack;

                        if desired_total > max_total {
                            let attempted_raise = desired_total.saturating_sub(self.current_bet);
                            return Err(GameError::RaiseTooLarge {
                                max_raise,
                                attempted: attempted_raise,
                            });
                        }
                    }
                    BettingStructure::FixedLimit { .. } => {
                        // In fixed-limit, all-in is always allowed (player may not have
                        // enough chips for a full bet/raise). If this would be a raise,
                        // check the raise cap.
                        let current_bet = self.players[self.current_player].current_bet;
                        let desired_total = current_bet + stack;
                        if desired_total > self.current_bet
                            && self.raises_this_round >= MAX_RAISES_PER_ROUND
                        {
                            return Err(GameError::MaxRaisesReached {
                                max_raises: MAX_RAISES_PER_ROUND,
                            });
                        }
                    }
                    BettingStructure::NoLimit => {}
                }

                let actual = self.players[self.current_player].place_bet(stack);
                self.pot.add_bet(self.current_player, actual);

                let new_total = self.players[self.current_player].current_bet;
                if new_total > self.current_bet {
                    // min_raise is the raise size, but never below big_blind
                    self.min_raise = (new_total - self.current_bet).max(self.big_blind);
                    self.current_bet = new_total;
                    self.raises_this_round += 1;

                    // Reset has_acted for all other players since there's a new bet
                    for (idx, player) in self.players.iter_mut().enumerate() {
                        if idx != self.current_player {
                            player.has_acted_this_round = false;
                        }
                    }
                }
            }
            PlayerAction::ShowCards(card_indices) => {
                // ShowCards is handled via handle_show_cards, not through handle_action
                return self.handle_show_cards(user_id, card_indices);
            }
        }

        // Record last action for display
        self.players[self.current_player].last_action = Some(match &action {
            PlayerAction::Fold => "Fold".to_string(),
            PlayerAction::Check => "Check".to_string(),
            PlayerAction::Call => {
                let bet = self.players[self.current_player].current_bet;
                format!("Call ${}", bet)
            }
            PlayerAction::Raise(amt) => format!("Raise ${}", amt),
            PlayerAction::AllIn => {
                let bet = self.players[self.current_player].current_bet;
                format!("All In ${}", bet)
            }
            PlayerAction::ShowCards(_) => "Show Cards".to_string(),
        });

        // Mark player as having acted
        self.players[self.current_player].has_acted_this_round = true;

        tracing::info!(
            "Player {} acted, advancing. Current player before: {}",
            self.players[self.current_player].username,
            self.current_player
        );

        // Move to next player or next phase
        self.advance_action_with_bot_pacing(acted_by_bot);

        tracing::info!(
            "After advance: current_player={}, state={:?}",
            self.current_player,
            if self.current_player < self.players.len() {
                format!("{:?}", self.players[self.current_player].state)
            } else {
                "INVALID INDEX".to_string()
            }
        );

        Ok(())
    }

    pub fn handle_show_cards(&mut self, user_id: &str, card_indices: Vec<usize>) -> GameResult<()> {
        // Must be in Showdown phase with won_without_showdown
        if self.phase != GamePhase::Showdown || !self.won_without_showdown {
            return Err(GameError::InvalidAction {
                reason: "Can only show cards after winning without showdown".to_string(),
            });
        }

        // Bots cannot show cards
        if user_id.starts_with("bot_") {
            return Err(GameError::InvalidAction {
                reason: "Bots cannot show cards".to_string(),
            });
        }

        // Find the player
        let player = self.players.iter_mut().find(|p| p.user_id == user_id);
        let player = match player {
            Some(p) => p,
            None => {
                return Err(GameError::PlayerNotFound {
                    user_id: user_id.to_string(),
                })
            }
        };

        // Must be the winner
        if !player.is_winner {
            return Err(GameError::InvalidAction {
                reason: "Only the winner can show cards".to_string(),
            });
        }

        // Validate and set shown cards
        for &idx in &card_indices {
            if idx >= player.hole_cards.len() {
                return Err(GameError::InvalidAction {
                    reason: format!("Invalid card index: {}", idx),
                });
            }
            if idx < player.shown_cards.len() {
                player.shown_cards[idx] = true;
            }
        }

        Ok(())
    }

    pub(crate) fn advance_action(&mut self) {
        self.advance_action_with_bot_pacing(false);
    }

    fn advance_action_with_bot_pacing(&mut self, acted_by_bot: bool) {
        // If everyone but one player has folded, award the pot immediately.
        // This covers cases like "folds to the big blind" where the last
        // remaining player may not have acted in the round yet.
        let active_in_hand_count = self
            .players
            .iter()
            .filter(|p| p.is_active_in_hand())
            .count();
        if active_in_hand_count == 1 {
            self.advance_phase();
            return;
        }

        // Check if betting round is complete
        if self.is_betting_round_complete() {
            if acted_by_bot {
                // Preserve call/check chips and action badges briefly for clients
                // before advancing the street/showdown.
                self.last_phase_change_time = Some(current_timestamp_ms());
            } else {
                self.advance_phase();
            }
        } else {
            self.current_player = self.next_active_player(self.current_player);

            // Auto-fold sitting out players in tournaments
            if self.format.eliminates_players() {
                let current = &self.players[self.current_player];
                if current.state == PlayerState::SittingOut {
                    tracing::info!("Auto-folding sitting out player: {}", current.username);
                    self.players[self.current_player].fold();
                    self.players[self.current_player].has_acted_this_round = true;
                    self.players[self.current_player].last_action =
                        Some("Auto-Fold (Sitting Out)".to_string());
                    // Recursively advance to next player
                    self.advance_action_with_bot_pacing(false);
                }
            }

            // Auto-fold disconnected players (in any game format)
            {
                let current = &self.players[self.current_player];
                if current.state == PlayerState::Disconnected {
                    tracing::info!("Auto-folding disconnected player: {}", current.username);
                    self.players[self.current_player].fold();
                    self.players[self.current_player].has_acted_this_round = true;
                    self.players[self.current_player].last_action =
                        Some("Auto-Fold (Disconnected)".to_string());
                    self.advance_action_with_bot_pacing(false);
                }
            }
        }
    }

    pub(crate) fn is_betting_round_complete(&self) -> bool {
        let active_players: Vec<&Player> = self.players.iter().filter(|p| p.can_act()).collect();

        if active_players.is_empty() {
            return true;
        }

        // Need at least 2 checks:
        // 1. All active players have acted this round
        // 2. All active players have matched the current bet

        let all_acted = active_players.iter().all(|p| p.has_acted_this_round);
        let all_matched = active_players
            .iter()
            .all(|p| p.current_bet == self.current_bet);

        // Debug logging
        tracing::debug!(
            "Betting round check: phase={:?}, current_bet={}, active_players={}, all_acted={}, all_matched={}",
            self.phase, self.current_bet, active_players.len(), all_acted, all_matched
        );
        for (i, p) in self.players.iter().enumerate() {
            tracing::debug!(
                "  Player {}: {}, bet={}, acted={}, state={:?}",
                i,
                p.username,
                p.current_bet,
                p.has_acted_this_round,
                p.state
            );
        }

        all_acted && all_matched
    }
}
