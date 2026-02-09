use super::*;

impl PokerTable {
    pub fn add_player(
        &mut self,
        user_id: String,
        username: String,
        buyin: i64,
    ) -> GameResult<usize> {
        // Check if player is already at the table and not broke
        if let Some(existing) = self.players.iter().find(|p| p.user_id == user_id) {
            // If player is sitting out with no chips (broke), remove them so they can rebuy
            if existing.state == PlayerState::SittingOut && existing.stack == 0 {
                self.remove_player(&user_id);
            } else {
                return Err(GameError::PlayerAlreadySeated);
            }
        }

        if self.players.len() >= self.max_seats {
            return Err(GameError::TableFull);
        }

        // Find the next available seat number
        let mut occupied = vec![false; self.max_seats];
        for p in &self.players {
            if p.seat < self.max_seats {
                occupied[p.seat] = true;
            }
        }
        let seat = occupied
            .iter()
            .position(|taken| !*taken)
            .ok_or(GameError::TableFull)?;
        let mut player = Player::new(user_id, username, seat, buyin);

        // If a hand is in progress, make player wait until next hand
        if self.phase != GamePhase::Waiting {
            player.state = PlayerState::WaitingForHand;
            tracing::debug!(
                "Player {} joining mid-hand, setting to WaitingForHand",
                player.username
            );
        }

        self.players.push(player);

        // Start game if we have enough players AND the format allows auto-start
        // Cash games: auto-start with 2+ players
        // Tournaments (SNG/MTT): do NOT auto-start, wait for explicit start
        if self.players.len() >= MIN_PLAYERS_TO_START
            && self.phase == GamePhase::Waiting
            && self.format.should_auto_start()
        {
            self.start_new_hand();
        }

        Ok(seat)
    }

    pub fn remove_player(&mut self, user_id: &str) {
        // Find the index of the player being removed
        let removed_idx = self.players.iter().position(|p| p.user_id == user_id);

        self.players.retain(|p| p.user_id != user_id);

        // Seats are preserved — players keep their physical seat numbers.

        // Adjust dealer_seat and current_player vec indices after removal
        if let Some(removed_idx) = removed_idx {
            if !self.players.is_empty() {
                // If dealer was removed or is now out of bounds, move it to a valid position
                if self.dealer_seat >= self.players.len() || self.dealer_seat == removed_idx {
                    self.dealer_seat = if self.dealer_seat > 0 {
                        (self.dealer_seat - 1).min(self.players.len() - 1)
                    } else {
                        0
                    };
                } else if removed_idx < self.dealer_seat {
                    // Player before dealer was removed, adjust dealer index
                    self.dealer_seat -= 1;
                }

                // Same for current_player
                if self.current_player >= self.players.len() || self.current_player == removed_idx {
                    self.current_player = if self.current_player > 0 {
                        (self.current_player - 1).min(self.players.len() - 1)
                    } else {
                        0
                    };
                } else if removed_idx < self.current_player {
                    // Player before current was removed, adjust current index
                    self.current_player -= 1;
                }
            }
        }
    }

    pub fn take_seat(
        &mut self,
        user_id: String,
        username: String,
        seat: usize,
        buyin: i64,
    ) -> GameResult<usize> {
        // Check if player is already at the table
        if let Some(existing) = self.players.iter().find(|p| p.user_id == user_id) {
            // If player is sitting out with no chips (broke), remove them so they can rebuy
            if existing.state == PlayerState::SittingOut && existing.stack == 0 {
                self.players.retain(|p| p.user_id != user_id);
            } else {
                return Err(GameError::PlayerAlreadySeated);
            }
        }

        // Validate seat number
        if seat >= self.max_seats {
            return Err(GameError::InvalidSeat {
                seat,
                max_seats: self.max_seats,
            });
        }

        // Check if seat is occupied
        if self.players.iter().any(|p| p.seat == seat) {
            return Err(GameError::SeatOccupied { seat });
        }

        let mut player = Player::new(user_id, username, seat, buyin);

        // If a hand is in progress, make player wait until next hand
        if self.phase != GamePhase::Waiting {
            player.state = PlayerState::WaitingForHand;
            tracing::debug!(
                "Player {} joining mid-hand, setting to WaitingForHand",
                player.username
            );
        }

        self.players.push(player);

        // Start game if we have enough active players AND the format allows auto-start
        // Cash games: auto-start with 2+ players
        // Tournaments (SNG/MTT): do NOT auto-start, wait for explicit start
        if self.active_players_count() >= MIN_PLAYERS_TO_START
            && self.phase == GamePhase::Waiting
            && self.format.should_auto_start()
        {
            self.start_new_hand();
        }

        Ok(seat)
    }

    pub fn stand_up(&mut self, user_id: &str) -> GameResult<()> {
        let player_idx = self
            .players
            .iter()
            .position(|p| p.user_id == user_id)
            .ok_or(GameError::PlayerNotAtTable)?;

        let should_defer_standup = self.phase != GamePhase::Waiting
            && (self.players[player_idx].state == PlayerState::Active
                || self.players[player_idx].state == PlayerState::AllIn);
        let was_current_turn = player_idx == self.current_player;

        // If player is in an active hand, mark them to stand up after hand concludes.
        if should_defer_standup {
            self.players[player_idx].state = PlayerState::SittingOut;
            self.players[player_idx].last_action = Some("Stand Up".to_string());
            self.players[player_idx].has_acted_this_round = true;
            tracing::debug!(
                "Player {} will stand up after current hand",
                self.players[player_idx].username
            );

            // If they stand up on their betting turn, continue the hand immediately.
            if was_current_turn
                && matches!(
                    self.phase,
                    GamePhase::PreFlop | GamePhase::Flop | GamePhase::Turn | GamePhase::River
                )
            {
                self.advance_action();
            }
            return Ok(());
        }

        // Remove immediately and keep table indices consistent.
        let username = self.players[player_idx].username.clone();
        self.remove_player(user_id);
        tracing::debug!("Player {} stood up immediately", username);
        Ok(())
    }

    pub fn top_up(&mut self, user_id: &str, amount: i64) -> GameResult<()> {
        // Find the player
        let player = self
            .players
            .iter_mut()
            .find(|p| p.user_id == user_id)
            .ok_or(GameError::PlayerNotAtTable)?;

        // Validate top-up amount
        if amount <= 0 {
            return Err(GameError::InvalidAction {
                reason: "Top-up amount must be positive".to_string(),
            });
        }

        // Check if player can top up during a hand
        if self.phase != GamePhase::Waiting && player.state != PlayerState::SittingOut {
            return Err(GameError::GameInProgress);
        }

        // Add chips to player's stack
        player.stack += amount;

        // If player was sitting out due to being broke, set them to WaitingForHand
        if player.state == PlayerState::SittingOut && player.stack > 0 {
            player.state = PlayerState::WaitingForHand;
            tracing::info!(
                "Player {} topped up ${} and is now waiting for next hand",
                player.username,
                amount
            );
        }

        // If we're in Waiting phase, check if we can now start a new hand
        if self.phase == GamePhase::Waiting {
            let playable_count = self.players.iter().filter(|p| p.stack > 0).count();

            if playable_count >= MIN_PLAYERS_TO_START {
                tracing::info!("Enough players with chips after top-up, starting new hand");
                self.start_new_hand();
            }
        }

        Ok(())
    }
}
