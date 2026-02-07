use super::*;

impl PokerTable {
    /// Set tournament blinds for this hand (called at start of hand)
    /// Takes a snapshot of the current tournament level blinds
    // ==================== Tournament-Specific Methods ====================

    /// Update blinds and minimum raise (called when tournament blind level advances)
    /// Blinds are updated immediately but take effect at the start of the next hand
    pub fn update_blinds(&mut self, small_blind: i64, big_blind: i64) {
        self.update_blinds_and_ante(small_blind, big_blind, self.ante);
    }

    /// Update blinds, ante, and minimum raise (called when tournament blind level advances)
    /// Blinds are updated immediately but take effect at the start of the next hand.
    /// Note: min_raise is NOT reset here to avoid corrupting a hand in progress.
    /// It is reset to big_blind in start_new_hand() via dealing.rs.
    pub fn update_blinds_and_ante(&mut self, small_blind: i64, big_blind: i64, ante: i64) {
        self.small_blind = small_blind;
        self.big_blind = big_blind;
        self.ante = ante;
        tracing::info!(
            "Table {} blinds updated to {}/{} ante {} (will take effect at next hand)",
            self.table_id,
            small_blind,
            big_blind,
            ante
        );
    }

    /// Apply pending blinds if any (called at start of new hand)
    /// Apply antes from all active players (called at start of hand in tournament mode)
    pub fn apply_antes(&mut self, ante: i64) -> GameResult<()> {
        if ante == 0 {
            return Ok(());
        }

        for (seat, player) in self.players.iter_mut().enumerate() {
            if player.can_act() || player.state == PlayerState::WaitingForHand {
                let actual_ante = player.place_bet(ante);
                self.pot.add_bet(seat, actual_ante);
                tracing::debug!(
                    "Player {} posted ante: ${} (requested: ${})",
                    player.username,
                    actual_ante,
                    ante
                );
            }
        }

        Ok(())
    }

    /// Check for eliminated players (stack = 0 in tournament mode).
    /// Removes them from the table and appends their user_ids to `pending_eliminations`.
    /// Called at the start of each new hand and when transitioning to Waiting.
    pub fn check_eliminations(&mut self) -> Vec<String> {
        if !self.format.eliminates_players() {
            return vec![];
        }

        let mut eliminated = vec![];

        // Find all players with 0 stack
        let mut to_remove = vec![];
        for (idx, player) in self.players.iter().enumerate() {
            if player.stack == 0 && player.state != PlayerState::Eliminated {
                eliminated.push(player.user_id.clone());
                to_remove.push(idx);
                tracing::info!(
                    "Player {} ({}) eliminated from tournament (seat {})",
                    player.username,
                    player.user_id,
                    player.seat
                );
            }
        }

        if to_remove.is_empty() {
            return vec![];
        }

        // Remove eliminated players from the table (in reverse order to preserve indices)
        for &idx in to_remove.iter().rev() {
            self.players.remove(idx);

            // Adjust dealer_seat for the removal
            if !self.players.is_empty() {
                if self.dealer_seat >= self.players.len() {
                    self.dealer_seat = self.players.len() - 1;
                } else if idx < self.dealer_seat {
                    self.dealer_seat -= 1;
                }
            }
        }

        // Ensure current_player is still valid
        if !self.players.is_empty() {
            if self.current_player >= self.players.len() {
                self.current_player = 0;
            }
        }

        // Buffer eliminations for the background task to drain
        self.pending_eliminations.extend(eliminated.iter().cloned());

        eliminated
    }

    /// Drain and return buffered eliminations (consumed by the tournament lifecycle task).
    pub fn drain_pending_eliminations(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_eliminations)
    }

    /// Check if tournament is finished (1 or fewer players remaining)
    pub fn tournament_finished(&self) -> bool {
        if !self.format.eliminates_players() {
            return false;
        }

        // Since eliminated players are removed, just count remaining players with chips
        let active_count = self.players.iter().filter(|p| p.stack > 0).count();

        active_count <= 1
    }

    /// Get remaining tournament players (not eliminated)
    /// Note: eliminated players are removed from the table, so this returns all players
    pub fn get_remaining_players(&self) -> Vec<String> {
        self.players
            .iter()
            .filter(|p| p.stack > 0)
            .map(|p| p.user_id.clone())
            .collect()
    }
}
