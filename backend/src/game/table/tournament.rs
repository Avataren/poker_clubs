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
    /// Blinds are updated immediately but take effect at the start of the next hand
    pub fn update_blinds_and_ante(&mut self, small_blind: i64, big_blind: i64, ante: i64) {
        // Simple snapshot approach: just update the blinds and ante
        // They'll be used at the start of the next hand
        self.small_blind = small_blind;
        self.big_blind = big_blind;
        self.min_raise = big_blind;
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

    /// Check for eliminated players (stack = 0 in tournament mode)
    /// Returns list of user_ids who are eliminated
    /// Eliminated players are removed from the table
    pub fn check_eliminations(&mut self) -> Vec<String> {
        if !self.format.eliminates_players() {
            return vec![];
        }

        // Only eliminate players when no hand is in progress (Waiting phase)
        // This ensures players see the complete hand result (showdown, pot animation)
        // before being removed from the table
        if self.phase != GamePhase::Waiting {
            tracing::debug!(
                "check_eliminations: Skipping - phase is {:?}, not Waiting",
                self.phase
            );
            return vec![];
        }

        tracing::info!("check_eliminations: Processing in Waiting phase");

        let mut eliminated = vec![];

        // Find all players with 0 stack
        let mut to_remove = vec![];
        for (idx, player) in self.players.iter().enumerate() {
            if player.stack == 0 && player.state != PlayerState::Eliminated {
                eliminated.push(player.user_id.clone());
                to_remove.push(idx);
                tracing::info!(
                    "Player {} ({}) eliminated from tournament",
                    player.username,
                    player.user_id
                );
            }
        }

        // Remove eliminated players from the table (in reverse order to preserve indices)
        for &idx in to_remove.iter().rev() {
            self.players.remove(idx);
        }

        // Adjust current_player index if needed
        if !to_remove.is_empty() && !self.players.is_empty() {
            // Ensure current_player is still valid
            if self.current_player >= self.players.len() {
                self.current_player = 0;
            }
        }

        eliminated
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
