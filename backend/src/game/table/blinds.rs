use super::*;

impl PokerTable {
    pub(crate) fn post_blinds(&mut self) {
        // Post antes if configured
        if self.ante > 0 {
            for idx in 0..self.players.len() {
                let player = &self.players[idx];
                // Only active players with chips post antes (not eliminated or sitting out without chips)
                if player.stack > 0 && player.state != PlayerState::Eliminated {
                    let ante_amount = self.players[idx].place_bet(self.ante);
                    self.pot.add_bet(idx, ante_amount);
                    tracing::debug!(
                        "Player {} ({}) posted ante of ${}",
                        self.players[idx].username,
                        idx,
                        ante_amount
                    );
                }
            }
        }

        let num_players = self.players.len();

        if num_players == HEADS_UP_PLAYER_COUNT {
            // Heads-up special case: dealer posts SB, other player posts BB
            // Dealer is at self.dealer_seat
            let dealer_idx = self.dealer_seat;
            let other_idx = (dealer_idx + 1) % 2;

            // Dealer posts SB
            let sb_amount = self.players[dealer_idx].place_bet(self.small_blind);
            self.pot.add_bet(dealer_idx, sb_amount);

            // Other player posts BB
            let bb_amount = self.players[other_idx].place_bet(self.big_blind);
            self.pot.add_bet(other_idx, bb_amount);

            self.current_bet = self.big_blind;

            // In heads-up, dealer acts first pre-flop (after posting SB)
            self.current_player = dealer_idx;

            tracing::info!(
                "Heads-up blinds posted: Dealer (SB) at idx {} (seat {}, {}) posted ${}, BB at idx {} (seat {}, {}) posted ${}",
                dealer_idx,
                self.players[dealer_idx].seat,
                self.players[dealer_idx].username,
                sb_amount,
                other_idx,
                self.players[other_idx].seat,
                self.players[other_idx].username,
                bb_amount
            );
        } else {
            // 3+ players: normal blind posting
            // Small blind is the next player after dealer (includes sitting out in tournaments)
            let sb_seat = self.next_player_for_blind(self.dealer_seat);
            tracing::info!(
                "post_blinds: dealer_seat={}, sb_seat={}",
                self.dealer_seat,
                sb_seat
            );
            let sb_amount = self.players[sb_seat].place_bet(self.small_blind);
            self.pot.add_bet(sb_seat, sb_amount);
            // Posting blind does NOT count as acting - player can still raise

            // Big blind is the next player after small blind
            let bb_seat = self.next_player_for_blind(sb_seat);
            tracing::info!("post_blinds: bb_seat={}", bb_seat);
            let bb_amount = self.players[bb_seat].place_bet(self.big_blind);
            self.pot.add_bet(bb_seat, bb_amount);
            // Posting blind does NOT count as acting - BB has option to raise

            self.current_bet = self.big_blind;

            // First to act is after big blind
            self.current_player = self.next_active_player(bb_seat);

            tracing::info!(
                "Blinds posted: Dealer at idx {} (seat {}, {}), SB=${} at idx {} (seat {}, {}), BB=${} at idx {} (seat {}, {})",
                self.dealer_seat,
                self.players[self.dealer_seat].seat,
                self.players[self.dealer_seat].username,
                sb_amount,
                sb_seat,
                self.players[sb_seat].seat,
                self.players[sb_seat].username,
                bb_amount,
                bb_seat,
                self.players[bb_seat].seat,
                self.players[bb_seat].username
            );
        }
    }

    /// Find the next player for blind posting (includes sitting out players in tournaments)
    /// In tournaments: sitting out players still pay blinds
    /// In cash games: sitting out players don't pay blinds
    pub(crate) fn next_player_for_blind(&self, after: usize) -> usize {
        if self.players.is_empty() {
            tracing::warn!("next_player_for_blind called with no players");
            return 0;
        }

        let is_tournament = self.format.eliminates_players();
        let next = self.next_player_index_by_seat(after, |player| {
            if is_tournament {
                player.is_eligible_for_tournament_blind()
            } else {
                player.is_eligible_for_button()
            }
        });

        next.unwrap_or_else(|| after.min(self.players.len() - 1))
    }
}
