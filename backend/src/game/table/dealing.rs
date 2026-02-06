use super::*;

impl PokerTable {
    pub fn start_new_hand(&mut self) {
        self.last_winner_message = None;
        self.winning_hand = None;

        // Reset all players for new hand
        for player in &mut self.players {
            player.reset_for_new_hand();
        }

        // Count players who can actually play (have chips and not eliminated)
        let playable_count = self.players.iter().filter(|p| p.stack > 0).count();

        if playable_count < MIN_PLAYERS_TO_START {
            // Force to Waiting regardless of current phase (may be called from Showdown)
            self.phase = GamePhase::Waiting;
            self.community_cards.clear();
            tracing::info!("Not enough players with chips to start hand, going to Waiting phase");
            return;
        }

        // Clear community cards and reset game state
        self.community_cards.clear();
        self.pot.reset();
        self.deck.reset_and_shuffle();
        self.current_bet = 0;
        self.min_raise = self.big_blind;
        self.raises_this_round = 0;

        // Move dealer button to next eligible player (skip sitting out/broke players)
        // For the first hand from Waiting phase, find the first eligible player (not the next one)
        if self.phase == GamePhase::Waiting {
            self.dealer_seat = self.first_eligible_player_for_button();
        } else {
            self.dealer_seat = self.next_eligible_player_for_button(self.dealer_seat);
        }

        // Post blinds (also sets current_player)
        self.post_blinds();

        // Deal hole cards
        self.deal_hole_cards();

        // Set phase - current_player was already set by post_blinds
        if let Ok(next) = self.phase.transition_to(GamePhase::PreFlop) {
            self.phase = next;
        }
    }

    pub(crate) fn deal_hole_cards(&mut self) {
        let hole_cards_count = self.variant.hole_cards_count();

        // Calculate where dealing starts: small blind (first player after dealer)
        let sb_seat = self.next_eligible_player_for_button(self.dealer_seat);

        // Deal one card at a time to each player, just like real poker
        // Start from small blind and go around the table for each round
        for round in 0..hole_cards_count {
            let mut current_seat = sb_seat;
            for _ in 0..self.players.len() {
                if self.players[current_seat].can_act() {
                    if let Some(card) = self.deck.deal() {
                        self.players[current_seat].deal_cards(vec![card]);
                    }
                }
                current_seat = (current_seat + 1) % self.players.len();
            }
            tracing::debug!(
                "Dealt card round {} of {}, starting from seat {}",
                round + 1,
                hole_cards_count,
                sb_seat
            );
        }
    }
}
