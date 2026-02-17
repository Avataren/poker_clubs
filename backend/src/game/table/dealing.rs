use super::*;

impl PokerTable {
    pub fn start_new_hand(&mut self) {
        self.last_winner_message = None;
        self.winning_hand = None;
        self.won_without_showdown = false;
        self.clear_hand_action_history();

        // In cash games, a stand-up requested mid-hand is finalized here.
        if self.format.can_cash_out() {
            self.remove_pending_standups();
        }

        // Remove broke players before starting a new hand (tournament mode)
        self.check_eliminations();

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
            self.pot.reset();
            self.current_bet = 0;
            self.raises_this_round = 0;
            self.actions_this_round = 0;
            self.last_raiser_seat = None;
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
        self.actions_this_round = 0;
        self.last_raiser_seat = None;

        // Move dealer button to next eligible player (skip sitting out/broke players).
        // Only pick "first eligible" on true table startup (no prior phase timestamps).
        // Waiting between hands (e.g. MTT rebalance window) must still rotate dealer.
        let is_initial_hand_start =
            self.phase == GamePhase::Waiting && self.last_phase_change_time.is_none();
        if is_initial_hand_start {
            self.dealer_seat = self.first_eligible_player_for_button();
        } else {
            self.dealer_seat = self.next_eligible_player_for_button(self.dealer_seat);
        }

        // Post blinds (also sets current_player)
        self.post_blinds();

        // Deal hole cards
        self.deal_hole_cards();

        // Set phase - current_player was already set by post_blinds
        self.try_transition(GamePhase::PreFlop);
        self.last_phase_change_time = Some(current_timestamp_ms());
    }

    pub(crate) fn deal_hole_cards(&mut self) {
        let hole_cards_count = self.variant.hole_cards_count();
        if self.players.is_empty() {
            return;
        }

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
                current_seat = self
                    .next_player_index_by_seat(current_seat, |_| true)
                    .unwrap_or(current_seat);
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
