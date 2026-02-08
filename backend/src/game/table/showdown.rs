use super::*;

impl PokerTable {
    pub(crate) fn resolve_showdown(&mut self) {
        // Reset all highlighted cards and winner flags
        for card in &mut self.community_cards {
            card.highlighted = false;
        }
        for player in &mut self.players {
            for card in &mut player.hole_cards {
                card.highlighted = false;
            }
            player.is_winner = false;
        }

        // Evaluate hi hands for active players
        let mut hands: Vec<(usize, HandRank)> = Vec::new();

        for (idx, player) in self.players.iter().enumerate() {
            if player.is_active_in_hand() {
                let hand_rank = self
                    .variant
                    .evaluate_hand(&player.hole_cards, &self.community_cards);
                hands.push((idx, hand_rank));
            }
        }

        // Evaluate low hands if this is a hi-lo variant
        let is_hilo = self.variant.is_hi_lo();
        let low_hands: Vec<(usize, LowHandRank)> = if is_hilo {
            self.players
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_active_in_hand())
                .filter_map(|(idx, player)| {
                    self.variant
                        .evaluate_low_hand(&player.hole_cards, &self.community_cards)
                        .map(|low| (idx, low))
                })
                .collect()
        } else {
            Vec::new()
        };

        // Set winning hand description from overall best hand (for UI display)
        let winner_indices = determine_winners(hands.clone());
        if let Some(&first_winner_idx) = winner_indices.first() {
            if let Some((_, winning_hand_rank)) =
                hands.iter().find(|(idx, _)| *idx == first_winner_idx)
            {
                let mut desc = winning_hand_rank.description.clone();
                if is_hilo && !low_hands.is_empty() {
                    desc.push_str(" (Hi)");
                }
                self.winning_hand = Some(desc);
            }
        }

        // Calculate side pots based on each player's total contribution
        let player_bets: Vec<(usize, i64, bool)> = self
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.total_bet_this_hand > 0)
            .map(|(idx, p)| (idx, p.total_bet_this_hand, p.is_active_in_hand()))
            .collect();
        let uncontested = self.pot.calculate_side_pots(&player_bets);

        // Return uncontested amounts (e.g. uncalled overbets) immediately.
        // These are not showdown wins and should not mark winners.
        for (player_idx, amount) in uncontested {
            self.players[player_idx].add_chips(amount);
        }

        // Award pots -- hi-lo or standard
        let payouts = if is_hilo {
            // Determine hi and lo winners for each pot separately
            let mut hi_winners_by_pot = Vec::new();
            let mut lo_winners_by_pot = Vec::new();

            for pot in &self.pot.pots {
                // Hi winners for this pot
                let eligible_hands: Vec<(usize, HandRank)> = hands
                    .iter()
                    .filter(|(idx, _)| pot.eligible_players.contains(idx))
                    .cloned()
                    .collect();
                hi_winners_by_pot.push(determine_winners(eligible_hands));

                // Lo winners for this pot (best low hand among eligible)
                let eligible_lows: Vec<&(usize, LowHandRank)> = low_hands
                    .iter()
                    .filter(|(idx, _)| pot.eligible_players.contains(idx))
                    .collect();

                if eligible_lows.is_empty() {
                    lo_winners_by_pot.push(Vec::new());
                } else {
                    // Find the best (lowest) low hand
                    let best_low = eligible_lows.iter().map(|(_, low)| low).min().unwrap();
                    let lo_winners: Vec<usize> = eligible_lows
                        .iter()
                        .filter(|(_, low)| low == best_low)
                        .map(|(idx, _)| *idx)
                        .collect();
                    lo_winners_by_pot.push(lo_winners);
                }
            }

            self.pot
                .award_pots_hilo(hi_winners_by_pot, lo_winners_by_pot)
        } else {
            // Standard (non-hi-lo) pot awarding
            let mut winners_by_pot = Vec::new();
            for pot in &self.pot.pots {
                let eligible_hands: Vec<(usize, HandRank)> = hands
                    .iter()
                    .filter(|(idx, _)| pot.eligible_players.contains(idx))
                    .cloned()
                    .collect();
                winners_by_pot.push(determine_winners(eligible_hands));
            }
            self.pot.award_pots(winners_by_pot)
        };

        // Mark all payout recipients as winners and highlight their best 5-card hand.
        // This ensures side-pot winners also get highlighted, not just the overall best hand.
        let mut winner_names = Vec::new();
        for (player_idx, amount) in &payouts {
            self.players[*player_idx].add_chips(*amount);
            self.players[*player_idx].is_winner = true;
            self.players[*player_idx].pot_won = *amount;
            winner_names.push(format!(
                "{} wins ${}",
                self.players[*player_idx].username, amount
            ));

            // Highlight the best 5-card hand for this winner
            if let Some((_, hand_rank)) = hands.iter().find(|(idx, _)| *idx == *player_idx) {
                let best_cards = &hand_rank.best_cards;

                tracing::info!(
                    "Winner {} ({}) best cards: {:?}",
                    self.players[*player_idx].username,
                    player_idx,
                    best_cards
                );

                for community_card in &mut self.community_cards {
                    if best_cards
                        .iter()
                        .any(|c| c.rank == community_card.rank && c.suit == community_card.suit)
                    {
                        community_card.highlighted = true;
                    }
                }

                if let Some(winner) = self.players.get_mut(*player_idx) {
                    for hole_card in &mut winner.hole_cards {
                        if best_cards
                            .iter()
                            .any(|c| c.rank == hole_card.rank && c.suit == hole_card.suit)
                        {
                            hole_card.highlighted = true;
                        }
                    }
                }
            }
        }

        // DON'T reset pot here - keep it visible for animation
        // It will be reset when start_new_hand() is called
        self.last_winner_message = Some(winner_names.join(", "));

        // Record showdown time for delay before new hand
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_phase_change_time = Some(now);

        tracing::info!(
            "Showdown complete{}. Winner: {}. Will auto-advance after delay.",
            if is_hilo { " (Hi-Lo)" } else { "" },
            winner_names.join(", ")
        );
    }
}
