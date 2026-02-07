use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPot {
    pub amount: i64,
    pub eligible_player_seats: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicTableState {
    pub table_id: String,
    pub name: String,
    pub variant_id: String,
    pub variant_name: String,
    pub format_id: String,
    pub format_name: String,
    pub can_cash_out: bool,
    pub can_top_up: bool,
    pub phase: GamePhase,
    pub community_cards: Vec<Card>,
    pub pot_total: i64,
    pub pots: Vec<PublicPot>,
    pub current_bet: i64,
    pub ante: i64,
    pub current_player_seat: usize,
    pub players: Vec<PublicPlayerState>,
    pub max_seats: usize,
    pub last_winner_message: Option<String>,
    pub winning_hand: Option<String>,
    pub dealer_seat: Option<usize>,
    pub small_blind_seat: Option<usize>,
    pub big_blind_seat: Option<usize>,
    // Tournament info (only present for tournament tables)
    pub tournament_id: Option<String>,
    pub tournament_blind_level: Option<i64>,
    pub tournament_small_blind: Option<i64>,
    pub tournament_big_blind: Option<i64>,
    pub tournament_level_start_time: Option<String>,
    pub tournament_level_duration_secs: Option<i64>,
    pub tournament_ante: Option<i64>,
    pub tournament_next_small_blind: Option<i64>,
    pub tournament_next_big_blind: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPlayerState {
    pub user_id: String,
    pub username: String,
    pub seat: usize,
    pub stack: i64,
    pub current_bet: i64,
    pub state: PlayerState,
    pub hole_cards: Option<Vec<Card>>, // Only visible to the player themselves
    pub is_winner: bool,
    pub last_action: Option<String>,
    pub pot_won: i64, // Amount won from pot (for animation)
    pub shown_cards: Option<Vec<bool>>, // Which cards the winner chose to show (fold-win only)
}

/// Tournament info to include in table state
#[derive(Debug, Clone)]
pub struct TournamentInfo {
    pub tournament_id: String,
    pub blind_level: i64,
    pub ante: i64,
    pub level_start_time: Option<String>,
    pub level_duration_secs: i64,
    pub next_small_blind: Option<i64>,
    pub next_big_blind: Option<i64>,
}

impl PokerTable {
    pub fn get_public_state(&self, for_user_id: Option<&str>) -> PublicTableState {
        self.get_public_state_with_tournament(for_user_id, None)
    }

    pub fn get_public_state_with_tournament(
        &self,
        for_user_id: Option<&str>,
        tournament_info: Option<TournamentInfo>,
    ) -> PublicTableState {
        // Get the actual seat number of the current player (not array index)
        // During Waiting phase or invalid states, use the first player's seat if available
        let current_player_seat = if self.phase == GamePhase::Waiting {
            // In waiting phase, no one's turn - use first player's seat as placeholder
            if !self.players.is_empty() {
                self.players[0].seat
            } else {
                0
            }
        } else if self.current_player < self.players.len() {
            self.players[self.current_player].seat
        } else {
            // Fallback: try to find any active player's seat
            self.players
                .iter()
                .find(|p| p.can_act())
                .map(|p| p.seat)
                .unwrap_or(0)
        };

        // Compute pot breakdown for display
        let public_pots = match self.phase {
            GamePhase::Showdown => {
                // During showdown, use already-calculated pots
                self.pot.pots.iter().map(|p| PublicPot {
                    amount: p.amount,
                    eligible_player_seats: p.eligible_players.iter()
                        .filter_map(|&idx| self.players.get(idx).map(|pl| pl.seat))
                        .collect(),
                }).collect()
            }
            GamePhase::Waiting => vec![],
            _ => {
                // Only show split pots when at least one player is all-in;
                // otherwise different bet levels are just mid-round action, not real side pots
                let has_allin = self.players.iter().any(|p| p.state == PlayerState::AllIn);
                if !has_allin {
                    vec![]
                } else {
                    let player_bets: Vec<(usize, i64, bool)> = self
                        .players
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.total_bet_this_hand > 0)
                        .map(|(idx, p)| (idx, p.total_bet_this_hand, p.is_active_in_hand()))
                        .collect();
                    if player_bets.is_empty() {
                        vec![]
                    } else {
                        self.pot.preview_side_pots(&player_bets).iter().map(|p| PublicPot {
                            amount: p.amount,
                            eligible_player_seats: p.eligible_players.iter()
                                .filter_map(|&idx| self.players.get(idx).map(|pl| pl.seat))
                                .collect(),
                        }).collect()
                    }
                }
            }
        };

        PublicTableState {
            table_id: self.table_id.clone(),
            name: self.name.clone(),
            variant_id: self.variant.id().to_string(),
            variant_name: self.variant.name().to_string(),
            phase: self.phase.clone(),
            community_cards: self.community_cards.clone(),
            pot_total: self.pot.total(),
            pots: public_pots,
            current_bet: self.current_bet,
            ante: self.ante,
            current_player_seat,
            players: self
                .players
                .iter()
                .map(|p| {
                    // Check if this is an all-in runout situation (fewer than 2 players can act)
                    // In this case, all active players' cards should be revealed like in real poker
                    let is_allin_runout = self.players.iter().filter(|pl| pl.can_act()).count() < 2
                        && self.players.iter().filter(|pl| pl.is_active_in_hand()).count() >= 2;

                    let hole_cards = if Some(p.user_id.as_str()) == for_user_id {
                        // Show own cards face-up
                        Some(p.hole_cards.clone())
                    } else if self.phase == GamePhase::Showdown && self.won_without_showdown && p.is_winner {
                        // Fold-win: only show cards the winner chose to reveal
                        if p.shown_cards.iter().any(|&s| s) {
                            Some(p.hole_cards.iter().enumerate().map(|(i, card)| {
                                if i < p.shown_cards.len() && p.shown_cards[i] {
                                    card.clone()
                                } else {
                                    Card { rank: 0, suit: 0, highlighted: false, face_up: false }
                                }
                            }).collect())
                        } else {
                            // No cards shown yet - send face-down placeholders
                            let num_cards = p.hole_cards.len();
                            Some(vec![
                                Card { rank: 0, suit: 0, highlighted: false, face_up: false };
                                num_cards
                            ])
                        }
                    } else if (self.phase == GamePhase::Showdown || is_allin_runout) && p.is_active_in_hand() {
                        // During normal showdown OR all-in runout, show all active players' cards face-up
                        Some(p.hole_cards.clone())
                    } else if p.is_active_in_hand() && !p.hole_cards.is_empty() {
                        // For other players still in the hand, send placeholder face-down cards
                        // WITHOUT revealing the actual rank/suit (security: prevent cheating via network inspection)
                        let num_cards = p.hole_cards.len();
                        Some(vec![
                            Card {
                                rank: 0,
                                suit: 0,
                                highlighted: false,
                                face_up: false
                            };
                            num_cards
                        ])
                    } else {
                        // No cards for folded/sitting out players
                        None
                    };

                    let shown_cards_field = if self.won_without_showdown && p.is_winner && !p.shown_cards.is_empty() {
                        Some(p.shown_cards.clone())
                    } else {
                        None
                    };

                    PublicPlayerState {
                        user_id: p.user_id.clone(),
                        username: p.username.clone(),
                        seat: p.seat,
                        stack: p.stack,
                        current_bet: p.current_bet,
                        state: p.state.clone(),
                        hole_cards,
                        is_winner: p.is_winner,
                        last_action: p.last_action.clone(),
                        pot_won: p.pot_won,
                        shown_cards: shown_cards_field,
                    }
                })
                .collect(),
            max_seats: self.max_seats,
            last_winner_message: self.last_winner_message.clone(),
            winning_hand: self.winning_hand.clone(),
            format_id: self.format.format_id().to_string(),
            format_name: self.format.name().to_string(),
            can_cash_out: self.format.can_cash_out(),
            can_top_up: self.format.can_top_up(),
            dealer_seat: if self.phase != GamePhase::Waiting && !self.players.is_empty() {
                Some(self.players[self.dealer_seat].seat)
            } else {
                None
            },
            small_blind_seat: if self.phase != GamePhase::Waiting && self.players.len() >= 2 {
                let sb_idx = self.next_eligible_player_for_button(self.dealer_seat);
                Some(self.players[sb_idx].seat)
            } else {
                None
            },
            big_blind_seat: if self.phase != GamePhase::Waiting && self.players.len() >= 2 {
                let sb_idx = self.next_eligible_player_for_button(self.dealer_seat);
                let bb_idx = self.next_eligible_player_for_button(sb_idx);
                Some(self.players[bb_idx].seat)
            } else {
                None
            },
            tournament_id: tournament_info.as_ref().map(|t| t.tournament_id.clone()),
            tournament_blind_level: tournament_info.as_ref().map(|t| t.blind_level),
            // Show current tournament level blinds (always show what the tournament is at now)
            tournament_small_blind: tournament_info.as_ref().map(|_| self.small_blind),
            tournament_big_blind: tournament_info.as_ref().map(|_| self.big_blind),
            tournament_level_start_time: tournament_info.as_ref().and_then(|t| t.level_start_time.clone()),
            tournament_level_duration_secs: tournament_info.as_ref().map(|t| t.level_duration_secs),
            tournament_ante: tournament_info.as_ref().map(|t| t.ante),
            tournament_next_small_blind: tournament_info.as_ref().and_then(|t| t.next_small_blind),
            tournament_next_big_blind: tournament_info.as_ref().and_then(|t| t.next_big_blind),
        }
    }
}
