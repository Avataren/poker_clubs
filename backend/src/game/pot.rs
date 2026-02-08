use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pot {
    pub amount: i64,
    pub eligible_players: Vec<usize>, // Player indices eligible for this pot
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotManager {
    pub pots: Vec<Pot>,
    pub current_bets: HashMap<usize, i64>, // Player index -> amount bet this round
}

impl PotManager {
    pub fn new() -> Self {
        Self {
            pots: vec![Pot {
                amount: 0,
                eligible_players: vec![],
            }],
            current_bets: HashMap::new(),
        }
    }

    /// Add a bet from a player
    pub fn add_bet(&mut self, player_idx: usize, amount: i64) {
        *self.current_bets.entry(player_idx).or_insert(0) += amount;
        self.pots[0].amount += amount;

        // Ensure player is in eligible players list
        if !self.pots[0].eligible_players.contains(&player_idx) {
            self.pots[0].eligible_players.push(player_idx);
        }
    }

    /// Build contested side pots from player bets.
    ///
    /// Returns:
    /// - Pots that have at least two eligible players (actually contested)
    /// - Uncontested amounts to immediately return to a single eligible player
    fn build_side_pots(player_bets: &[(usize, i64, bool)]) -> (Vec<Pot>, HashMap<usize, i64>) {
        let mut sorted: Vec<(usize, i64, bool)> = player_bets.to_vec();
        sorted.sort_by_key(|(_, bet, _)| *bet);

        let mut raw_pots = Vec::new();
        let mut uncontested = HashMap::new();
        let mut prev_level = 0i64;

        for i in 0..sorted.len() {
            let (_, bet_level, _) = sorted[i];
            if bet_level <= prev_level {
                continue;
            }

            let level_contribution = bet_level - prev_level;
            let contributors = sorted
                .iter()
                .filter(|(_, bet, _)| *bet > prev_level)
                .count();
            let pot_amount = level_contribution * contributors as i64;

            let eligible: Vec<usize> = sorted
                .iter()
                .filter(|(_, bet, active)| *bet >= bet_level && *active)
                .map(|(idx, _, _)| *idx)
                .collect();

            if pot_amount > 0 {
                // A "pot" with one (or zero) eligible players is uncontested.
                // In real poker this amount is not a split pot; it is returned/pushed
                // to the lone eligible player.
                if eligible.len() <= 1 {
                    if let Some(&player_idx) = eligible.first() {
                        *uncontested.entry(player_idx).or_insert(0) += pot_amount;
                    } else if contributors == 1 {
                        // Safety fallback: if only one contributor exists and no eligible
                        // player was marked active, return it to that contributor.
                        if let Some((player_idx, _, _)) =
                            sorted.iter().find(|(_, bet, _)| *bet > prev_level)
                        {
                            *uncontested.entry(*player_idx).or_insert(0) += pot_amount;
                        }
                    }
                } else {
                    raw_pots.push(Pot {
                        amount: pot_amount,
                        eligible_players: eligible,
                    });
                }
            }

            prev_level = bet_level;
        }

        // Merge consecutive pots that have the same eligible players.
        // Folded players create bet levels that split pots unnecessarily
        // since they can't win — merge those back together.
        let mut merged = Vec::new();
        for pot in raw_pots {
            if let Some(last) = merged.last_mut() {
                let last: &mut Pot = last;
                if last.eligible_players == pot.eligible_players {
                    last.amount += pot.amount;
                    continue;
                }
            }
            merged.push(pot);
        }

        (merged, uncontested)
    }

    /// Calculate side pots based on each player's total contribution to the hand.
    /// `player_bets` contains (player_idx, total_bet_this_hand, is_active_in_hand).
    /// All contributing players (including folded) affect pot sizes, but only
    /// active-in-hand players are eligible to win.
    ///
    /// Returns uncontested amounts that should be immediately pushed back to
    /// the sole eligible player (not treated as split pots).
    pub fn calculate_side_pots(
        &mut self,
        player_bets: &[(usize, i64, bool)],
    ) -> HashMap<usize, i64> {
        if player_bets.is_empty() {
            return HashMap::new();
        }
        let (new_pots, uncontested) = Self::build_side_pots(player_bets);
        self.pots = if new_pots.is_empty() {
            vec![Pot {
                amount: 0,
                eligible_players: vec![],
            }]
        } else {
            new_pots
        };
        uncontested
    }

    /// End the current betting round
    pub fn end_betting_round(&mut self) {
        self.current_bets.clear();
    }

    /// Get the total pot amount
    pub fn total(&self) -> i64 {
        self.pots.iter().map(|p| p.amount).sum()
    }

    /// Award pots to winners
    /// Returns map of player_idx -> amount won
    pub fn award_pots(&self, winners_by_pot: Vec<Vec<usize>>) -> HashMap<usize, i64> {
        let mut payouts = HashMap::new();

        for (pot_idx, pot) in self.pots.iter().enumerate() {
            if pot_idx >= winners_by_pot.len() {
                break;
            }

            let winners = &winners_by_pot[pot_idx];
            if winners.is_empty() {
                continue;
            }

            // Split pot evenly among winners
            let share = pot.amount / winners.len() as i64;
            let remainder = pot.amount % winners.len() as i64;

            for (i, winner_idx) in winners.iter().enumerate() {
                let amount = if i == 0 {
                    share + remainder // First winner gets the odd chips
                } else {
                    share
                };
                *payouts.entry(*winner_idx).or_insert(0) += amount;
            }
        }

        payouts
    }

    /// Award pots in a hi-lo game.
    /// For each pot, the hi winners and lo winners each get half.
    /// If there are no qualifying lo winners for a pot, hi winners get the entire pot.
    /// Odd chip goes to hi winner (standard rule).
    pub fn award_pots_hilo(
        &self,
        hi_winners_by_pot: Vec<Vec<usize>>,
        lo_winners_by_pot: Vec<Vec<usize>>,
    ) -> HashMap<usize, i64> {
        let mut payouts = HashMap::new();

        for (pot_idx, pot) in self.pots.iter().enumerate() {
            let hi_winners = hi_winners_by_pot.get(pot_idx).cloned().unwrap_or_default();
            let lo_winners = lo_winners_by_pot.get(pot_idx).cloned().unwrap_or_default();

            if hi_winners.is_empty() {
                continue;
            }

            if lo_winners.is_empty() {
                // No qualifying low -- entire pot to hi winners
                let share = pot.amount / hi_winners.len() as i64;
                let remainder = pot.amount % hi_winners.len() as i64;
                for (i, &winner_idx) in hi_winners.iter().enumerate() {
                    let amount = if i == 0 { share + remainder } else { share };
                    *payouts.entry(winner_idx).or_insert(0) += amount;
                }
            } else {
                // Split pot: hi gets half, lo gets half. Odd chip to hi.
                let hi_half = (pot.amount + 1) / 2; // ceiling division -- odd chip to hi
                let lo_half = pot.amount - hi_half;

                // Distribute hi half
                if !hi_winners.is_empty() {
                    let share = hi_half / hi_winners.len() as i64;
                    let remainder = hi_half % hi_winners.len() as i64;
                    for (i, &winner_idx) in hi_winners.iter().enumerate() {
                        let amount = if i == 0 { share + remainder } else { share };
                        *payouts.entry(winner_idx).or_insert(0) += amount;
                    }
                }

                // Distribute lo half
                if lo_half > 0 {
                    let share = lo_half / lo_winners.len() as i64;
                    let remainder = lo_half % lo_winners.len() as i64;
                    for (i, &winner_idx) in lo_winners.iter().enumerate() {
                        let amount = if i == 0 { share + remainder } else { share };
                        *payouts.entry(winner_idx).or_insert(0) += amount;
                    }
                }
            }
        }

        payouts
    }

    /// Compute side pot breakdown without mutating state.
    /// Same algorithm as `calculate_side_pots` but returns the result instead.
    pub fn preview_side_pots(&self, player_bets: &[(usize, i64, bool)]) -> Vec<Pot> {
        if player_bets.is_empty() {
            return self.pots.clone();
        }
        let (result, _) = Self::build_side_pots(player_bets);
        result
    }

    /// Reset for a new hand
    pub fn reset(&mut self) {
        self.pots = vec![Pot {
            amount: 0,
            eligible_players: vec![],
        }];
        self.current_bets.clear();
    }
}

impl Default for PotManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pot() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.add_bet(0, 100);
        pot_mgr.add_bet(1, 100);
        pot_mgr.add_bet(2, 100);

        assert_eq!(pot_mgr.total(), 300);
        assert_eq!(pot_mgr.pots.len(), 1);
    }

    #[test]
    fn test_side_pots() {
        let mut pot_mgr = PotManager::new();

        // Player 0 bets 50 (all-in)
        // Player 1 bets 100
        // Player 2 bets 100
        pot_mgr.add_bet(0, 50);
        pot_mgr.add_bet(1, 100);
        pot_mgr.add_bet(2, 100);

        // (player_idx, total_bet, is_active_in_hand)
        let player_bets = vec![(0, 50, true), (1, 100, true), (2, 100, true)];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Should have 2 pots:
        // Main pot: 150 (50 from each of 3 players)
        // Side pot: 100 (50 more from players 1 and 2)
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].amount, 150);
        assert_eq!(pot_mgr.pots[1].amount, 100);
        assert!(uncontested.is_empty());
    }

    #[test]
    fn test_side_pots_short_stack_wins() {
        let mut pot_mgr = PotManager::new();

        // Player 0 goes all-in with 5000
        // Player 1 goes all-in with 10000
        pot_mgr.add_bet(0, 5000);
        pot_mgr.add_bet(1, 10000);

        let player_bets = vec![(0, 5000, true), (1, 10000, true)];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Main pot: 10000 (5000 from each), eligible: [0, 1]
        // Remaining 5000 from player 1 is uncontested and returned.
        assert_eq!(pot_mgr.pots.len(), 1);
        assert_eq!(pot_mgr.pots[0].amount, 10000);
        assert!(pot_mgr.pots[0].eligible_players.contains(&0));
        assert!(pot_mgr.pots[0].eligible_players.contains(&1));
        assert_eq!(uncontested.get(&1), Some(&5000));

        // If player 0 wins: gets main pot (10000), player 1 gets uncontested 5000 back
        let payouts = pot_mgr.award_pots(vec![vec![0]]);
        assert_eq!(payouts.get(&0), Some(&10000));
        assert_eq!(payouts.get(&1), None);
    }

    #[test]
    fn test_side_pots_folded_player_contribution() {
        let mut pot_mgr = PotManager::new();

        // Player 0 bets 100 then folds
        // Player 1 goes all-in with 200
        // Player 2 calls 200
        pot_mgr.add_bet(0, 100);
        pot_mgr.add_bet(1, 200);
        pot_mgr.add_bet(2, 200);

        // Player 0 folded (active=false), but their 100 is still in the pot
        let player_bets = vec![(0, 100, false), (1, 200, true), (2, 200, true)];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Both levels have the same eligible players [1, 2] (player 0 folded),
        // so they merge into a single pot: 300 + 200 = 500
        assert_eq!(pot_mgr.pots.len(), 1);
        assert_eq!(pot_mgr.pots[0].amount, 500);
        assert!(!pot_mgr.pots[0].eligible_players.contains(&0));
        assert!(pot_mgr.pots[0].eligible_players.contains(&1));
        assert!(pot_mgr.pots[0].eligible_players.contains(&2));
        assert!(uncontested.is_empty());
    }

    #[test]
    fn test_side_pots_three_players_different_allins() {
        let mut pot_mgr = PotManager::new();

        // Player 0 all-in 1000, Player 1 all-in 3000, Player 2 all-in 5000
        pot_mgr.add_bet(0, 1000);
        pot_mgr.add_bet(1, 3000);
        pot_mgr.add_bet(2, 5000);

        let player_bets = vec![(0, 1000, true), (1, 3000, true), (2, 5000, true)];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Main pot: 1000*3 = 3000, eligible: [0, 1, 2]
        // Side pot 1: 2000*2 = 4000, eligible: [1, 2]
        // Remaining 2000 above the second stack is uncontested and returned to player 2.
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].amount, 3000);
        assert_eq!(pot_mgr.pots[1].amount, 4000);
        assert_eq!(uncontested.get(&2), Some(&2000));

        assert_eq!(pot_mgr.pots[0].eligible_players, vec![0, 1, 2]);
        assert_eq!(pot_mgr.pots[1].eligible_players, vec![1, 2]);

        // Total should be conserved
        let total: i64 = pot_mgr.pots.iter().map(|p| p.amount).sum();
        assert_eq!(total + uncontested.values().sum::<i64>(), 9000);

        // If shortest stack (player 0) has best hand:
        // P0 wins main pot (3000), P1 wins side pot 1 (4000), P2 gets 2000 back uncontested.
        let payouts = pot_mgr.award_pots(vec![vec![0], vec![1]]);
        assert_eq!(payouts.get(&0), Some(&3000));
        assert_eq!(payouts.get(&1), Some(&4000));
        assert_eq!(payouts.get(&2), None);
    }

    #[test]
    fn test_side_pots_equal_allins_no_side_pot() {
        let mut pot_mgr = PotManager::new();

        // Both players all-in for same amount — no side pot needed
        pot_mgr.add_bet(0, 5000);
        pot_mgr.add_bet(1, 5000);

        let player_bets = vec![(0, 5000, true), (1, 5000, true)];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        assert_eq!(pot_mgr.pots.len(), 1);
        assert_eq!(pot_mgr.pots[0].amount, 10000);
        assert!(pot_mgr.pots[0].eligible_players.contains(&0));
        assert!(pot_mgr.pots[0].eligible_players.contains(&1));
        assert!(uncontested.is_empty());
    }

    #[test]
    fn test_side_pots_chips_always_conserved() {
        let mut pot_mgr = PotManager::new();

        // 4 players with varied bets, one folded
        pot_mgr.add_bet(0, 500);
        pot_mgr.add_bet(1, 2000);
        pot_mgr.add_bet(2, 2000);
        pot_mgr.add_bet(3, 800);

        // P0 folded, P3 all-in at 800, P1 and P2 active
        let player_bets = vec![
            (0, 500, false),
            (1, 2000, true),
            (2, 2000, true),
            (3, 800, true),
        ];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Total of all pots must equal total of all bets
        let total_bets: i64 = 500 + 2000 + 2000 + 800;
        let total_pots: i64 = pot_mgr.pots.iter().map(|p| p.amount).sum();
        let total_uncontested: i64 = uncontested.values().sum();
        assert_eq!(
            total_pots + total_uncontested,
            total_bets,
            "Chips must be conserved"
        );

        // Folded player should not be eligible for any pot
        for pot in &pot_mgr.pots {
            assert!(
                !pot.eligible_players.contains(&0),
                "Folded player should not be eligible"
            );
        }
    }

    #[test]
    fn test_pots_consolidated_when_multiple_folds() {
        let mut pot_mgr = PotManager::new();

        // P0 folds at 200, P1 folds at 500, P2 all-in 1000, P3 calls 3000, P4 calls 3000
        pot_mgr.add_bet(0, 200);
        pot_mgr.add_bet(1, 500);
        pot_mgr.add_bet(2, 1000);
        pot_mgr.add_bet(3, 3000);
        pot_mgr.add_bet(4, 3000);

        let player_bets = vec![
            (0, 200, false), // folded
            (1, 500, false), // folded
            (2, 1000, true), // all-in
            (3, 3000, true), // active
            (4, 3000, true), // active
        ];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Without consolidation we'd get 4 pots (at levels 200, 500, 1000, 3000).
        // Levels 200 and 500 both have eligible [2, 3, 4], same as level 1000.
        // So those three merge into one pot. Level 3000 has eligible [3, 4].
        // Result: 2 pots.
        assert_eq!(pot_mgr.pots.len(), 2);

        // First pot: all bets up to 1000 level (eligible: [2, 3, 4])
        // 200*5 + 300*4 + 500*3 = 1000 + 1200 + 1500 = 3700
        assert_eq!(pot_mgr.pots[0].amount, 3700);
        assert_eq!(pot_mgr.pots[0].eligible_players, vec![2, 3, 4]);

        // Second pot: bets above 1000 (eligible: [3, 4])
        // 2000*2 = 4000
        assert_eq!(pot_mgr.pots[1].amount, 4000);
        assert_eq!(pot_mgr.pots[1].eligible_players, vec![3, 4]);

        // Total conserved
        let total: i64 = pot_mgr.pots.iter().map(|p| p.amount).sum();
        assert_eq!(total, 7700);
        assert!(uncontested.is_empty());
    }

    #[test]
    fn test_pots_not_merged_when_different_eligible() {
        let mut pot_mgr = PotManager::new();

        // Three all-in at different levels — each has different eligible set, no merging
        pot_mgr.add_bet(0, 100);
        pot_mgr.add_bet(1, 200);
        pot_mgr.add_bet(2, 300);

        let player_bets = vec![(0, 100, true), (1, 200, true), (2, 300, true)];
        let uncontested = pot_mgr.calculate_side_pots(&player_bets);

        // Top level is uncontested and returned to player 2.
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].eligible_players, vec![0, 1, 2]);
        assert_eq!(pot_mgr.pots[1].eligible_players, vec![1, 2]);
        assert_eq!(uncontested.get(&2), Some(&100));
    }

    #[test]
    fn test_award_pots() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![Pot {
            amount: 300,
            eligible_players: vec![0, 1, 2],
        }];

        let winners = vec![vec![1]]; // Player 1 wins
        let payouts = pot_mgr.award_pots(winners);

        assert_eq!(payouts.get(&1), Some(&300));
    }

    #[test]
    fn test_split_pot() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![Pot {
            amount: 300,
            eligible_players: vec![0, 1, 2],
        }];

        let winners = vec![vec![0, 2]]; // Players 0 and 2 tie
        let payouts = pot_mgr.award_pots(winners);

        assert_eq!(payouts.get(&0), Some(&150));
        assert_eq!(payouts.get(&2), Some(&150));
    }

    #[test]
    fn test_hilo_split_even_pot() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![Pot {
            amount: 300,
            eligible_players: vec![0, 1],
        }];

        // Player 0 wins hi, Player 1 wins lo
        let payouts = pot_mgr.award_pots_hilo(vec![vec![0]], vec![vec![1]]);
        assert_eq!(payouts.get(&0), Some(&150)); // hi half
        assert_eq!(payouts.get(&1), Some(&150)); // lo half
    }

    #[test]
    fn test_hilo_odd_chip_to_hi() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![Pot {
            amount: 301,
            eligible_players: vec![0, 1],
        }];

        let payouts = pot_mgr.award_pots_hilo(vec![vec![0]], vec![vec![1]]);
        assert_eq!(payouts.get(&0), Some(&151)); // hi gets odd chip
        assert_eq!(payouts.get(&1), Some(&150)); // lo gets remainder
                                                 // Total conserved
        let total: i64 = payouts.values().sum();
        assert_eq!(total, 301);
    }

    #[test]
    fn test_hilo_no_qualifying_low() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![Pot {
            amount: 300,
            eligible_players: vec![0, 1],
        }];

        // No qualifying low hand -- empty lo winners
        let payouts = pot_mgr.award_pots_hilo(vec![vec![0]], vec![vec![]]);
        assert_eq!(payouts.get(&0), Some(&300)); // hi gets entire pot
        assert_eq!(payouts.get(&1), None);
    }

    #[test]
    fn test_hilo_same_player_wins_both() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![Pot {
            amount: 400,
            eligible_players: vec![0, 1],
        }];

        // Player 0 wins both hi and lo (scoops)
        let payouts = pot_mgr.award_pots_hilo(vec![vec![0]], vec![vec![0]]);
        assert_eq!(payouts.get(&0), Some(&400)); // scoops entire pot
        assert_eq!(payouts.get(&1), None);
    }

    #[test]
    fn test_preview_side_pots_returns_correct_pots() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.add_bet(0, 1000);
        pot_mgr.add_bet(1, 3000);
        pot_mgr.add_bet(2, 5000);

        let player_bets = vec![(0, 1000, true), (1, 3000, true), (2, 5000, true)];
        let preview = pot_mgr.preview_side_pots(&player_bets);

        // Top level is uncontested and should not appear as a split pot.
        assert_eq!(preview.len(), 2);
        assert_eq!(preview[0].amount, 3000);
        assert_eq!(preview[1].amount, 4000);
        let total: i64 = preview.iter().map(|p| p.amount).sum();
        assert_eq!(total, 7000);
    }

    #[test]
    fn test_preview_side_pots_does_not_mutate() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.add_bet(0, 50);
        pot_mgr.add_bet(1, 100);
        pot_mgr.add_bet(2, 100);

        // Snapshot original state
        let original_pots_len = pot_mgr.pots.len();
        let original_total = pot_mgr.total();

        let player_bets = vec![(0, 50, true), (1, 100, true), (2, 100, true)];
        let _preview = pot_mgr.preview_side_pots(&player_bets);

        // Original state unchanged
        assert_eq!(pot_mgr.pots.len(), original_pots_len);
        assert_eq!(pot_mgr.total(), original_total);
    }

    #[test]
    fn test_preview_side_pots_single_bet_level() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.add_bet(0, 100);
        pot_mgr.add_bet(1, 100);

        let player_bets = vec![(0, 100, true), (1, 100, true)];
        let preview = pot_mgr.preview_side_pots(&player_bets);

        assert_eq!(preview.len(), 1);
        assert_eq!(preview[0].amount, 200);
    }

    #[test]
    fn test_hilo_with_side_pots() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![
            Pot {
                amount: 300,
                eligible_players: vec![0, 1, 2],
            },
            Pot {
                amount: 200,
                eligible_players: vec![1, 2],
            },
        ];

        // Main pot: P0 wins hi, P1 wins lo
        // Side pot: P2 wins hi, P1 wins lo
        let payouts = pot_mgr.award_pots_hilo(vec![vec![0], vec![2]], vec![vec![1], vec![1]]);
        // Main: P0 gets 150 (hi half), P1 gets 150 (lo half)
        // Side: P2 gets 100 (hi half), P1 gets 100 (lo half)
        assert_eq!(payouts.get(&0), Some(&150));
        assert_eq!(payouts.get(&1), Some(&250)); // 150 + 100
        assert_eq!(payouts.get(&2), Some(&100));
        // Total conserved
        let total: i64 = payouts.values().sum();
        assert_eq!(total, 500);
    }
}
