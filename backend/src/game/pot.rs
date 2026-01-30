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

    /// Calculate side pots based on each player's total contribution to the hand.
    /// `player_bets` contains (player_idx, total_bet_this_hand, is_active_in_hand).
    /// All contributing players (including folded) affect pot sizes, but only
    /// active-in-hand players are eligible to win.
    pub fn calculate_side_pots(&mut self, player_bets: &[(usize, i64, bool)]) {
        if player_bets.is_empty() {
            return;
        }

        // Sort by total bet amount
        let mut sorted: Vec<(usize, i64, bool)> = player_bets.to_vec();
        sorted.sort_by_key(|(_, bet, _)| *bet);

        let mut new_pots = Vec::new();
        let mut prev_level = 0i64;

        for i in 0..sorted.len() {
            let (_, bet_level, _) = sorted[i];
            if bet_level <= prev_level {
                continue; // Skip duplicate bet levels
            }

            let level_contribution = bet_level - prev_level;
            // All players who bet more than prev_level contribute to this pot
            let contributors = sorted.iter().filter(|(_, bet, _)| *bet > prev_level).count();
            let pot_amount = level_contribution * contributors as i64;

            // Only active-in-hand players who bet at least this level can win this pot
            let eligible: Vec<usize> = sorted.iter()
                .filter(|(_, bet, active)| *bet >= bet_level && *active)
                .map(|(idx, _, _)| *idx)
                .collect();

            if pot_amount > 0 {
                new_pots.push(Pot {
                    amount: pot_amount,
                    eligible_players: eligible,
                });
            }

            prev_level = bet_level;
        }

        if !new_pots.is_empty() {
            self.pots = new_pots;
        }
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
        pot_mgr.calculate_side_pots(&player_bets);

        // Should have 2 pots:
        // Main pot: 150 (50 from each of 3 players)
        // Side pot: 100 (50 more from players 1 and 2)
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].amount, 150);
        assert_eq!(pot_mgr.pots[1].amount, 100);
    }

    #[test]
    fn test_side_pots_short_stack_wins() {
        let mut pot_mgr = PotManager::new();

        // Player 0 goes all-in with 5000
        // Player 1 goes all-in with 10000
        pot_mgr.add_bet(0, 5000);
        pot_mgr.add_bet(1, 10000);

        let player_bets = vec![(0, 5000, true), (1, 10000, true)];
        pot_mgr.calculate_side_pots(&player_bets);

        // Main pot: 10000 (5000 from each), eligible: [0, 1]
        // Side pot: 5000 (remaining 5000 from player 1), eligible: [1] only
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].amount, 10000);
        assert!(pot_mgr.pots[0].eligible_players.contains(&0));
        assert!(pot_mgr.pots[0].eligible_players.contains(&1));
        assert_eq!(pot_mgr.pots[1].amount, 5000);
        assert!(!pot_mgr.pots[1].eligible_players.contains(&0));
        assert!(pot_mgr.pots[1].eligible_players.contains(&1));

        // If player 0 wins: gets main pot (10000), player 1 gets side pot (5000) back
        let payouts = pot_mgr.award_pots(vec![vec![0], vec![1]]);
        assert_eq!(payouts.get(&0), Some(&10000));
        assert_eq!(payouts.get(&1), Some(&5000));
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
        pot_mgr.calculate_side_pots(&player_bets);

        // Pot at level 100: 100*3 = 300, eligible: [1, 2] (player 0 folded)
        // Pot at level 200: 100*2 = 200, eligible: [1, 2]
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].amount, 300);
        assert_eq!(pot_mgr.pots[1].amount, 200);
        // Folded player 0 is not eligible for any pot
        assert!(!pot_mgr.pots[0].eligible_players.contains(&0));
        assert!(!pot_mgr.pots[1].eligible_players.contains(&0));
    }

    #[test]
    fn test_side_pots_three_players_different_allins() {
        let mut pot_mgr = PotManager::new();

        // Player 0 all-in 1000, Player 1 all-in 3000, Player 2 all-in 5000
        pot_mgr.add_bet(0, 1000);
        pot_mgr.add_bet(1, 3000);
        pot_mgr.add_bet(2, 5000);

        let player_bets = vec![(0, 1000, true), (1, 3000, true), (2, 5000, true)];
        pot_mgr.calculate_side_pots(&player_bets);

        // Main pot: 1000*3 = 3000, eligible: [0, 1, 2]
        // Side pot 1: 2000*2 = 4000, eligible: [1, 2]
        // Side pot 2: 2000*1 = 2000, eligible: [2]
        assert_eq!(pot_mgr.pots.len(), 3);
        assert_eq!(pot_mgr.pots[0].amount, 3000);
        assert_eq!(pot_mgr.pots[1].amount, 4000);
        assert_eq!(pot_mgr.pots[2].amount, 2000);

        assert_eq!(pot_mgr.pots[0].eligible_players, vec![0, 1, 2]);
        assert_eq!(pot_mgr.pots[1].eligible_players, vec![1, 2]);
        assert_eq!(pot_mgr.pots[2].eligible_players, vec![2]);

        // Total should be conserved
        let total: i64 = pot_mgr.pots.iter().map(|p| p.amount).sum();
        assert_eq!(total, 9000);

        // If shortest stack (player 0) has best hand:
        // P0 wins main pot (3000), P1 wins side pot 1 (4000 - next best), P2 gets side pot 2 (2000 - refund)
        let payouts = pot_mgr.award_pots(vec![vec![0], vec![1], vec![2]]);
        assert_eq!(payouts.get(&0), Some(&3000));
        assert_eq!(payouts.get(&1), Some(&4000));
        assert_eq!(payouts.get(&2), Some(&2000));
    }

    #[test]
    fn test_side_pots_equal_allins_no_side_pot() {
        let mut pot_mgr = PotManager::new();

        // Both players all-in for same amount — no side pot needed
        pot_mgr.add_bet(0, 5000);
        pot_mgr.add_bet(1, 5000);

        let player_bets = vec![(0, 5000, true), (1, 5000, true)];
        pot_mgr.calculate_side_pots(&player_bets);

        assert_eq!(pot_mgr.pots.len(), 1);
        assert_eq!(pot_mgr.pots[0].amount, 10000);
        assert!(pot_mgr.pots[0].eligible_players.contains(&0));
        assert!(pot_mgr.pots[0].eligible_players.contains(&1));
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
        let player_bets = vec![(0, 500, false), (1, 2000, true), (2, 2000, true), (3, 800, true)];
        pot_mgr.calculate_side_pots(&player_bets);

        // Total of all pots must equal total of all bets
        let total_bets: i64 = 500 + 2000 + 2000 + 800;
        let total_pots: i64 = pot_mgr.pots.iter().map(|p| p.amount).sum();
        assert_eq!(total_pots, total_bets, "Chips must be conserved");

        // Folded player should not be eligible for any pot
        for pot in &pot_mgr.pots {
            assert!(!pot.eligible_players.contains(&0), "Folded player should not be eligible");
        }
    }

    #[test]
    fn test_award_pots() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![
            Pot {
                amount: 300,
                eligible_players: vec![0, 1, 2],
            },
        ];

        let winners = vec![vec![1]]; // Player 1 wins
        let payouts = pot_mgr.award_pots(winners);

        assert_eq!(payouts.get(&1), Some(&300));
    }

    #[test]
    fn test_split_pot() {
        let mut pot_mgr = PotManager::new();
        pot_mgr.pots = vec![
            Pot {
                amount: 300,
                eligible_players: vec![0, 1, 2],
            },
        ];

        let winners = vec![vec![0, 2]]; // Players 0 and 2 tie
        let payouts = pot_mgr.award_pots(winners);

        assert_eq!(payouts.get(&0), Some(&150));
        assert_eq!(payouts.get(&2), Some(&150));
    }
}
