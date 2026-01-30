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

    /// Calculate side pots when players are all-in with different amounts
    #[allow(dead_code)] // Will be integrated when side pot display is added
    pub fn calculate_side_pots(&mut self, all_in_players: &HashMap<usize, i64>) {
        if all_in_players.is_empty() || self.current_bets.is_empty() {
            return;
        }

        let mut sorted_bets: Vec<(usize, i64)> = self.current_bets.iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        sorted_bets.sort_by_key(|(_, amt)| *amt);

        let mut new_pots = Vec::new();
        let mut prev_level = 0i64;

        // Group players by bet level and create side pots
        let mut remaining_players: Vec<usize> = sorted_bets.iter().map(|(idx, _)| *idx).collect();

        for (_, bet_level) in sorted_bets.iter() {
            if *bet_level > prev_level {
                let level_contribution = bet_level - prev_level;
                let pot_amount = level_contribution * remaining_players.len() as i64;

                if pot_amount > 0 {
                    new_pots.push(Pot {
                        amount: pot_amount,
                        eligible_players: remaining_players.clone(),
                    });
                }

                prev_level = *bet_level;
            }

            // Remove all-in players from remaining for next pot
            remaining_players.retain(|idx| {
                !all_in_players.contains_key(idx) ||
                all_in_players.get(idx).unwrap() > bet_level
            });
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

        let mut all_in = HashMap::new();
        all_in.insert(0, 50);

        pot_mgr.calculate_side_pots(&all_in);

        // Should have 2 pots:
        // Main pot: 150 (50 from each of 3 players)
        // Side pot: 100 (50 more from players 1 and 2)
        assert_eq!(pot_mgr.pots.len(), 2);
        assert_eq!(pot_mgr.pots[0].amount, 150);
        assert_eq!(pot_mgr.pots[1].amount, 100);
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
