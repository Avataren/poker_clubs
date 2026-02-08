//! Prize Structure and Distribution
//!
//! Handles prize pool calculations and payouts for tournaments

use serde::{Deserialize, Serialize};

/// Prize winner information
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PrizeWinner {
    pub user_id: String,
    pub username: String,
    pub position: i32,
    pub prize_amount: i64,
}

/// Prize distribution structure
#[derive(Debug, Clone)]
pub struct PrizeStructure {
    /// Percentages for each position (index 0 = 1st place)
    pub payouts: Vec<f64>,
}

impl PrizeStructure {
    /// Standard payout for heads-up (winner takes all)
    pub fn heads_up() -> Self {
        Self {
            payouts: vec![100.0],
        }
    }

    /// Standard payout for 3-player tournament
    pub fn three_player() -> Self {
        Self {
            payouts: vec![65.0, 35.0],
        }
    }

    /// Standard payout for 6-max tournament
    pub fn six_max() -> Self {
        Self {
            payouts: vec![65.0, 35.0],
        }
    }

    /// Standard payout for 9-player tournament
    pub fn nine_player() -> Self {
        Self {
            payouts: vec![50.0, 30.0, 20.0],
        }
    }

    /// Standard payout for 18-player tournament
    pub fn eighteen_player() -> Self {
        Self {
            payouts: vec![40.0, 25.0, 17.0, 10.0, 8.0],
        }
    }

    /// Get prize structure based on player count
    pub fn for_player_count(count: i32) -> Self {
        match count {
            1..=2 => Self::heads_up(),
            3 => Self::three_player(),
            4..=6 => Self::six_max(),
            7..=9 => Self::nine_player(),
            10..=27 => Self::eighteen_player(),
            _ => {
                // For larger tournaments, pay top ~15%
                let payout_positions = (count as f64 * 0.15).ceil() as usize;
                Self::large_tournament(payout_positions)
            }
        }
    }

    /// Create payout structure for large tournaments
    /// Pays decreasing percentages for top positions
    fn large_tournament(positions: usize) -> Self {
        if positions == 0 {
            return Self {
                payouts: vec![100.0],
            };
        }

        // Use simple decreasing percentages
        // First gets most, each subsequent position gets less
        let mut payouts = Vec::new();
        let mut total_weight = 0.0;

        // Calculate weights (position 1 gets weight 'positions', position 2 gets 'positions-1', etc.)
        for i in 0..positions {
            let weight = (positions - i) as f64;
            payouts.push(weight);
            total_weight += weight;
        }

        // Convert weights to percentages
        payouts = payouts.iter().map(|w| (w / total_weight) * 100.0).collect();

        Self { payouts }
    }

    /// Calculate prize for a specific position
    pub fn prize_for_position(&self, position: i32, total_prize_pool: i64) -> i64 {
        let prizes = self.calculate_all_prizes(total_prize_pool);
        let idx = (position - 1) as usize;
        prizes.get(idx).copied().unwrap_or(0)
    }

    /// Calculate all prizes at once, ensuring they sum exactly to the prize pool
    /// Any remaining chips from rounding are awarded to 1st place
    pub fn calculate_all_prizes(&self, total_prize_pool: i64) -> Vec<i64> {
        if self.payouts.is_empty() {
            return vec![];
        }

        let mut prizes: Vec<i64> = self
            .payouts
            .iter()
            .map(|pct| (total_prize_pool as f64 * pct / 100.0) as i64)
            .collect();

        // Calculate how much was lost to rounding
        let sum: i64 = prizes.iter().sum();
        let remainder = total_prize_pool - sum;

        // Award any remainder to 1st place to ensure prize pool is fully distributed
        if remainder > 0 && !prizes.is_empty() {
            prizes[0] += remainder;
        }

        prizes
    }

    /// Get number of paid positions
    pub fn paid_positions(&self) -> usize {
        self.payouts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heads_up_payout() {
        let structure = PrizeStructure::heads_up();
        assert_eq!(structure.prize_for_position(1, 1000), 1000);
        assert_eq!(structure.prize_for_position(2, 1000), 0);
    }

    #[test]
    fn test_nine_player_payout() {
        let structure = PrizeStructure::nine_player();
        let pool = 9000;

        assert_eq!(structure.prize_for_position(1, pool), 4500); // 50%
        assert_eq!(structure.prize_for_position(2, pool), 2700); // 30%
        assert_eq!(structure.prize_for_position(3, pool), 1800); // 20%
        assert_eq!(structure.prize_for_position(4, pool), 0);
    }

    #[test]
    fn test_player_count_selection() {
        assert_eq!(PrizeStructure::for_player_count(2).paid_positions(), 1);
        assert_eq!(PrizeStructure::for_player_count(6).paid_positions(), 2);
        assert_eq!(PrizeStructure::for_player_count(9).paid_positions(), 3);
        assert_eq!(PrizeStructure::for_player_count(18).paid_positions(), 5);
    }

    #[test]
    fn test_large_tournament_payout() {
        let structure = PrizeStructure::large_tournament(10);
        assert_eq!(structure.paid_positions(), 10);

        // First place should get most
        let pool = 10000;
        let first = structure.prize_for_position(1, pool);
        let second = structure.prize_for_position(2, pool);

        assert!(first > second, "First place should be more than second");
        // With 10 positions, first gets about 18% (10/55 of the pool)
        assert!(
            first >= pool / 6,
            "First place should be at least ~17%: got {}",
            first
        );

        // Total should equal pool (within rounding)
        let mut total = 0;
        for pos in 1..=10 {
            total += structure.prize_for_position(pos, pool);
        }
        assert!(
            (total - pool).abs() < 100,
            "Total should match pool within rounding"
        );

        // Verify decreasing amounts
        for pos in 1..10 {
            let current = structure.prize_for_position(pos, pool);
            let next = structure.prize_for_position(pos + 1, pool);
            assert!(
                current > next,
                "Position {} should pay more than position {}",
                pos,
                pos + 1
            );
        }
    }

    #[test]
    fn test_prize_pool_adds_up_exactly() {
        // Test that all prizes sum to exactly the prize pool (no rounding loss)
        let test_cases = vec![
            (PrizeStructure::heads_up(), 1000),
            (PrizeStructure::heads_up(), 1001),
            (PrizeStructure::heads_up(), 9999),
            (PrizeStructure::three_player(), 1000),
            (PrizeStructure::three_player(), 1001),
            (PrizeStructure::three_player(), 3333),
            (PrizeStructure::six_max(), 5000),
            (PrizeStructure::six_max(), 5001),
            (PrizeStructure::nine_player(), 9000),
            (PrizeStructure::nine_player(), 9001),
            (PrizeStructure::nine_player(), 10000),
            (PrizeStructure::eighteen_player(), 18000),
            (PrizeStructure::eighteen_player(), 18001),
            (PrizeStructure::large_tournament(10), 10000),
            (PrizeStructure::large_tournament(10), 10001),
            (PrizeStructure::large_tournament(20), 20000),
            (PrizeStructure::large_tournament(20), 20001),
        ];

        for (structure, pool) in test_cases {
            let prizes = structure.calculate_all_prizes(pool);
            let total: i64 = prizes.iter().sum();
            assert_eq!(
                total,
                pool,
                "Prize pool {} should equal sum of prizes {} for {} positions",
                pool,
                total,
                structure.paid_positions()
            );

            // Also verify using prize_for_position
            let mut sum_via_position = 0;
            for pos in 1..=structure.paid_positions() as i32 {
                sum_via_position += structure.prize_for_position(pos, pool);
            }
            assert_eq!(
                sum_via_position,
                pool,
                "Prize pool {} should equal sum via prize_for_position {} for {} positions",
                pool,
                sum_via_position,
                structure.paid_positions()
            );
        }
    }

    #[test]
    fn test_no_negative_prizes() {
        let structures = vec![
            PrizeStructure::heads_up(),
            PrizeStructure::three_player(),
            PrizeStructure::six_max(),
            PrizeStructure::nine_player(),
            PrizeStructure::eighteen_player(),
            PrizeStructure::large_tournament(10),
            PrizeStructure::large_tournament(50),
        ];

        for structure in structures {
            let prizes = structure.calculate_all_prizes(10000);
            for (idx, &prize) in prizes.iter().enumerate() {
                assert!(
                    prize >= 0,
                    "Prize for position {} should not be negative: {}",
                    idx + 1,
                    prize
                );
            }
        }
    }

    #[test]
    fn test_prizes_are_decreasing() {
        let structures = vec![
            (PrizeStructure::three_player(), 3000),
            (PrizeStructure::six_max(), 6000),
            (PrizeStructure::nine_player(), 9000),
            (PrizeStructure::eighteen_player(), 18000),
            (PrizeStructure::large_tournament(10), 10000),
        ];

        for (structure, pool) in structures {
            let prizes = structure.calculate_all_prizes(pool);
            for i in 0..prizes.len() - 1 {
                assert!(
                    prizes[i] > prizes[i + 1],
                    "Position {} ({}) should be more than position {} ({})",
                    i + 1,
                    prizes[i],
                    i + 2,
                    prizes[i + 1]
                );
            }
        }
    }

    #[test]
    fn test_first_place_gets_most() {
        let test_cases = vec![
            (PrizeStructure::three_player(), 3000, 0.60), // At least 60%
            (PrizeStructure::six_max(), 6000, 0.60),      // At least 60%
            (PrizeStructure::nine_player(), 9000, 0.45),  // At least 45%
            (PrizeStructure::eighteen_player(), 18000, 0.35), // At least 35%
        ];

        for (structure, pool, min_pct) in test_cases {
            let first_prize = structure.prize_for_position(1, pool);
            let min_expected = (pool as f64 * min_pct) as i64;
            assert!(
                first_prize >= min_expected,
                "First place should get at least {}% of {}: got {}",
                min_pct * 100.0,
                pool,
                first_prize
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        // Zero prize pool
        let structure = PrizeStructure::nine_player();
        let prizes = structure.calculate_all_prizes(0);
        let total: i64 = prizes.iter().sum();
        assert_eq!(total, 0, "Zero prize pool should yield zero prizes");

        // Very small prize pool
        let prizes = structure.calculate_all_prizes(1);
        let total: i64 = prizes.iter().sum();
        assert_eq!(total, 1, "Prize pool of 1 should be fully distributed");

        // Very large prize pool
        let large_pool = 1_000_000_000;
        let prizes = structure.calculate_all_prizes(large_pool);
        let total: i64 = prizes.iter().sum();
        assert_eq!(
            total, large_pool,
            "Large prize pool should be fully distributed"
        );
    }

    #[test]
    fn test_all_percentages_sum_to_100() {
        let structures = vec![
            PrizeStructure::heads_up(),
            PrizeStructure::three_player(),
            PrizeStructure::six_max(),
            PrizeStructure::nine_player(),
            PrizeStructure::eighteen_player(),
            PrizeStructure::large_tournament(10),
            PrizeStructure::large_tournament(20),
        ];

        for structure in structures {
            let total_pct: f64 = structure.payouts.iter().sum();
            assert!(
                (total_pct - 100.0).abs() < 0.01,
                "Percentages should sum to 100%, got {}",
                total_pct
            );
        }
    }
}
