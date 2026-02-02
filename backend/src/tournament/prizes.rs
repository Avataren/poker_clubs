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
            return Self { payouts: vec![100.0] };
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
        let idx = (position - 1) as usize;
        if idx < self.payouts.len() {
            (total_prize_pool as f64 * self.payouts[idx] / 100.0) as i64
        } else {
            0
        }
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
        assert!(first >= pool / 6, "First place should be at least ~17%: got {}", first);
        
        // Total should equal pool (within rounding)
        let mut total = 0;
        for pos in 1..=10 {
            total += structure.prize_for_position(pos, pool);
        }
        assert!((total - pool).abs() < 100, "Total should match pool within rounding");
        
        // Verify decreasing amounts
        for pos in 1..10 {
            let current = structure.prize_for_position(pos, pool);
            let next = structure.prize_for_position(pos + 1, pool);
            assert!(current > next, "Position {} should pay more than position {}", pos, pos + 1);
        }
    }
}
