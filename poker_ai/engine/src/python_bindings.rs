use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::game_state::SimTable;

/// Python-facing poker environment.
#[pyclass]
pub struct PokerEnv {
    table: SimTable,
    rng: ChaCha20Rng,
}

#[pymethods]
impl PokerEnv {
    #[new]
    #[pyo3(signature = (num_players=6, starting_stack=10000, small_blind=50, big_blind=100, seed=None))]
    fn new(
        num_players: usize,
        starting_stack: i64,
        small_blind: i64,
        big_blind: i64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        Self {
            table: SimTable::new(num_players, starting_stack, small_blind, big_blind),
            rng,
        }
    }

    /// Reset and start a new hand. Returns (current_player, observation, legal_actions_mask).
    fn reset(&mut self) -> PyResult<(usize, Vec<f32>, Vec<bool>)> {
        self.table.advance_dealer();
        let current = self.table.start_hand(&mut self.rng);
        let obs = self.table.encode_observation(current);
        let mask = self.table.legal_actions_mask();
        Ok((current, obs, mask.to_vec()))
    }

    /// Apply action. Returns (current_player, observation, legal_mask, rewards, done).
    /// rewards is a list of floats per player (0 until hand is over).
    /// done is True if hand is over.
    fn step(
        &mut self,
        action_idx: usize,
    ) -> PyResult<(usize, Vec<f32>, Vec<bool>, Vec<f64>, bool)> {
        let (done, next_player) = self.table.apply_action(action_idx);

        let obs = if done {
            vec![0.0f32; 569]
        } else {
            self.table.encode_observation(next_player)
        };

        let mask = if done {
            vec![false; 8]
        } else {
            self.table.legal_actions_mask().to_vec()
        };

        let rewards = if done {
            self.table
                .rewards
                .iter()
                .map(|r| *r / self.table.big_blind as f64)
                .collect()
        } else {
            vec![0.0; self.table.num_players]
        };

        Ok((next_player, obs, mask, rewards, done))
    }

    /// Get current observation for a specific player.
    fn get_observation(&self, seat: usize) -> PyResult<Vec<f32>> {
        Ok(self.table.encode_observation(seat))
    }

    /// Get action history encoded for LSTM (list of 7-float arrays).
    fn get_action_history(&self) -> PyResult<Vec<Vec<f32>>> {
        Ok(self
            .table
            .encode_action_history()
            .iter()
            .map(|a| a.to_vec())
            .collect())
    }

    /// Get current player index.
    fn current_player(&self) -> usize {
        self.table.current_player
    }

    /// Get number of players.
    fn num_players(&self) -> usize {
        self.table.num_players
    }

    /// Get stacks as list.
    fn stacks(&self) -> Vec<i64> {
        self.table.stacks.clone()
    }

    /// Get pot size.
    fn pot(&self) -> i64 {
        self.table.pot
    }

    /// Get big blind.
    fn big_blind(&self) -> i64 {
        self.table.big_blind
    }

    /// Whether the hand is over.
    fn is_done(&self) -> bool {
        self.table.phase == crate::game_state::Phase::HandOver
    }

    /// Set stacks for all players (for tournament-style or custom setups).
    fn set_stacks(&mut self, stacks: Vec<i64>) -> PyResult<()> {
        if stacks.len() != self.table.num_players {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stacks length must match num_players",
            ));
        }
        self.table.stacks = stacks;
        Ok(())
    }
}

/// Batch environment for running multiple games in parallel.
#[pyclass]
pub struct BatchPokerEnv {
    envs: Vec<(SimTable, ChaCha20Rng)>,
}

#[pymethods]
impl BatchPokerEnv {
    #[new]
    #[pyo3(signature = (num_envs, num_players=6, starting_stack=10000, small_blind=50, big_blind=100, base_seed=None))]
    fn new(
        num_envs: usize,
        num_players: usize,
        starting_stack: i64,
        small_blind: i64,
        big_blind: i64,
        base_seed: Option<u64>,
    ) -> Self {
        let envs = (0..num_envs)
            .map(|i| {
                let rng = match base_seed {
                    Some(s) => ChaCha20Rng::seed_from_u64(s + i as u64),
                    None => ChaCha20Rng::from_entropy(),
                };
                let table = SimTable::new(num_players, starting_stack, small_blind, big_blind);
                (table, rng)
            })
            .collect();
        Self { envs }
    }

    /// Reset all environments. Returns list of (current_player, obs, mask) tuples.
    fn reset_all(&mut self) -> PyResult<Vec<(usize, Vec<f32>, Vec<bool>)>> {
        let mut results = Vec::with_capacity(self.envs.len());
        for (table, rng) in &mut self.envs {
            table.advance_dealer();
            let current = table.start_hand(rng);
            let obs = table.encode_observation(current);
            let mask = table.legal_actions_mask().to_vec();
            results.push((current, obs, mask));
        }
        Ok(results)
    }

    /// Step a single environment by index.
    fn step(
        &mut self,
        env_idx: usize,
        action_idx: usize,
    ) -> PyResult<(usize, Vec<f32>, Vec<bool>, Vec<f64>, bool)> {
        let (table, _) = &mut self.envs[env_idx];
        let (done, next_player) = table.apply_action(action_idx);

        let obs = if done {
            vec![0.0f32; 569]
        } else {
            table.encode_observation(next_player)
        };

        let mask = if done {
            vec![false; 8]
        } else {
            table.legal_actions_mask().to_vec()
        };

        let rewards = if done {
            table
                .rewards
                .iter()
                .map(|r| *r / table.big_blind as f64)
                .collect()
        } else {
            vec![0.0; table.num_players]
        };

        Ok((next_player, obs, mask, rewards, done))
    }

    /// Reset a single environment by index.
    fn reset_env(&mut self, env_idx: usize) -> PyResult<(usize, Vec<f32>, Vec<bool>)> {
        let (table, rng) = &mut self.envs[env_idx];
        table.advance_dealer();
        // Reset stacks if busted
        let starting_stack = table.initial_stacks[0]; // use first player's initial
        for i in 0..table.num_players {
            if table.stacks[i] <= 0 {
                table.stacks[i] = starting_stack;
            }
        }
        let current = table.start_hand(rng);
        let obs = table.encode_observation(current);
        let mask = table.legal_actions_mask().to_vec();
        Ok((current, obs, mask))
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn num_players(&self) -> usize {
        if self.envs.is_empty() {
            0
        } else {
            self.envs[0].0.num_players
        }
    }
}
