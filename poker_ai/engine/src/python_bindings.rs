use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::mem::size_of_val;
use std::slice;

use crate::game_state::{PlayerStats, SimTable};

/// Observation vector size: 364 (cards) + 86 (game state) + 128 (history placeholder) + 52 (hand strength)
const OBS_SIZE: usize = 710;

fn f32_as_pybytes<'py>(py: Python<'py>, data: &[f32]) -> Bound<'py, PyBytes> {
    let bytes = unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) };
    PyBytes::new(py, bytes)
}

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
        self.table.rebuy_busted_players();
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
            vec![0.0f32; OBS_SIZE]
        } else {
            self.table.encode_observation(next_player)
        };

        let mask = if done {
            vec![false; 9]
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

    /// Get action history encoded for history MLP (list of 11-float arrays).
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
    obs_buf: [f32; 710], // reusable observation buffer
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
        Self { envs, obs_buf: [0.0f32; 710] }
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
            vec![0.0f32; OBS_SIZE]
        } else {
            table.encode_observation(next_player)
        };

        let mask = if done {
            vec![false; 9]
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
        table.rebuy_busted_players();
        let current = table.start_hand(rng);
        let obs = table.encode_observation(current);
        let mask = table.legal_actions_mask().to_vec();
        Ok((current, obs, mask))
    }

    /// Step multiple environments at once. Takes list of (env_idx, action_idx) pairs.
    /// Returns (players, obs_flat, masks_flat, rewards_flat, dones) as flat arrays.
    /// obs_flat shape: (n, OBS_SIZE), masks_flat: (n, 9), rewards_flat: (n, num_players), dones: (n,)
    fn step_batch(
        &mut self,
        actions: Vec<(usize, usize)>,
    ) -> PyResult<(Vec<usize>, Vec<f32>, Vec<bool>, Vec<f64>, Vec<bool>)> {
        let n = actions.len();
        let num_players = if self.envs.is_empty() { 0 } else { self.envs[0].0.num_players };

        let mut players = Vec::with_capacity(n);
        let mut obs_flat = Vec::with_capacity(n * OBS_SIZE);
        let mut masks_flat = Vec::with_capacity(n * 9);
        let mut rewards_flat = Vec::with_capacity(n * num_players);
        let mut dones = Vec::with_capacity(n);

        for (env_idx, action_idx) in actions {
            let (table, _) = &mut self.envs[env_idx];
            let (done, next_player) = table.apply_action(action_idx);

            players.push(next_player);
            dones.push(done);

            if done {
                obs_flat.extend(std::iter::repeat(0.0f32).take(OBS_SIZE));
                masks_flat.extend(std::iter::repeat(false).take(9));
                let rewards: Vec<f64> = table
                    .rewards
                    .iter()
                    .map(|r| *r / table.big_blind as f64)
                    .collect();
                rewards_flat.extend(rewards);
            } else {
                let obs = table.encode_observation(next_player);
                obs_flat.extend(obs);
                let mask = table.legal_actions_mask();
                masks_flat.extend(mask);
                rewards_flat.extend(std::iter::repeat(0.0f64).take(num_players));
            }
        }

        Ok((players, obs_flat, masks_flat, rewards_flat, dones))
    }

    /// Dense stepping API for training hot-path.
    /// Takes one action per environment, ordered by env index.
    /// Returns bytes for fast NumPy decode:
    /// (players, obs_f32_bytes, masks_u8_bytes, rewards_f32_bytes, dones_u8_bytes)
    fn step_batch_dense<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<(
        Vec<usize>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let n = actions.len();
        if n != self.envs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "actions length must match num_envs",
            ));
        }
        let num_players = if self.envs.is_empty() { 0 } else { self.envs[0].0.num_players };

        let mut players = Vec::with_capacity(n);
        let mut obs_flat = Vec::with_capacity(n * OBS_SIZE);
        let mut masks_flat = Vec::with_capacity(n * 9);
        let mut rewards_flat = Vec::with_capacity(n * num_players);
        let mut dones = Vec::with_capacity(n);

        for (env_idx, action_idx) in actions.into_iter().enumerate() {
            let (table, _) = &mut self.envs[env_idx];
            let (done, next_player) = table.apply_action(action_idx);

            players.push(next_player);
            dones.push(u8::from(done));

            if done {
                obs_flat.extend(std::iter::repeat(0.0f32).take(OBS_SIZE));
                masks_flat.extend(std::iter::repeat(0u8).take(9));
                rewards_flat.extend(
                    table
                        .rewards
                        .iter()
                        .map(|r| (*r as f32) / table.big_blind as f32),
                );
            } else {
                table.encode_observation_into(next_player, &mut self.obs_buf);
                obs_flat.extend_from_slice(&self.obs_buf);
                let mask = table.legal_actions_mask();
                masks_flat.extend(mask.into_iter().map(u8::from));
                rewards_flat.extend(std::iter::repeat(0.0f32).take(num_players));
            }
        }

        Ok((
            players,
            f32_as_pybytes(py, &obs_flat),
            PyBytes::new(py, &masks_flat),
            f32_as_pybytes(py, &rewards_flat),
            PyBytes::new(py, &dones),
        ))
    }

    /// Reset multiple environments at once. Returns (players, obs_flat, masks_flat).
    fn reset_batch(
        &mut self,
        env_indices: Vec<usize>,
    ) -> PyResult<(Vec<usize>, Vec<f32>, Vec<bool>)> {
        let n = env_indices.len();
        let mut players = Vec::with_capacity(n);
        let mut obs_flat = Vec::with_capacity(n * OBS_SIZE);
        let mut masks_flat = Vec::with_capacity(n * 9);

        for env_idx in env_indices {
            let (table, rng) = &mut self.envs[env_idx];
            table.advance_dealer();
            table.rebuy_busted_players();
            let current = table.start_hand(rng);
            let obs = table.encode_observation(current);
            let mask = table.legal_actions_mask();
            players.push(current);
            obs_flat.extend(obs);
            masks_flat.extend(mask);
        }

        Ok((players, obs_flat, masks_flat))
    }

    /// Dense reset API for training hot-path.
    /// Returns bytes for fast NumPy decode: (players, obs_f32_bytes, masks_u8_bytes)
    fn reset_batch_dense<'py>(
        &mut self,
        py: Python<'py>,
        env_indices: Vec<usize>,
    ) -> PyResult<(Vec<usize>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let n = env_indices.len();
        let mut players = Vec::with_capacity(n);
        let mut obs_flat = Vec::with_capacity(n * OBS_SIZE);
        let mut masks_flat = Vec::with_capacity(n * 9);

        for env_idx in env_indices {
            let (table, rng) = &mut self.envs[env_idx];
            table.advance_dealer();
            table.rebuy_busted_players();
            let current = table.start_hand(rng);
            table.encode_observation_into(current, &mut self.obs_buf);
            let mask = table.legal_actions_mask();
            players.push(current);
            obs_flat.extend_from_slice(&self.obs_buf);
            masks_flat.extend(mask.into_iter().map(u8::from));
        }

        Ok((
            players,
            f32_as_pybytes(py, &obs_flat),
            PyBytes::new(py, &masks_flat),
        ))
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

    /// Reset HUD stats for specific seats in specific environments.
    /// Simulates a new unknown player arriving at the table.
    fn reset_player_stats(&mut self, env_indices: Vec<usize>, seat_indices: Vec<usize>) {
        for (&env_idx, &seat) in env_indices.iter().zip(seat_indices.iter()) {
            let (table, _) = &mut self.envs[env_idx];
            if seat < table.player_stats.len() {
                table.player_stats[seat] = PlayerStats::new();
            }
        }
    }
}
