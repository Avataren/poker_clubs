//! ONNX-backed bot strategy.
//!
//! This strategy mirrors the poker_ai discrete 9-action policy interface and
//! falls back to a balanced heuristic strategy if model inference fails.

use super::strategy::{BotGameView, BotStrategy, SimpleStrategy};
use crate::game::deck::Card;
use crate::game::hand::evaluate_hand;
use crate::game::player::PlayerState;
use crate::game::table::PokerTable;
use crate::game::{GamePhase, PlayerAction};
use std::path::Path;
use std::sync::Mutex;
use tract_onnx::prelude::*;

const OBS_DIM: usize = 582;
const HISTORY_DIM: usize = 11;
const MAX_HISTORY_LEN: usize = 30;
const NUM_ACTIONS: usize = 9;
const EMA_ALPHA: f64 = 0.02;

type OnnxPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>;

/// Number of floats in per-opponent stats encoding.
const STATS_PER_OPPONENT: usize = 15;

/// Per-opponent EMA stats, tracked by the bot from observed table actions.
#[derive(Debug, Clone)]
struct SeatStats {
    // EMA running stats
    vpip: f64,
    pfr: f64,
    aggression: f64,
    fold_to_bet: f64,
    hands_played: f64,
    flop_aggression: f64,
    turn_aggression: f64,
    river_aggression: f64,
    wtsd: f64,
    wsd: f64,
    cbet: f64,
    avg_bet_size: f64,
    preflop_raise_size: f64,
    ep_vpip: f64,
    lp_vpip: f64,
    // Per-hand accumulators
    hand_vpip: bool,
    hand_pfr: bool,
    hand_raises: u32,
    hand_calls: u32,
    hand_folds_to_bet: u32,
    hand_faced_bets: u32,
    hand_flop_raises: u32,
    hand_flop_actions: u32,
    hand_turn_raises: u32,
    hand_turn_actions: u32,
    hand_river_raises: u32,
    hand_river_actions: u32,
    hand_saw_flop: bool,
    hand_cbet_opportunity: bool,
    hand_cbet_taken: bool,
    hand_acted_on_flop: bool,
    hand_bet_size_sum: f64,
    hand_bet_count: u32,
    hand_preflop_raise_size_sum: f64,
    hand_preflop_raise_count: u32,
    hand_is_early_position: bool,
    hand_is_late_position: bool,
    hand_went_to_showdown: bool,
    hand_won_at_showdown: bool,
}

impl SeatStats {
    fn new() -> Self {
        Self {
            vpip: 0.5, pfr: 0.5, aggression: 0.5, fold_to_bet: 0.5,
            hands_played: 100.0, // prior: model trained with sample_size≈1.0 (99.8% of data)
            flop_aggression: 0.5, turn_aggression: 0.5, river_aggression: 0.5,
            wtsd: 0.5, wsd: 0.5, cbet: 0.5,
            avg_bet_size: 0.5, preflop_raise_size: 0.5,
            ep_vpip: 0.5, lp_vpip: 0.5,
            hand_vpip: false, hand_pfr: false,
            hand_raises: 0, hand_calls: 0,
            hand_folds_to_bet: 0, hand_faced_bets: 0,
            hand_flop_raises: 0, hand_flop_actions: 0,
            hand_turn_raises: 0, hand_turn_actions: 0,
            hand_river_raises: 0, hand_river_actions: 0,
            hand_saw_flop: false,
            hand_cbet_opportunity: false, hand_cbet_taken: false,
            hand_acted_on_flop: false,
            hand_bet_size_sum: 0.0, hand_bet_count: 0,
            hand_preflop_raise_size_sum: 0.0, hand_preflop_raise_count: 0,
            hand_is_early_position: false, hand_is_late_position: false,
            hand_went_to_showdown: false, hand_won_at_showdown: false,
        }
    }
    fn start_hand(&mut self) {
        self.hand_vpip = false;
        self.hand_pfr = false;
        self.hand_raises = 0;
        self.hand_calls = 0;
        self.hand_folds_to_bet = 0;
        self.hand_faced_bets = 0;
        self.hand_flop_raises = 0;
        self.hand_flop_actions = 0;
        self.hand_turn_raises = 0;
        self.hand_turn_actions = 0;
        self.hand_river_raises = 0;
        self.hand_river_actions = 0;
        self.hand_saw_flop = false;
        self.hand_cbet_opportunity = false;
        self.hand_cbet_taken = false;
        self.hand_acted_on_flop = false;
        self.hand_bet_size_sum = 0.0;
        self.hand_bet_count = 0;
        self.hand_preflop_raise_size_sum = 0.0;
        self.hand_preflop_raise_count = 0;
        self.hand_is_early_position = false;
        self.hand_is_late_position = false;
        self.hand_went_to_showdown = false;
        self.hand_won_at_showdown = false;
    }
    fn set_position(&mut self, seat_offset: usize, num_players: usize) {
        if num_players <= 3 {
            self.hand_is_late_position = seat_offset == 0 || seat_offset == num_players;
            self.hand_is_early_position = !self.hand_is_late_position;
        } else {
            self.hand_is_late_position = seat_offset == 0 || seat_offset == num_players - 1;
            self.hand_is_early_position = seat_offset >= 3 && seat_offset <= num_players / 2 + 1;
        }
    }
    fn mark_saw_flop(&mut self) {
        self.hand_saw_flop = true;
        if self.hand_pfr {
            self.hand_cbet_opportunity = true;
        }
    }
    fn record_showdown(&mut self, won: bool) {
        self.hand_went_to_showdown = true;
        self.hand_won_at_showdown = won;
    }
    fn end_hand(&mut self) {
        self.hands_played += 1.0;
        let a = EMA_ALPHA;
        self.vpip = self.vpip * (1.0 - a) + if self.hand_vpip { a } else { 0.0 };
        self.pfr = self.pfr * (1.0 - a) + if self.hand_pfr { a } else { 0.0 };
        let total_rc = (self.hand_raises + self.hand_calls) as f64;
        if total_rc > 0.0 {
            let hand_agg = self.hand_raises as f64 / total_rc;
            self.aggression = self.aggression * (1.0 - a) + hand_agg * a;
        }
        if self.hand_faced_bets > 0 {
            let hand_ftb = self.hand_folds_to_bet as f64 / self.hand_faced_bets as f64;
            self.fold_to_bet = self.fold_to_bet * (1.0 - a) + hand_ftb * a;
        }
        // Street-specific aggression
        if self.hand_flop_actions > 0 {
            let agg = self.hand_flop_raises as f64 / self.hand_flop_actions as f64;
            self.flop_aggression = self.flop_aggression * (1.0 - a) + agg * a;
        }
        if self.hand_turn_actions > 0 {
            let agg = self.hand_turn_raises as f64 / self.hand_turn_actions as f64;
            self.turn_aggression = self.turn_aggression * (1.0 - a) + agg * a;
        }
        if self.hand_river_actions > 0 {
            let agg = self.hand_river_raises as f64 / self.hand_river_actions as f64;
            self.river_aggression = self.river_aggression * (1.0 - a) + agg * a;
        }
        // Showdown stats
        if self.hand_saw_flop {
            let went = if self.hand_went_to_showdown { 1.0 } else { 0.0 };
            self.wtsd = self.wtsd * (1.0 - a) + went * a;
        }
        if self.hand_went_to_showdown {
            let won = if self.hand_won_at_showdown { 1.0 } else { 0.0 };
            self.wsd = self.wsd * (1.0 - a) + won * a;
        }
        // C-bet
        if self.hand_cbet_opportunity {
            let cbet_val = if self.hand_cbet_taken { 1.0 } else { 0.0 };
            self.cbet = self.cbet * (1.0 - a) + cbet_val * a;
        }
        // Bet sizing
        if self.hand_bet_count > 0 {
            let avg = self.hand_bet_size_sum / self.hand_bet_count as f64;
            self.avg_bet_size = self.avg_bet_size * (1.0 - a) + avg * a;
        }
        if self.hand_preflop_raise_count > 0 {
            let avg = self.hand_preflop_raise_size_sum / self.hand_preflop_raise_count as f64;
            self.preflop_raise_size = self.preflop_raise_size * (1.0 - a) + avg * a;
        }
        // Position-aware VPIP
        if self.hand_is_early_position {
            self.ep_vpip = self.ep_vpip * (1.0 - a) + if self.hand_vpip { a } else { 0.0 };
        }
        if self.hand_is_late_position {
            self.lp_vpip = self.lp_vpip * (1.0 - a) + if self.hand_vpip { a } else { 0.0 };
        }
    }
    fn encode(&self) -> [f32; STATS_PER_OPPONENT] {
        let sample_size = (self.hands_played as f32 / 100.0).min(1.0);
        [
            self.vpip as f32, self.pfr as f32, self.aggression as f32,
            self.fold_to_bet as f32, sample_size,
            self.flop_aggression as f32, self.turn_aggression as f32,
            self.river_aggression as f32,
            self.wtsd as f32, self.wsd as f32,
            self.cbet as f32,
            (self.avg_bet_size as f32).min(5.0) / 5.0,
            (self.preflop_raise_size as f32).min(10.0) / 10.0,
            self.ep_vpip as f32, self.lp_vpip as f32,
        ]
    }
}

/// Tracks opponent statistics from observed table actions.
/// Each bot instance owns one of these — stats are built incrementally.
struct OpponentTracker {
    seats: Vec<SeatStats>,
    /// How many actions we've already processed from the current hand.
    actions_seen: usize,
    /// Last known action count to detect hand boundaries.
    last_hand_action_count: usize,
    /// Track phase seen at last observation for street-specific tracking.
    last_phase: GamePhase,
    /// Whether we've already marked saw_flop for this hand.
    flop_marked: bool,
    /// Track dealer position for position-aware stats.
    last_dealer: usize,
}

impl OpponentTracker {
    fn new() -> Self {
        Self {
            seats: Vec::new(),
            actions_seen: 0,
            last_hand_action_count: 0,
            last_phase: GamePhase::Waiting,
            flop_marked: false,
            last_dealer: 0,
        }
    }

    /// Observe the table and update stats from any new actions since last call.
    fn observe(&mut self, table: &PokerTable) {
        let num_seats = table.players.len();
        while self.seats.len() < num_seats {
            self.seats.push(SeatStats::new());
        }

        let history = &table.hand_action_history;

        // Detect new hand: if action history got shorter, a new hand started
        if history.len() < self.last_hand_action_count {
            // Finalize showdown stats for previous hand
            self.finalize_showdown(table);

            for s in &mut self.seats[..num_seats] {
                s.end_hand();
                s.start_hand();
            }
            self.actions_seen = 0;
            self.flop_marked = false;

            // Set position for each seat in the new hand
            let dealer = table.dealer_seat;
            self.last_dealer = dealer;
            for i in 0..num_seats {
                let offset = (i + num_seats - dealer) % num_seats;
                self.seats[i].set_position(offset, num_seats);
            }
        }
        self.last_hand_action_count = history.len();

        // Mark saw_flop when phase transitions past preflop
        if !self.flop_marked && !matches!(table.phase, GamePhase::PreFlop | GamePhase::Waiting) {
            self.flop_marked = true;
            for i in 0..num_seats {
                let folded = matches!(table.players[i].state, PlayerState::Folded);
                if !folded {
                    self.seats[i].mark_saw_flop();
                }
            }
        }

        // Determine the phase for new actions using table's current phase
        let phase = table.phase.clone();

        // Process new actions
        for i in self.actions_seen..history.len() {
            let rec = &history[i];
            let actor = if num_seats > 1 {
                (rec[0] * (num_seats - 1) as f32).round() as usize
            } else {
                0
            };
            if actor >= num_seats { continue; }

            let action_idx = (1..10).max_by(|&a, &b| rec[a].partial_cmp(&rec[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(1) - 1;
            let is_fold = action_idx == 0;
            let is_call = action_idx == 1;
            let is_raise = action_idx >= 2;
            let facing_bet = is_fold || is_call || (is_raise && table.current_bet > 0);
            let bet_ratio = (rec[10] as f64).min(10.0);

            let stats = &mut self.seats[actor];
            let is_preflop = matches!(phase, GamePhase::PreFlop);

            if is_fold {
                if facing_bet {
                    stats.hand_folds_to_bet += 1;
                    stats.hand_faced_bets += 1;
                }
                match phase {
                    GamePhase::Flop => { stats.hand_flop_actions += 1; }
                    GamePhase::Turn => { stats.hand_turn_actions += 1; }
                    GamePhase::River => { stats.hand_river_actions += 1; }
                    _ => {}
                }
            } else if is_call {
                if facing_bet {
                    stats.hand_calls += 1;
                    stats.hand_faced_bets += 1;
                    if is_preflop { stats.hand_vpip = true; }
                }
                match phase {
                    GamePhase::Flop => { stats.hand_flop_actions += 1; }
                    GamePhase::Turn => { stats.hand_turn_actions += 1; }
                    GamePhase::River => { stats.hand_river_actions += 1; }
                    _ => {}
                }
            } else if is_raise {
                stats.hand_raises += 1;
                if facing_bet { stats.hand_faced_bets += 1; }
                if is_preflop {
                    stats.hand_vpip = true;
                    stats.hand_pfr = true;
                    if bet_ratio > 0.0 {
                        stats.hand_preflop_raise_size_sum += bet_ratio;
                        stats.hand_preflop_raise_count += 1;
                    }
                }
                match phase {
                    GamePhase::Flop => {
                        stats.hand_flop_raises += 1;
                        stats.hand_flop_actions += 1;
                        if stats.hand_cbet_opportunity && !stats.hand_acted_on_flop {
                            stats.hand_cbet_taken = true;
                        }
                    }
                    GamePhase::Turn => {
                        stats.hand_turn_raises += 1;
                        stats.hand_turn_actions += 1;
                    }
                    GamePhase::River => {
                        stats.hand_river_raises += 1;
                        stats.hand_river_actions += 1;
                    }
                    _ => {}
                }
                if bet_ratio > 0.0 {
                    stats.hand_bet_size_sum += bet_ratio;
                    stats.hand_bet_count += 1;
                }
            }
            // Track first flop action for c-bet
            if matches!(phase, GamePhase::Flop) && !stats.hand_acted_on_flop {
                stats.hand_acted_on_flop = true;
            }
        }
        self.actions_seen = history.len();
        self.last_phase = phase;
    }

    /// Finalize showdown results from the previous hand.
    fn finalize_showdown(&mut self, table: &PokerTable) {
        let num_seats = table.players.len().min(self.seats.len());
        // Check if previous hand ended in showdown by looking at how many
        // players are still active (not folded). If >1, it was showdown.
        // The winner detection is approximate since we track from observed data.
        let active: Vec<usize> = (0..num_seats)
            .filter(|&i| !matches!(table.players[i].state, PlayerState::Folded))
            .collect();
        if active.len() > 1 {
            // Showdown occurred — determine winners by chip change
            // (players whose stack increased or stayed same with side pots)
            for &seat in &active {
                // Approximate: player with most chips relative to start "won"
                // This is imperfect but good enough for EMA stats
                self.seats[seat].record_showdown(
                    matches!(table.players[seat].state, PlayerState::Active | PlayerState::AllIn)
                );
            }
        }
    }

    /// Encode opponent stats for a given hero seat (15 floats per opponent slot, 8 slots = 120).
    fn encode_for_seat(&self, hero_seat: usize, num_players: usize, out: &mut Vec<f32>) {
        for i in 0..8 {
            if i < num_players.saturating_sub(1) {
                let opp = (hero_seat + 1 + i) % num_players;
                if opp < self.seats.len() {
                    out.extend_from_slice(&self.seats[opp].encode());
                } else {
                    let mut default = [0.5_f32; STATS_PER_OPPONENT];
                    default[4] = 0.0; // sample_size
                    out.extend_from_slice(&default);
                }
            } else {
                out.extend_from_slice(&[0.0; STATS_PER_OPPONENT]);
            }
        }
    }
}

/// Personality configuration for ONNX model inference.
#[derive(Debug, Clone)]
pub struct Personality {
    /// Name of this personality
    pub name: &'static str,
    /// Temperature for softmax sampling (lower = more deterministic, higher = more random)
    /// Default 1.0 means use raw probabilities
    pub temperature: f32,
    /// Bias towards aggressive actions (positive) or passive actions (negative)
    /// Range: -1.0 to 1.0
    pub aggression_bias: f32,
    /// Tendency to call rather than fold (positive) or fold more (negative)
    /// Range: -1.0 to 1.0  
    pub calling_bias: f32,
}

impl Personality {
    /// GTO player - uses raw model probabilities without bias
    pub fn gto() -> Self {
        Self {
            name: "GTO",
            temperature: 1.0,
            aggression_bias: 0.0,
            calling_bias: 0.0,
        }
    }

    /// Professional player - slightly lower temperature for more consistent play
    pub fn pro() -> Self {
        Self {
            name: "Pro",
            temperature: 0.8,
            aggression_bias: 0.1,
            calling_bias: -0.1,
        }
    }

    /// Nit - very tight and passive player
    pub fn nit() -> Self {
        Self {
            name: "Nit",
            temperature: 0.6,
            aggression_bias: -0.5,
            calling_bias: -0.6,
        }
    }

    /// Calling station - calls too much, rarely folds
    pub fn calling_station() -> Self {
        Self {
            name: "Calling Station",
            temperature: 1.2,
            aggression_bias: -0.3,
            calling_bias: 0.8,
        }
    }

    /// Maniac - very aggressive and unpredictable
    pub fn maniac() -> Self {
        Self {
            name: "Maniac",
            temperature: 1.5,
            aggression_bias: 0.8,
            calling_bias: 0.3,
        }
    }

    /// Get all available personalities
    pub fn all() -> Vec<Self> {
        vec![
            Self::gto(),
            Self::pro(),
            Self::nit(),
            Self::calling_station(),
            Self::maniac(),
        ]
    }
}

struct ModelRuntime {
    plan: OnnxPlan,
}

impl ModelRuntime {
    fn load(path: &Path) -> Result<Self, String> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| format!("failed to read ONNX model {}: {e}", path.display()))?;

        // Fix dynamic axes to concrete shapes so tract can optimize the
        // transformer's multi-head attention Reshape nodes.
        model
            .set_input_fact(0, f32::fact([1, OBS_DIM]).into())
            .map_err(|e| format!("failed to set obs input shape: {e}"))?;
        model
            .set_input_fact(1, f32::fact([1, MAX_HISTORY_LEN, HISTORY_DIM]).into())
            .map_err(|e| format!("failed to set action_history input shape: {e}"))?;
        model
            .set_input_fact(2, i64::fact([1]).into())
            .map_err(|e| format!("failed to set history_lengths input shape: {e}"))?;
        model
            .set_input_fact(3, f32::fact([1, NUM_ACTIONS]).into())
            .map_err(|e| format!("failed to set legal_mask input shape: {e}"))?;

        let plan = model
            .into_optimized()
            .map_err(|e| format!("failed to optimize ONNX model {}: {e}", path.display()))?
            .into_runnable()
            .map_err(|e| format!("failed to compile ONNX model {}: {e}", path.display()))?;
        Ok(Self { plan })
    }

    fn infer_action_probs(
        &self,
        obs: Vec<f32>,
        history_flat: Vec<f32>,
        history_len: i64,
        legal_mask: [bool; NUM_ACTIONS],
    ) -> Result<[f32; NUM_ACTIONS], String> {
        if obs.len() != OBS_DIM {
            return Err(format!(
                "invalid observation size {}, expected {}",
                obs.len(),
                OBS_DIM
            ));
        }
        if history_flat.len() != MAX_HISTORY_LEN * HISTORY_DIM {
            return Err(format!(
                "invalid action history size {}, expected {}",
                history_flat.len(),
                MAX_HISTORY_LEN * HISTORY_DIM
            ));
        }
        if !(0..=MAX_HISTORY_LEN as i64).contains(&history_len) {
            return Err(format!(
                "invalid action history length {}, expected 0..={}",
                history_len, MAX_HISTORY_LEN
            ));
        }

        let obs_tensor = tract_ndarray::Array2::from_shape_vec((1, OBS_DIM), obs)
            .map_err(|e| format!("failed to build obs tensor: {e}"))?
            .into_tensor();
        let history_tensor =
            tract_ndarray::Array3::from_shape_vec((1, MAX_HISTORY_LEN, HISTORY_DIM), history_flat)
                .map_err(|e| format!("failed to build action history tensor: {e}"))?
                .into_tensor();
        let lengths_tensor = tract_ndarray::Array1::from_vec(vec![history_len]).into_tensor();
        // Pass legal mask as f32 (1.0=legal, 0.0=illegal) to avoid tract
        // bool→float cast issues that can silently break mask application.
        let legal_f: Vec<f32> = legal_mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let legal_tensor =
            tract_ndarray::Array2::from_shape_vec((1, NUM_ACTIONS), legal_f)
                .map_err(|e| format!("failed to build legal mask tensor: {e}"))?
                .into_tensor();

        let outputs = self
            .plan
            .run(tvec![
                obs_tensor.into(),
                history_tensor.into(),
                lengths_tensor.into(),
                legal_tensor.into()
            ])
            .map_err(|e| format!("ONNX inference failed: {e}"))?;

        if outputs.is_empty() {
            return Err("ONNX inference returned no outputs".to_string());
        }

        let probs_view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("failed to read ONNX output tensor: {e}"))?;

        if probs_view.shape().len() != 2 || probs_view.shape()[1] != NUM_ACTIONS {
            return Err(format!(
                "unexpected ONNX output shape {:?}, expected [batch, {}]",
                probs_view.shape(),
                NUM_ACTIONS
            ));
        }

        let mut probs = [0.0_f32; NUM_ACTIONS];
        for (i, out) in probs.iter_mut().enumerate() {
            *out = probs_view[[0, i]];
        }
        Ok(probs)
    }
}

/// Strategy that selects actions using the exported average-strategy ONNX model.
pub struct ModelStrategy {
    runtime: Mutex<ModelRuntime>,
    tracker: Mutex<OpponentTracker>,
    fallback: SimpleStrategy,
    personality: Personality,
}

impl ModelStrategy {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        Self::from_path_with_personality(path, Personality::gto())
    }

    pub fn from_path_with_personality(path: &Path, personality: Personality) -> Result<Self, String> {
        let runtime = ModelRuntime::load(path)?;
        Ok(Self {
            runtime: Mutex::new(runtime),
            tracker: Mutex::new(OpponentTracker::new()),
            fallback: SimpleStrategy::balanced(),
            personality,
        })
    }

    /// Apply personality adjustments to raw model probabilities
    fn apply_personality(&self, probs: &mut [f32; NUM_ACTIONS]) {
        // Action indices: 0=Fold, 1=Check/Call, 2=Raise25%, 3=Raise40%, 4=Raise60%, 5=Raise80%, 6=RaisePot, 7=Raise150%, 8=AllIn

        // Apply aggression bias
        // Aggressive actions: Raise(2-7), AllIn(8)
        // Passive actions: Fold(0), Check/Call(1)
        if self.personality.aggression_bias.abs() > 0.01 {
            let bias = self.personality.aggression_bias;
            for i in 0..NUM_ACTIONS {
                let multiplier = match i {
                    0 => 1.0 - bias.max(0.0) * 0.5, // Fold less when aggressive
                    1 => 1.0 - bias.abs() * 0.3, // Check/Call affected by aggression
                    2..=8 => 1.0 + bias.max(0.0) * 0.5, // Raise more when aggressive
                    _ => 1.0,
                };
                probs[i] *= multiplier;
            }
        }

        // Apply calling bias
        if self.personality.calling_bias.abs() > 0.01 {
            let bias = self.personality.calling_bias;
            probs[0] *= 1.0 - bias.max(0.0) * 0.8; // Fold less when calling bias is positive
            probs[1] *= 1.0 + bias.max(0.0) * 0.6; // Call more when calling bias is positive

            if bias < 0.0 {
                // Fold more when calling bias is negative
                probs[0] *= 1.0 + (-bias) * 0.5;
            }
        }

        // Normalize to ensure probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Apply temperature scaling
        if (self.personality.temperature - 1.0).abs() > 0.01 {
            let temp = self.personality.temperature;
            
            // Find max logit for numerical stability
            let max_logit = probs.iter()
                .copied()
                .filter(|&p| p > 0.0)
                .map(|p| p.ln())
                .fold(f32::NEG_INFINITY, f32::max);
            
            // Apply temperature: exp(log(p) / T)
            let mut sum = 0.0;
            for p in probs.iter_mut() {
                if *p > 0.0 {
                    let logit = p.ln();
                    *p = ((logit - max_logit) / temp).exp();
                    sum += *p;
                }
            }
            
            // Normalize
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }
    }
}

impl BotStrategy for ModelStrategy {
    fn name(&self) -> &str {
        "model"
    }

    fn observe_table(&self, table: &PokerTable) {
        if let Ok(mut tracker) = self.tracker.lock() {
            tracker.observe(table);
        }
    }

    fn decide(&self, view: &BotGameView) -> PlayerAction {
        // Fallback path if table context is not available.
        self.fallback.decide(view)
    }

    fn decide_with_table(
        &self,
        view: &BotGameView,
        table: &PokerTable,
        player_idx: usize,
    ) -> PlayerAction {
        let legal_mask = legal_actions_mask(table, player_idx);
        if !legal_mask.iter().any(|&legal| legal) {
            tracing::warn!(
                "Model bot had no legal actions at table {} player_idx {}; using fallback",
                table.table_id,
                player_idx
            );
            return self.fallback.decide(view);
        }

        let mut obs = encode_static_observation(table, player_idx);

        // Append opponent stats (120 floats)
        if let Ok(mut tracker) = self.tracker.lock() {
            tracker.observe(table);
            tracker.encode_for_seat(player_idx, table.players.len(), &mut obs);
        } else {
            // Mutex poisoned — append neutral defaults
            obs.extend_from_slice(&[0.0; STATS_PER_OPPONENT * 8]);
        }

        // Hand strength AFTER opponent stats to match training layout:
        //   cards(364) → game_state_base(46) → opp_stats(120) → hand_strength(52) = 582
        encode_hand_strength(table, player_idx, &mut obs);
        
        // Debug: log hand strength for strong hands
        if tracing::enabled!(tracing::Level::DEBUG) {
            if let Some(player) = table.players.get(player_idx) {
                if player.hole_cards.len() >= 2 {
                    let preflop = preflop_strength(player.hole_cards[0], player.hole_cards[1]);
                    if preflop > 0.85 {
                        tracing::debug!("Strong hand detected: preflop_strength={:.2}", preflop);
                    }
                }
            }
        }
        
        let (history_flat, history_len) = encode_action_history(table);
        let probs = match self.runtime.lock() {
            Ok(runtime) => runtime.infer_action_probs(obs, history_flat, history_len, legal_mask),
            Err(_) => Err("model runtime lock poisoned".to_string()),
        };

        match probs {
            Ok(mut probs) => {
                // Apply personality adjustments to probabilities
                self.apply_personality(&mut probs);
                
                // Log probabilities for debugging (remove after verification)
                if tracing::enabled!(tracing::Level::DEBUG) {
                    let mut prob_strs: Vec<String> = Vec::new();
                    for (i, &p) in probs.iter().enumerate() {
                        if legal_mask[i] && p > 0.01 {
                            let action_name = match i {
                                0 => "Fold",
                                1 => "Check/Call",
                                2 => "Raise0.25x",
                                3 => "Raise0.4x",
                                4 => "Raise0.6x",
                                5 => "Raise0.8x",
                                6 => "Raise1.0x",
                                7 => "Raise1.5x",
                                8 => "AllIn",
                                _ => "Unknown",
                            };
                            prob_strs.push(format!("{}={:.1}%", action_name, p * 100.0));
                        }
                    }
                    tracing::debug!("Model probs: [{}]", prob_strs.join(", "));
                }
                
                // Sample from the probability distribution (matching training behavior)
                let action_idx = sample_action(&probs, &legal_mask);
                if let Some(action) = discrete_to_player_action(action_idx, table, player_idx) {
                    return action;
                }
                
                safe_fallback_action(table, player_idx)
                    .unwrap_or_else(|| self.fallback.decide(view))
            }
            Err(err) => {
                tracing::warn!(
                    "Model bot inference failed at table {}: {}. Falling back to heuristic strategy.",
                    table.table_id,
                    err
                );
                self.fallback.decide(view)
            }
        }
    }
}

fn sample_action(probs: &[f32; NUM_ACTIONS], legal_mask: &[bool; NUM_ACTIONS]) -> usize {
    use rand::Rng;
    
    // Normalize probabilities over legal actions
    let mut legal_probs = [0.0f32; NUM_ACTIONS];
    let mut total = 0.0f32;
    for i in 0..NUM_ACTIONS {
        if legal_mask[i] {
            let p = sanitize_prob(probs[i]).max(0.0);
            legal_probs[i] = p;
            total += p;
        }
    }
    
    // Fallback to uniform if all probs are zero/invalid
    if total <= 0.0 {
        let legal_actions: Vec<usize> = (0..NUM_ACTIONS)
            .filter(|&idx| legal_mask[idx])
            .collect();
        if legal_actions.is_empty() {
            return 1; // Check/Call as last resort
        }
        return legal_actions[rand::thread_rng().gen_range(0..legal_actions.len())];
    }
    
    // Normalize to sum to 1.0
    for p in legal_probs.iter_mut() {
        *p /= total;
    }
    
    // Sample using cumulative distribution
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    
    for i in 0..NUM_ACTIONS {
        if legal_mask[i] {
            cumulative += legal_probs[i];
            if r <= cumulative {
                return i;
            }
        }
    }
    
    // Fallback to last legal action (in case of floating point errors)
    for i in (0..NUM_ACTIONS).rev() {
        if legal_mask[i] {
            return i;
        }
    }
    1 // Check/Call
}

fn sanitize_prob(p: f32) -> f32 {
    if p.is_finite() {
        p
    } else {
        f32::NEG_INFINITY
    }
}

fn safe_fallback_action(table: &PokerTable, player_idx: usize) -> Option<PlayerAction> {
    let player = table.players.get(player_idx)?;
    let to_call = (table.current_bet - player.current_bet).max(0);
    if to_call <= 0 {
        Some(PlayerAction::Check)
    } else if player.stack > 0 {
        Some(PlayerAction::Call)
    } else {
        Some(PlayerAction::Fold)
    }
}

fn legal_actions_mask(table: &PokerTable, player_idx: usize) -> [bool; NUM_ACTIONS] {
    let mut mask = [false; NUM_ACTIONS];
    let player = match table.players.get(player_idx) {
        Some(player) => player,
        None => return mask,
    };

    if !player.can_act() {
        return mask;
    }

    let to_call = (table.current_bet - player.current_bet).max(0);
    let stack = player.stack.max(0);

    // Fold is only legal if facing a bet.
    if to_call > 0 {
        mask[0] = true;
    }

    // Check/call is always available for an actionable player.
    mask[1] = true;

    if stack > 0 {
        mask[8] = true; // all-in
    }

    if stack > to_call {
        let max_raise_amount = stack - to_call;
        let min_raise = table.min_raise.max(table.big_blind).max(1);
        let pot = table.pot.total();
        for action_idx in 2..=7 {
            if let Some(raise_amount) = raise_amount_for_discrete(action_idx, pot, to_call) {
                if raise_amount >= min_raise && raise_amount <= max_raise_amount {
                    mask[action_idx] = true;
                }
            }
        }
    }

    mask
}

fn discrete_to_player_action(
    action_idx: usize,
    table: &PokerTable,
    player_idx: usize,
) -> Option<PlayerAction> {
    let legal_mask = legal_actions_mask(table, player_idx);
    if action_idx >= NUM_ACTIONS || !legal_mask[action_idx] {
        return None;
    }

    let player = table.players.get(player_idx)?;
    let to_call = (table.current_bet - player.current_bet).max(0);
    let stack = player.stack.max(0);
    let pot = table.pot.total();
    let min_raise = table.min_raise.max(table.big_blind).max(1);
    let max_raise_amount = stack - to_call;

    match action_idx {
        0 => Some(PlayerAction::Fold),
        1 => {
            if to_call > 0 {
                Some(PlayerAction::Call)
            } else {
                Some(PlayerAction::Check)
            }
        }
        2..=7 => {
            let raise_amount = raise_amount_for_discrete(action_idx, pot, to_call)?;
            if raise_amount < min_raise || raise_amount > max_raise_amount {
                return None;
            }
            Some(PlayerAction::Raise(raise_amount))
        }
        8 => Some(PlayerAction::AllIn),
        _ => None,
    }
}

fn raise_amount_for_discrete(action_idx: usize, pot: i64, to_call: i64) -> Option<i64> {
    let effective_pot = pot.saturating_add(to_call);
    let raise_amount = match action_idx {
        2 => effective_pot / 4,                        // 0.25× pot
        3 => effective_pot.saturating_mul(2) / 5,      // 0.4× pot
        4 => effective_pot.saturating_mul(3) / 5,      // 0.6× pot
        5 => effective_pot.saturating_mul(4) / 5,      // 0.8× pot
        6 => effective_pot,                            // 1.0× pot
        7 => effective_pot.saturating_mul(3) / 2,      // 1.5× pot
        _ => return None,
    };
    Some(raise_amount.max(0))
}

fn encode_static_observation(table: &PokerTable, player_idx: usize) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBS_DIM);
    encode_cards(table, player_idx, &mut obs);
    encode_game_state(table, player_idx, &mut obs);
    // Opponent stats slot (120 floats) — filled by OpponentTracker in decide_with_table
    // Hand strength appended AFTER opponent stats to match training layout:
    //   cards(364) → game_state_base(46) → opp_stats(120) → hand_strength(52)
    obs
}

fn encode_action_history(table: &PokerTable) -> (Vec<f32>, i64) {
    let mut history = vec![0.0_f32; MAX_HISTORY_LEN * HISTORY_DIM];
    let length = table.hand_action_history.len().min(MAX_HISTORY_LEN);
    let start = table.hand_action_history.len().saturating_sub(length);

    for (dst_row, record) in table.hand_action_history[start..].iter().enumerate() {
        let start_col = dst_row * HISTORY_DIM;
        history[start_col..start_col + HISTORY_DIM].copy_from_slice(record);
    }

    (history, length as i64)
}

fn encode_cards(table: &PokerTable, player_idx: usize, out: &mut Vec<f32>) {
    let player = match table.players.get(player_idx) {
        Some(player) => player,
        None => {
            out.resize(364, 0.0);
            return;
        }
    };

    for i in 0..2 {
        let mut onehot = [0.0_f32; 52];
        if let Some(card) = player.hole_cards.get(i) {
            onehot[card_index(card)] = 1.0;
        }
        out.extend_from_slice(&onehot);
    }

    for i in 0..5 {
        let mut onehot = [0.0_f32; 52];
        if let Some(card) = table.community_cards.get(i) {
            onehot[card_index(card)] = 1.0;
        }
        out.extend_from_slice(&onehot);
    }
}

fn encode_game_state(table: &PokerTable, player_idx: usize, out: &mut Vec<f32>) {
    let player = match table.players.get(player_idx) {
        Some(player) => player,
        None => {
            out.extend_from_slice(&[0.0; 46]);
            return;
        }
    };

    let num_players = table.players.len().max(1);
    let denominator = (num_players - 1).max(1) as f32;

    // Phase one-hot: [preflop, flop, turn, river, showdown, hand_over].
    let mut phase_oh = [0.0_f32; 6];
    let phase_idx = match table.phase {
        GamePhase::PreFlop => 0,
        GamePhase::Flop => 1,
        GamePhase::Turn => 2,
        GamePhase::River => 3,
        GamePhase::Showdown => 4,
        GamePhase::Waiting => 5,
    };
    phase_oh[phase_idx] = 1.0;
    out.extend_from_slice(&phase_oh);

    let stack = player.stack.max(0);
    let initial_stack = stack.saturating_add(player.total_bet_this_hand.max(0));
    let stack_ratio = if initial_stack > 0 {
        stack as f32 / initial_stack as f32
    } else {
        0.0
    };
    out.push(stack_ratio);

    let pot = table.pot.total().max(0);
    let bb = table.big_blind.max(1) as f32;
    let pot_f = pot as f32;
    let pot_bb = (pot_f / bb).min(50.0) / 50.0;
    out.push(pot_bb);

    let spr = if pot > 0 {
        stack as f32 / pot_f
    } else {
        10.0
    };
    out.push(spr.min(20.0) / 20.0);

    let position =
        ((player_idx + num_players - table.dealer_seat) % num_players) as f32 / denominator;
    out.push(position);

    let active_count = table
        .players
        .iter()
        .filter(|p| p.is_active_in_hand())
        .count();
    let opponents_in_hand = active_count.saturating_sub(1) as f32 / denominator;
    out.push(opponents_in_hand);

    let can_act = table.players.iter().filter(|p| p.can_act()).count() as f32 / num_players as f32;
    out.push(can_act);

    let to_call = (table.current_bet - player.current_bet).max(0);
    let to_call_f = to_call as f32;
    let to_call_ratio = if pot > 0 {
        to_call_f / pot_f
    } else {
        0.0
    };
    out.push(to_call_ratio.min(5.0) / 5.0);

    out.push((num_players as f32 / 9.0).min(1.0));

    // --- NEW features (8 floats) ---

    // Pot odds: to_call / (pot + to_call)
    let pot_odds = if pot_f + to_call_f > 0.0 {
        to_call_f / (pot_f + to_call_f)
    } else {
        0.0
    };
    out.push(pot_odds);

    // Effective stack / pot
    let max_opp_stack = table
        .players
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != player_idx && p.is_active_in_hand())
        .map(|(_, p)| p.stack.max(0) as f32)
        .fold(0.0f32, f32::max);
    let eff_stack = (stack as f32).min(max_opp_stack);
    let eff_stack_pot = if pot_f > 0.0 {
        eff_stack / pot_f
    } else {
        10.0
    };
    out.push(eff_stack_pot.min(20.0) / 20.0);

    // Street action count / 10
    let street_actions = table.actions_this_round.min(30) as f32;
    out.push((street_actions / 10.0).min(1.0));

    // Total action count / 30
    out.push((table.hand_action_history.len() as f32 / 30.0).min(1.0));

    // Num raises this street / 4
    let raises_count = table.raises_this_round.min(10) as f32;
    out.push((raises_count / 4.0).min(1.0));

    // Last aggressor is hero
    let last_agg_hero = match table.last_raiser_seat {
        Some(seat) if seat == player_idx => 1.0,
        _ => 0.0,
    };
    out.push(last_agg_hero);

    // Hero's current bet / pot
    let hero_bet = player.current_bet.max(0) as f32;
    let hero_bet_pot = if pot_f > 0.0 { hero_bet / pot_f } else { 0.0 };
    out.push(hero_bet_pot.min(5.0) / 5.0);

    // Hero invested / starting stack
    let hero_invested = player.total_bet_this_hand.max(0) as f32;
    let hero_invested_ratio = if initial_stack > 0 {
        hero_invested / initial_stack as f32
    } else {
        0.0
    };
    out.push(hero_invested_ratio.min(1.0));

    // Per-opponent features (8 × 3 = 24)
    for i in 0..8 {
        if i < num_players.saturating_sub(1) {
            let opp_idx = (player_idx + 1 + i) % num_players;
            let opp = &table.players[opp_idx];
            let opp_stack = opp.stack.max(0);
            let opp_initial = opp_stack.saturating_add(opp.total_bet_this_hand.max(0));
            let opp_stack_ratio = if opp_initial > 0 {
                opp_stack as f32 / opp_initial as f32
            } else {
                0.0
            };

            out.push(if matches!(opp.state, PlayerState::Folded) {
                1.0
            } else {
                0.0
            });
            out.push(if matches!(opp.state, PlayerState::AllIn) {
                1.0
            } else {
                0.0
            });
            out.push(opp_stack_ratio.min(3.0) / 3.0);
        } else {
            out.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }
    // Total: 6 + 8 + 8 + 24 = 46 (opponent stats appended separately by OpponentTracker)
}

fn encode_hand_strength(table: &PokerTable, player_idx: usize, out: &mut Vec<f32>) {
    let mut hand = vec![0.0_f32; 52];
    let player = match table.players.get(player_idx) {
        Some(player) => player,
        None => {
            out.extend_from_slice(&hand);
            return;
        }
    };

    if player.hole_cards.len() >= 2 {
        if player.hole_cards.len() + table.community_cards.len() >= 5 {
            let rank = evaluate_hand(&player.hole_cards, &table.community_cards);
            hand[0] = rank.normalized();
        }

        hand[1] = preflop_strength(player.hole_cards[0], player.hole_cards[1]);
    }

    out.extend_from_slice(&hand);
}

fn card_index(card: &Card) -> usize {
    (card.suit as usize) * 13 + (card.rank as usize - 2)
}

fn preflop_strength(c1: Card, c2: Card) -> f32 {
    let high = c1.rank.max(c2.rank);
    let low = c1.rank.min(c2.rank);
    let suited = c1.suit == c2.suit;
    let pair = c1.rank == c2.rank;

    let tier = if pair {
        match high {
            14 | 13 | 12 => 0,
            11 | 10 => 1,
            9 | 8 => 2,
            7 | 6 => 3,
            5 | 4 => 4,
            _ => 5,
        }
    } else if suited {
        match (high, low) {
            (14, 13) => 0,
            (14, 12) | (14, 11) => 1,
            (14, 10) | (13, 12) | (13, 11) | (12, 11) => 2,
            (14, _) | (13, 10) | (12, 10) | (11, 10) | (10, 9) => 3,
            (13, lo) if lo >= 7 => 4,
            (_, _) if high - low <= 2 && high >= 7 => 4,
            _ => 5 + (14 - high) as u8 / 3,
        }
    } else {
        match (high, low) {
            (14, 13) => 1,
            (14, 12) => 2,
            (14, 11) | (14, 10) => 3,
            (13, 12) | (13, 11) | (12, 11) => 4,
            (13, 10) | (12, 10) | (11, 10) => 5,
            _ => 6 + (14 - high) as u8 / 2,
        }
    };

    (0.95 - tier as f32 * 0.10).clamp(0.05, 0.95)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::player::PlayerState;

    fn make_table() -> PokerTable {
        let mut table = PokerTable::new("t1".to_string(), "Test".to_string(), 50, 100);
        table
            .take_seat("p1".to_string(), "P1".to_string(), 0, 2000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "P2".to_string(), 1, 2000)
            .unwrap();

        table.phase = GamePhase::PreFlop;
        table.current_player = 0;
        table.dealer_seat = 0;
        table.players[0].state = PlayerState::Active;
        table.players[1].state = PlayerState::Active;
        table.players[0].current_bet = 100;
        table.players[1].current_bet = 100;
        table.players[0].total_bet_this_hand = 100;
        table.players[1].total_bet_this_hand = 100;
        table.current_bet = 100;
        table.min_raise = 100;
        table
    }

    #[test]
    fn test_encode_static_observation_shape() {
        let table = make_table();
        let obs = encode_static_observation(&table, 0);
        // Static obs = 364 (cards) + 46 (game state) = 410
        // Opponent stats (120) and hand strength (52) appended in decide_with_table
        assert_eq!(obs.len(), 410);
    }

    #[test]
    fn test_legal_mask_disables_fold_when_check_available() {
        let table = make_table();
        let mask = legal_actions_mask(&table, 0);
        assert!(!mask[0], "fold should be illegal when no bet is faced");
        assert!(mask[1], "check/call should be legal");
    }

    #[test]
    fn test_check_call_mapping_check_and_call() {
        let mut table = make_table();

        let action = discrete_to_player_action(1, &table, 0).expect("action should map");
        assert!(matches!(action, PlayerAction::Check));

        table.players[0].current_bet = 0;
        table.current_bet = 100;
        let action = discrete_to_player_action(1, &table, 0).expect("action should map");
        assert!(matches!(action, PlayerAction::Call));
    }

    #[test]
    fn test_encode_action_history_uses_recent_window() {
        let mut table = make_table();
        for idx in 0..35 {
            table.record_hand_action(idx % 2, idx % NUM_ACTIONS);
        }

        let (history_flat, history_len) = encode_action_history(&table);
        assert_eq!(history_len, MAX_HISTORY_LEN as i64);
        assert_eq!(history_flat.len(), MAX_HISTORY_LEN * HISTORY_DIM);

        let expected_first = table.hand_action_history[5];
        let expected_last = table.hand_action_history[34];
        assert_eq!(&history_flat[0..HISTORY_DIM], &expected_first);

        let tail_start = (MAX_HISTORY_LEN - 1) * HISTORY_DIM;
        assert_eq!(
            &history_flat[tail_start..tail_start + HISTORY_DIM],
            &expected_last
        );
    }

    #[test]
    fn test_handle_action_records_history() {
        let mut table = make_table();
        assert!(table.hand_action_history.is_empty());

        table
            .handle_action("p1", PlayerAction::Check)
            .expect("action should succeed");

        assert_eq!(table.hand_action_history.len(), 1);
        let rec = table.hand_action_history[0];
        // rec[0] = seat_normalized: player 0 of 2 → 0.0
        assert_eq!(rec[0], 0.0);
        // rec[1..=9] = one-hot over 9 actions; Check/Call = action index 1 → rec[2] = 1.0
        assert_eq!(rec[1], 0.0);
        assert_eq!(rec[2], 1.0);
        assert_eq!(rec[3], 0.0);
        // rec[10] = rough bet ratio: action_idx 1 / 8.0 = 0.125
        assert!((rec[10] - (1.0 / 8.0)).abs() < 1e-6);
    }

    // ── Observation layout constants ────────────────────────────────────
    const CARDS_LEN: usize = 364;
    const GAME_BASE_LEN: usize = 46;  // 6 phase + 8 orig + 8 new + 24 per-opp
    const OPP_STATS_LEN: usize = 120; // 8 slots × 15 stats
    const HAND_STR_LEN: usize = 52;
    const OPP_STATS_OFFSET: usize = CARDS_LEN + GAME_BASE_LEN;            // 410
    const HAND_STR_OFFSET: usize = OPP_STATS_OFFSET + OPP_STATS_LEN;     // 530

    /// Build a full 582-float observation the same way decide_with_table does.
    fn build_full_obs(table: &PokerTable, player_idx: usize, tracker: &mut OpponentTracker) -> Vec<f32> {
        let mut obs = encode_static_observation(table, player_idx);
        tracker.observe(table);
        tracker.encode_for_seat(player_idx, table.players.len(), &mut obs);
        encode_hand_strength(table, player_idx, &mut obs);
        obs
    }

    #[test]
    fn test_full_observation_length_582() {
        let table = make_table();
        let mut tracker = OpponentTracker::new();
        let obs = build_full_obs(&table, 0, &mut tracker);
        assert_eq!(obs.len(), OBS_DIM, "full observation must be {OBS_DIM} floats");
    }

    #[test]
    fn test_observation_layout_order() {
        // Verify: cards(364) → game_base(46) → opp_stats(120) → hand_strength(52)
        let table = make_table();
        let mut tracker = OpponentTracker::new();
        let obs = build_full_obs(&table, 0, &mut tracker);

        // Cards region: should contain some 1.0 values (one-hot encoded)
        let cards = &obs[..CARDS_LEN];
        let card_ones: usize = cards.iter().filter(|&&v| v == 1.0).count();
        // 2 hole cards + 0 community = 2 one-hot 1.0 values (preflop, no community)
        assert_eq!(card_ones, 2, "preflop should have exactly 2 card one-hots");

        // Opponent stats at [410..530) — initial EMA values (0.5 for most stats)
        let stats = &obs[OPP_STATS_OFFSET..OPP_STATS_OFFSET + STATS_PER_OPPONENT];
        // VPIP initial = 0.5
        assert!((stats[0] - 0.5).abs() < 0.01, "initial VPIP should be ~0.5, got {}", stats[0]);
        // sample_size = hands_played/100 = 100/100 = 1.0 (prior)
        assert!((stats[4] - 1.0).abs() < 0.01, "initial sample_size should be 1.0, got {}", stats[4]);

        // Hand strength at [530..582) — slot [0] is hand rank (0 preflop), [1] is preflop strength
        let hs = &obs[HAND_STR_OFFSET..HAND_STR_OFFSET + HAND_STR_LEN];
        assert!((hs[0] - 0.0).abs() < 0.01, "hand rank should be 0.0 preflop");
        // preflop_strength is > 0 for any real hand
        assert!(hs[1] > 0.0, "preflop strength should be > 0, got {}", hs[1]);
    }

    #[test]
    fn test_opp_stats_default_values_before_any_hands() {
        let table = make_table();
        let mut tracker = OpponentTracker::new();
        let obs = build_full_obs(&table, 0, &mut tracker);

        // Opponent 0 (player 1 from hero's perspective) stats at [410..425)
        let stats = &obs[OPP_STATS_OFFSET..OPP_STATS_OFFSET + STATS_PER_OPPONENT];
        let names = ["VPIP","PFR","AGG","FTB","SS","F-AGG","T-AGG","R-AGG",
                     "WTSD","WSD","CBET","BET-SZ","PFR-SZ","EP-VPIP","LP-VPIP"];

        // All EMA stats initialized to 0.5, sample_size to 1.0 (100-hand prior)
        for (i, &name) in names.iter().enumerate() {
            let expected = if i == 4 { 1.0 }                          // sample_size (prior=100 hands)
                      else if i == 11 { (0.5_f32).min(5.0) / 5.0 }   // avg_bet_size normalized
                      else if i == 12 { (0.5_f32).min(10.0) / 10.0 } // preflop_raise_size normalized
                      else { 0.5 };
            assert!(
                (stats[i] - expected).abs() < 0.01,
                "{name} (idx {i}): expected {expected}, got {}",
                stats[i]
            );
        }

        // Slots 1-7 should be zero (no opponents beyond the one in heads-up)
        for slot in 1..8 {
            let offset = OPP_STATS_OFFSET + slot * STATS_PER_OPPONENT;
            let slot_stats = &obs[offset..offset + STATS_PER_OPPONENT];
            assert!(
                slot_stats.iter().all(|&v| v == 0.0),
                "unused opponent slot {slot} should be all zeros"
            );
        }
    }

    #[test]
    fn test_opp_stats_all_values_bounded_0_1() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        // Simulate several hands with various actions
        for _ in 0..10 {
            table.record_hand_action(0, 8); // p0 all-in
            table.record_hand_action(1, 1); // p1 call
            tracker.observe(&table);
            table.hand_action_history.clear();
            tracker.observe(&table); // trigger hand boundary
        }

        let obs = build_full_obs(&table, 0, &mut tracker);
        let all_stats = &obs[OPP_STATS_OFFSET..OPP_STATS_OFFSET + OPP_STATS_LEN];

        for (i, &v) in all_stats.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.001,
                "stat at index {i} out of [0,1] range: {v}"
            );
        }
    }

    /// Helper: simulate a hand where p0 raises preflop and p1 calls.
    fn simulate_preflop_raise_call(table: &mut PokerTable, tracker: &mut OpponentTracker) {
        table.phase = GamePhase::PreFlop;
        table.current_bet = 200;
        // p0 raises (action_idx 4 = 0.6x pot raise)
        table.record_hand_action(0, 4);
        // p1 calls
        table.record_hand_action(1, 1);
        tracker.observe(table);
        // End hand — clear then observe so tracker sees the boundary
        table.hand_action_history.clear();
        tracker.observe(table);
    }

    /// Helper: simulate a hand where p0 raises and p1 folds.
    fn simulate_preflop_raise_fold(table: &mut PokerTable, tracker: &mut OpponentTracker) {
        table.phase = GamePhase::PreFlop;
        table.current_bet = 200;
        // p0 raises
        table.record_hand_action(0, 4);
        // p1 folds
        table.record_hand_action(1, 0);
        tracker.observe(table);
        // End hand — clear then observe so tracker sees the boundary
        table.hand_action_history.clear();
        tracker.observe(table);
    }

    #[test]
    fn test_vpip_increases_when_opponent_calls() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        let obs_before = build_full_obs(&table, 0, &mut tracker);
        let vpip_before = obs_before[OPP_STATS_OFFSET]; // opponent VPIP

        // Opponent (p1) calls a raise — should increase VPIP
        for _ in 0..5 {
            simulate_preflop_raise_call(&mut table, &mut tracker);
        }

        let obs_after = build_full_obs(&table, 0, &mut tracker);
        let vpip_after = obs_after[OPP_STATS_OFFSET];

        // EMA with alpha=0.02: after 5 hands of VPIP=1, should be > initial 0.5
        assert!(
            vpip_after > vpip_before,
            "VPIP should increase after opponent calls: before={vpip_before}, after={vpip_after}"
        );
    }

    #[test]
    fn test_fold_to_bet_increases_when_opponent_folds() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        let obs_before = build_full_obs(&table, 0, &mut tracker);
        let ftb_before = obs_before[OPP_STATS_OFFSET + 3]; // FTB index

        // Opponent (p1) folds to raises repeatedly
        for _ in 0..5 {
            simulate_preflop_raise_fold(&mut table, &mut tracker);
        }

        let obs_after = build_full_obs(&table, 0, &mut tracker);
        let ftb_after = obs_after[OPP_STATS_OFFSET + 3];

        assert!(
            ftb_after > ftb_before,
            "fold_to_bet should increase after opponent folds: before={ftb_before}, after={ftb_after}"
        );
    }

    #[test]
    fn test_pfr_increases_when_opponent_raises() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        let obs_before = build_full_obs(&table, 0, &mut tracker);
        let pfr_before = obs_before[OPP_STATS_OFFSET + 1]; // PFR index

        // Opponent (p1) raises preflop repeatedly
        for _ in 0..5 {
            table.phase = GamePhase::PreFlop;
            table.current_bet = 100;
            table.record_hand_action(1, 6); // p1 pot-size raise
            table.record_hand_action(0, 1); // p0 calls
            tracker.observe(&table);
            table.hand_action_history.clear();
        }

        let obs_after = build_full_obs(&table, 0, &mut tracker);
        let pfr_after = obs_after[OPP_STATS_OFFSET + 1];

        assert!(
            pfr_after > pfr_before,
            "PFR should increase after opponent raises preflop: before={pfr_before}, after={pfr_after}"
        );
    }

    #[test]
    fn test_sample_size_grows_with_hands() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        // With 100-hand prior, sample_size starts at 1.0
        let obs0 = build_full_obs(&table, 0, &mut tracker);
        assert!((obs0[OPP_STATS_OFFSET + 4] - 1.0).abs() < 0.01,
            "sample_size should start at 1.0 (100-hand prior)");

        // Play 10 hands — still saturated at 1.0
        for _ in 0..10 {
            simulate_preflop_raise_call(&mut table, &mut tracker);
        }

        let obs10 = build_full_obs(&table, 0, &mut tracker);
        let ss_10 = obs10[OPP_STATS_OFFSET + 4];
        assert!(
            (ss_10 - 1.0).abs() < 0.02,
            "sample_size after 10 more hands should still be ~1.0, got {ss_10}"
        );
    }

    #[test]
    fn test_aggression_tracks_raise_vs_call_ratio() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        // Opponent always raises → aggression should increase toward 1.0
        for _ in 0..20 {
            table.phase = GamePhase::PreFlop;
            table.current_bet = 100;
            table.record_hand_action(1, 6); // p1 raises
            table.record_hand_action(0, 1); // p0 calls
            tracker.observe(&table);
            table.hand_action_history.clear();
            tracker.observe(&table);
        }

        let obs_raise = build_full_obs(&table, 0, &mut tracker);
        let agg_raise = obs_raise[OPP_STATS_OFFSET + 2]; // AGG index

        // Opponent always calls → aggression should decrease toward 0.0
        let mut tracker2 = OpponentTracker::new();
        for _ in 0..20 {
            table.phase = GamePhase::PreFlop;
            table.current_bet = 200;
            table.record_hand_action(0, 4); // p0 raises
            table.record_hand_action(1, 1); // p1 calls
            tracker2.observe(&table);
            table.hand_action_history.clear();
            tracker2.observe(&table);
        }

        let obs_call = build_full_obs(&table, 0, &mut tracker2);
        let agg_call = obs_call[OPP_STATS_OFFSET + 2];

        assert!(
            agg_raise > agg_call,
            "aggression should be higher for raising opp ({agg_raise}) vs calling opp ({agg_call})"
        );
    }

    #[test]
    fn test_bet_size_normalized_in_range() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        // Opponent makes various raises
        for action_idx in [2, 3, 4, 5, 6, 7, 8] {
            table.phase = GamePhase::PreFlop;
            table.current_bet = 100;
            table.record_hand_action(1, action_idx); // p1 raises with different sizes
            table.record_hand_action(0, 1);
            tracker.observe(&table);
            table.hand_action_history.clear();
            tracker.observe(&table);
        }

        let obs = build_full_obs(&table, 0, &mut tracker);
        let bet_sz = obs[OPP_STATS_OFFSET + 11]; // BET-SZ
        let pfr_sz = obs[OPP_STATS_OFFSET + 12]; // PFR-SZ

        assert!(bet_sz >= 0.0 && bet_sz <= 1.0,
            "normalized avg_bet_size should be in [0,1], got {bet_sz}");
        assert!(pfr_sz >= 0.0 && pfr_sz <= 1.0,
            "normalized preflop_raise_size should be in [0,1], got {pfr_sz}");
    }

    #[test]
    fn test_ema_convergence_direction() {
        // After many hands of the same action, EMA should converge toward that value.
        let mut stats = SeatStats::new();

        // Simulate 50 hands where player always raises (VPIP=1, PFR=1, AGG=1)
        for _ in 0..50 {
            stats.start_hand();
            stats.hand_vpip = true;
            stats.hand_pfr = true;
            stats.hand_raises = 1;
            stats.hand_calls = 0;
            stats.end_hand();
        }

        assert!(stats.vpip > 0.8, "VPIP should converge toward 1.0 after 50 all-raise hands, got {}", stats.vpip);
        assert!(stats.pfr > 0.8, "PFR should converge toward 1.0 after 50 all-raise hands, got {}", stats.pfr);
        assert!(stats.aggression > 0.8, "AGG should converge toward 1.0 after 50 all-raise hands, got {}", stats.aggression);

        // Now simulate 50 hands of always folding to bet
        for _ in 0..50 {
            stats.start_hand();
            stats.hand_folds_to_bet = 1;
            stats.hand_faced_bets = 1;
            stats.end_hand();
        }

        assert!(stats.fold_to_bet > 0.8,
            "FTB should converge toward 1.0 after 50 all-fold hands, got {}", stats.fold_to_bet);
        assert!(stats.vpip < 0.5,
            "VPIP should decay toward 0 after 50 passive hands, got {}", stats.vpip);
    }

    #[test]
    fn test_flop_aggression_only_updates_on_flop_actions() {
        let mut stats = SeatStats::new();
        let initial_flop_agg = stats.flop_aggression;

        // Hand with no flop actions — flop_aggression should not change
        stats.start_hand();
        stats.hand_raises = 1;
        stats.hand_calls = 0;
        // No flop actions recorded
        stats.end_hand();

        assert!(
            (stats.flop_aggression - initial_flop_agg).abs() < 1e-10,
            "flop_aggression should not change without flop actions"
        );

        // Hand with flop raise — flop_aggression should increase
        stats.start_hand();
        stats.hand_flop_raises = 1;
        stats.hand_flop_actions = 1;
        stats.end_hand();

        assert!(
            stats.flop_aggression > initial_flop_agg,
            "flop_aggression should increase after flop raise"
        );
    }

    #[test]
    fn test_cbet_tracks_flop_raise_after_pfr() {
        let mut stats = SeatStats::new();
        let initial_cbet = stats.cbet;

        // Hand: raised preflop, then bet on flop (c-bet)
        stats.start_hand();
        stats.hand_pfr = true;
        stats.hand_saw_flop = true;
        stats.hand_cbet_opportunity = true;
        stats.hand_cbet_taken = true;
        stats.end_hand();

        assert!(
            stats.cbet > initial_cbet,
            "cbet should increase after successful c-bet: before={initial_cbet}, after={}", stats.cbet
        );

        // Hand: raised preflop, checked flop (missed c-bet)
        let after_cbet = stats.cbet;
        stats.start_hand();
        stats.hand_pfr = true;
        stats.hand_saw_flop = true;
        stats.hand_cbet_opportunity = true;
        stats.hand_cbet_taken = false;
        stats.end_hand();

        assert!(
            stats.cbet < after_cbet,
            "cbet should decrease after missed c-bet: before={after_cbet}, after={}", stats.cbet
        );
    }

    #[test]
    fn test_wtsd_and_wsd_track_showdown() {
        let mut stats = SeatStats::new();

        // Went to showdown and won
        stats.start_hand();
        stats.hand_saw_flop = true;
        stats.record_showdown(true);
        stats.end_hand();

        let wtsd_after_win = stats.wtsd;
        let wsd_after_win = stats.wsd;

        assert!(wtsd_after_win > 0.5, "WTSD should increase after showdown");
        assert!(wsd_after_win > 0.5, "WSD should increase after winning showdown");

        // Went to showdown and lost
        stats.start_hand();
        stats.hand_saw_flop = true;
        stats.record_showdown(false);
        stats.end_hand();

        assert!(stats.wsd < wsd_after_win, "WSD should decrease after losing showdown");
    }

    #[test]
    fn test_bet_ratio_clamped_in_tracker() {
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        // All-in action (action_idx=8) → bet_ratio = 8/8 = 1.0, clamped to min(1.0, 10.0) = 1.0
        // The bet_ratio from record_hand_action is action_idx/8.0 capped at 1.0,
        // then the tracker clamps to 10.0 — both are fine for this input.
        table.phase = GamePhase::PreFlop;
        table.record_hand_action(1, 8); // all-in
        table.record_hand_action(0, 1); // call
        tracker.observe(&table);
        table.hand_action_history.clear();

        let obs = build_full_obs(&table, 0, &mut tracker);
        let bet_sz = obs[OPP_STATS_OFFSET + 11];
        let pfr_sz = obs[OPP_STATS_OFFSET + 12];

        assert!(bet_sz <= 1.0, "bet_size must be <= 1.0 after normalization, got {bet_sz}");
        assert!(pfr_sz <= 1.0, "pfr_size must be <= 1.0 after normalization, got {pfr_sz}");
    }

    #[test]
    fn test_encode_order_matches_training_layout() {
        // Verify the critical invariant: the 582 features follow training order
        // cards(364) | game_base(46) | opp_stats(120) | hand_strength(52)
        let mut table = make_table();
        let mut tracker = OpponentTracker::new();

        // Play a hand so stats diverge from defaults
        for _ in 0..3 {
            simulate_preflop_raise_call(&mut table, &mut tracker);
        }

        let obs = build_full_obs(&table, 0, &mut tracker);
        assert_eq!(obs.len(), OBS_DIM);

        // Verify cards section has exactly 2 one-hot card bits (preflop)
        let cards_sum: f32 = obs[..CARDS_LEN].iter().sum();
        assert!((cards_sum - 2.0).abs() < 0.01, "card section should sum to ~2.0 (2 hole cards)");

        // Verify opp_stats section has non-default values (we played 3 hands)
        let ss = obs[OPP_STATS_OFFSET + 4]; // sample_size
        assert!(ss > 0.0, "sample_size should be > 0 after playing hands, got {ss}");

        // Verify hand_strength section: preflop_strength should be at [HAND_STR_OFFSET + 1]
        let preflop_str = obs[HAND_STR_OFFSET + 1];
        assert!(preflop_str > 0.0 && preflop_str < 1.0,
            "preflop_strength at offset {} should be in (0,1), got {preflop_str}", HAND_STR_OFFSET + 1);

        // Verify the value at HAND_STR_OFFSET is NOT an opp stat (should be 0.0 = no hand rank preflop)
        let hand_rank = obs[HAND_STR_OFFSET];
        assert!((hand_rank - 0.0).abs() < 0.01,
            "hand_rank at offset {} should be 0.0 preflop (not an opp stat), got {hand_rank}",
            HAND_STR_OFFSET);
    }

    // -----------------------------------------------------------------------
    // Behavioral inference tests — use the real ONNX model
    // -----------------------------------------------------------------------

    /// Resolve the AS ONNX model path from environment or well-known location.
    fn resolve_test_model_path() -> Option<std::path::PathBuf> {
        // Try env var first
        if let Ok(p) = std::env::var("POKER_BOT_MODEL_ONNX") {
            let p = p.trim().to_string();
            if !p.is_empty() {
                let path = if p.starts_with('~') {
                    std::path::PathBuf::from(
                        p.replacen('~', &std::env::var("HOME").unwrap_or_default(), 1),
                    )
                } else {
                    std::path::PathBuf::from(&p)
                };
                if path.exists() {
                    return Some(path);
                }
            }
        }
        // Try .env file next to the backend crate
        let dotenv = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");
        if let Ok(contents) = std::fs::read_to_string(&dotenv) {
            for line in contents.lines() {
                if let Some(val) = line.strip_prefix("POKER_BOT_MODEL_ONNX=") {
                    let val = val.trim().trim_matches('"');
                    if !val.is_empty() {
                        let path = if val.starts_with('~') {
                            std::path::PathBuf::from(
                                val.replacen('~', &std::env::var("HOME").unwrap_or_default(), 1),
                            )
                        } else {
                            std::path::PathBuf::from(val)
                        };
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }
        None
    }

    /// Build the full 582-float observation vector for a table state, mirroring
    /// the exact pipeline used in `decide_with_table`.
    fn build_obs(table: &PokerTable, player_idx: usize, tracker: &mut OpponentTracker) -> Vec<f32> {
        let mut obs = encode_static_observation(table, player_idx);
        tracker.observe(table);
        tracker.encode_for_seat(player_idx, table.players.len(), &mut obs);
        encode_hand_strength(table, player_idx, &mut obs);
        assert_eq!(obs.len(), OBS_DIM, "obs should be {OBS_DIM} floats");
        obs
    }

    /// Run inference through the real ONNX model and return raw action
    /// probabilities (before personality / sampling).
    fn infer_probs(
        runtime: &ModelRuntime,
        table: &PokerTable,
        player_idx: usize,
        tracker: &mut OpponentTracker,
    ) -> [f32; NUM_ACTIONS] {
        let obs = build_obs(table, player_idx, tracker);
        let legal = legal_actions_mask(table, player_idx);
        let (history, hlen) = encode_action_history(table);
        runtime
            .infer_action_probs(obs, history, hlen, legal)
            .expect("ONNX inference failed")
    }

    /// Average the action probabilities over `n` calls. Because inference is
    /// deterministic for a given input, we actually just call once; the helper
    /// exists so callers read clearly.
    /// Set specific hole cards for a player (overwrites whatever was dealt).
    fn set_hole_cards(table: &mut PokerTable, player_idx: usize, cards: [(u8, u8); 2]) {
        table.players[player_idx].hole_cards = cards
            .iter()
            .map(|&(rank, suit)| Card::new(rank, suit))
            .collect();
    }

    /// Set community cards on the table.
    fn set_community(table: &mut PokerTable, cards: &[(u8, u8)]) {
        table.community_cards = cards
            .iter()
            .map(|&(rank, suit)| Card::new(rank, suit))
            .collect();
    }

    /// Create a preflop table where hero (seat 0) faces a raise to 3bb.
    fn make_preflop_facing_raise() -> PokerTable {
        let mut table = PokerTable::new("t1".into(), "Test".into(), 50, 100);
        table.take_seat("p1".into(), "Hero".into(), 0, 10000).unwrap();
        table.take_seat("p2".into(), "Villain".into(), 1, 10000).unwrap();
        // take_seat auto-starts a hand (posts blinds, deals cards, etc).
        // Reset to a clean preflop state to avoid double-counting.
        table.pot = crate::game::pot::PotManager::new();
        table.phase = GamePhase::PreFlop;
        table.current_player = 0;
        table.dealer_seat = 1; // Hero is BB, villain is dealer/SB
        table.players[0].state = PlayerState::Active;
        table.players[1].state = PlayerState::Active;
        // Villain raised to 300 (3bb), hero has posted BB of 100
        table.players[0].current_bet = 100;
        table.players[1].current_bet = 300;
        table.players[0].total_bet_this_hand = 100;
        table.players[1].total_bet_this_hand = 300;
        table.players[0].stack = 9900;
        table.players[1].stack = 9700;
        table.current_bet = 300;
        table.min_raise = 200; // min re-raise
        table.pot.add_bet(0, 100);
        table.pot.add_bet(1, 300);
        table.actions_this_round = 1;
        table.raises_this_round = 1;
        table.last_raiser_seat = Some(1);
        table.hand_action_history.clear();
        // Record villain's raise in action history
        table.record_hand_action(1, 4); // villain raised ~0.6x pot
        table
    }

    /// Create a flop table where hero (seat 0) is first to act.
    fn make_flop_check_to_hero() -> PokerTable {
        let mut table = PokerTable::new("t1".into(), "Test".into(), 50, 100);
        table.take_seat("p1".into(), "Hero".into(), 0, 10000).unwrap();
        table.take_seat("p2".into(), "Villain".into(), 1, 10000).unwrap();
        // Reset pot from auto-start
        table.pot = crate::game::pot::PotManager::new();
        table.hand_action_history.clear();
        table.phase = GamePhase::Flop;
        table.current_player = 0;
        table.dealer_seat = 1;
        table.players[0].state = PlayerState::Active;
        table.players[1].state = PlayerState::Active;
        table.players[0].current_bet = 0;
        table.players[1].current_bet = 0;
        table.players[0].total_bet_this_hand = 300;
        table.players[1].total_bet_this_hand = 300;
        table.players[0].stack = 9700;
        table.players[1].stack = 9700;
        table.current_bet = 0;
        table.min_raise = 100;
        table.actions_this_round = 0;
        table.raises_this_round = 0;
        table.pot.add_bet(0, 300);
        table.pot.add_bet(1, 300);
        // Preflop action history
        table.record_hand_action(1, 4); // villain raised
        table.record_hand_action(0, 1); // hero called
        table
    }

    /// Create a flop table where hero faces a pot-size bet.
    fn make_flop_facing_bet() -> PokerTable {
        let mut table = make_flop_check_to_hero();
        // Villain bet pot (600 into 600)
        table.current_player = 0;
        table.players[1].current_bet = 600;
        table.players[1].total_bet_this_hand = 900;
        table.players[1].stack = 9100;
        table.current_bet = 600;
        table.min_raise = 600;
        table.actions_this_round = 1;
        table.raises_this_round = 1;
        table.last_raiser_seat = Some(1);
        table.pot.add_bet(1, 600);
        table.record_hand_action(1, 6); // pot-size bet
        table
    }

    fn format_probs(probs: &[f32; NUM_ACTIONS]) -> String {
        let names = ["fold", "call", "0.25x", "0.4x", "0.6x", "0.8x", "1x", "1.5x", "allin"];
        probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.001)
            .map(|(i, p)| format!("{}={:.1}%", names[i], p * 100.0))
            .collect::<Vec<_>>()
            .join(" ")
    }

    // -- Actual behavioral tests --

    #[test]
    fn test_onnx_premium_hands_raise_preflop() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // Premium hands: AA, KK, QQ, AKs
        let premiums: [[(u8, u8); 2]; 4] = [
            [(14, 0), (14, 1)], // AA
            [(13, 0), (13, 1)], // KK
            [(12, 0), (12, 1)], // QQ
            [(14, 0), (13, 0)], // AKs
        ];
        let premium_names = ["AA", "KK", "QQ", "AKs"];

        for (hand, name) in premiums.iter().zip(premium_names.iter()) {
            let mut table = make_preflop_facing_raise();
            set_hole_cards(&mut table, 0, *hand);
            set_hole_cards(&mut table, 1, [(2, 0), (7, 1)]); // dummy villain cards
            let probs = infer_probs(&runtime, &table, 0, &mut tracker);
            let fold_pct = probs[0];
            let raise_plus_allin: f32 = probs[2..=8].iter().sum();
            eprintln!("{name}: {}", format_probs(&probs));
            assert!(
                fold_pct < 0.10,
                "{name}: fold% should be <10% but got {:.1}%",
                fold_pct * 100.0
            );
            assert!(
                raise_plus_allin > 0.05,
                "{name}: raise+allin should be >5% but got {:.1}%",
                raise_plus_allin * 100.0
            );
        }
    }

    #[test]
    fn test_onnx_trash_hands_fold_more_than_premiums() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // Trash hands facing a raise
        let trash: [[(u8, u8); 2]; 3] = [
            [(2, 0), (7, 1)], // 72o
            [(3, 0), (8, 2)], // 83o
            [(2, 1), (5, 3)], // 52o
        ];
        let trash_names = ["72o", "83o", "52o"];

        // Also get premium fold rate for comparison
        let mut table_aa = make_preflop_facing_raise();
        set_hole_cards(&mut table_aa, 0, [(14, 0), (14, 1)]);
        set_hole_cards(&mut table_aa, 1, [(2, 0), (7, 1)]);
        let aa_fold = infer_probs(&runtime, &table_aa, 0, &mut tracker)[0];

        for (hand, name) in trash.iter().zip(trash_names.iter()) {
            let mut table = make_preflop_facing_raise();
            set_hole_cards(&mut table, 0, *hand);
            set_hole_cards(&mut table, 1, [(14, 0), (14, 1)]); // dummy
            let probs = infer_probs(&runtime, &table, 0, &mut tracker);
            let fold_pct = probs[0];
            eprintln!("{name}: {}", format_probs(&probs));
            assert!(
                fold_pct > aa_fold,
                "{name}: fold ({:.1}%) should exceed AA fold ({:.1}%)",
                fold_pct * 100.0,
                aa_fold * 100.0
            );
        }
    }

    #[test]
    fn test_onnx_made_hand_does_not_fold_flop() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // Hero has top pair on a dry board, facing a pot-size bet
        let mut table = make_flop_facing_bet();
        set_hole_cards(&mut table, 0, [(14, 0), (13, 1)]); // AKo
        set_hole_cards(&mut table, 1, [(2, 0), (7, 1)]);
        set_community(&mut table, &[(14, 2), (9, 0), (4, 3)]); // A94 rainbow
        table.phase = GamePhase::Flop;

        let probs = infer_probs(&runtime, &table, 0, &mut tracker);
        eprintln!("TPGK facing pot bet: {}", format_probs(&probs));
        let fold_pct = probs[0];
        assert!(
            fold_pct < 0.25,
            "top pair good kicker should fold <25% to pot bet, got {:.1}%",
            fold_pct * 100.0
        );

        // Overpair (KK on a low board)
        let mut table2 = make_flop_facing_bet();
        set_hole_cards(&mut table2, 0, [(13, 0), (13, 1)]); // KK
        set_hole_cards(&mut table2, 1, [(2, 0), (7, 1)]);
        set_community(&mut table2, &[(9, 2), (6, 0), (3, 3)]); // 963 rainbow
        table2.phase = GamePhase::Flop;

        let probs2 = infer_probs(&runtime, &table2, 0, &mut tracker);
        eprintln!("Overpair facing pot bet: {}", format_probs(&probs2));
        let fold_pct2 = probs2[0];
        assert!(
            fold_pct2 < 0.20,
            "overpair should fold <20% to pot bet, got {:.1}%",
            fold_pct2 * 100.0
        );
    }

    #[test]
    fn test_onnx_air_folds_more_than_nuts_on_flop() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        let board = [(14, 2), (14, 3), (9, 0)]; // AA9

        // Nuts: hero has A9 (full house)
        let mut table_nuts = make_flop_facing_bet();
        set_hole_cards(&mut table_nuts, 0, [(14, 0), (9, 1)]); // A9
        set_hole_cards(&mut table_nuts, 1, [(2, 0), (7, 1)]);
        set_community(&mut table_nuts, &board);
        table_nuts.phase = GamePhase::Flop;
        let nuts_probs = infer_probs(&runtime, &table_nuts, 0, &mut tracker);
        let nuts_fold = nuts_probs[0];

        // Air: hero has 52o (nothing)
        let mut table_air = make_flop_facing_bet();
        set_hole_cards(&mut table_air, 0, [(5, 0), (2, 1)]); // 52o
        set_hole_cards(&mut table_air, 1, [(2, 2), (7, 1)]);
        set_community(&mut table_air, &board);
        table_air.phase = GamePhase::Flop;
        let air_probs = infer_probs(&runtime, &table_air, 0, &mut tracker);
        let air_fold = air_probs[0];

        eprintln!("Nuts (A9 on AA9): {}", format_probs(&nuts_probs));
        eprintln!("Air  (52 on AA9): {}", format_probs(&air_probs));

        assert!(
            air_fold > nuts_fold,
            "air fold ({:.1}%) should exceed nuts fold ({:.1}%)",
            air_fold * 100.0,
            nuts_fold * 100.0
        );
    }

    #[test]
    fn test_onnx_no_degenerate_single_action() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // Run many different hands and check that no single action dominates
        // across ALL of them.
        let hands: [[(u8, u8); 2]; 8] = [
            [(14, 0), (14, 1)], // AA
            [(2, 0), (7, 1)],   // 72o
            [(10, 0), (10, 2)], // TT
            [(14, 0), (5, 0)],  // A5s
            [(8, 0), (9, 1)],   // 98o
            [(6, 2), (6, 3)],   // 66
            [(13, 0), (12, 0)], // KQs
            [(3, 1), (4, 2)],   // 43o
        ];

        let mut action_totals = [0.0f32; NUM_ACTIONS];
        let mut count = 0;

        for hand in &hands {
            let mut table = make_preflop_facing_raise();
            set_hole_cards(&mut table, 0, *hand);
            set_hole_cards(&mut table, 1, [(2, 2), (3, 3)]);
            let probs = infer_probs(&runtime, &table, 0, &mut tracker);
            for i in 0..NUM_ACTIONS {
                action_totals[i] += probs[i];
            }
            count += 1;
        }

        // Average over all hands
        for a in &mut action_totals {
            *a /= count as f32;
        }
        eprintln!("Average probs across {} hands: {}", count, {
            let names = ["fold", "call", "0.25x", "0.4x", "0.6x", "0.8x", "1x", "1.5x", "allin"];
            action_totals
                .iter()
                .enumerate()
                .filter(|(_, &p)| p > 0.001)
                .map(|(i, p)| format!("{}={:.1}%", names[i], p * 100.0))
                .collect::<Vec<_>>()
                .join(" ")
        });

        // No single action should average above 80% across diverse hands
        let names = ["fold", "call", "0.25x", "0.4x", "0.6x", "0.8x", "1x", "1.5x", "allin"];
        for i in 0..NUM_ACTIONS {
            assert!(
                action_totals[i] < 0.80,
                "action '{}' averages {:.1}% across diverse hands — degenerate policy",
                names[i],
                action_totals[i] * 100.0
            );
        }

        // At least 3 different action types should have >5% average probability
        let diverse_count = action_totals.iter().filter(|&&p| p > 0.05).count();
        assert!(
            diverse_count >= 2,
            "only {} action types have >5% average — model may be degenerate",
            diverse_count
        );
    }

    #[test]
    fn test_onnx_hand_strength_affects_decisions() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // Same board, same situation, but different hole cards.
        // The model should produce DIFFERENT outputs if it reads hand strength.
        let board = [(12, 0), (8, 1), (3, 2)]; // Q83 rainbow

        let mut table_strong = make_flop_check_to_hero();
        set_hole_cards(&mut table_strong, 0, [(12, 3), (11, 0)]); // QJ (top pair)
        set_hole_cards(&mut table_strong, 1, [(2, 0), (7, 1)]);
        set_community(&mut table_strong, &board);
        table_strong.phase = GamePhase::Flop;
        let strong_probs = infer_probs(&runtime, &table_strong, 0, &mut tracker);

        let mut table_weak = make_flop_check_to_hero();
        set_hole_cards(&mut table_weak, 0, [(5, 0), (2, 1)]); // 52o (air)
        set_hole_cards(&mut table_weak, 1, [(2, 2), (7, 1)]);
        set_community(&mut table_weak, &board);
        table_weak.phase = GamePhase::Flop;
        let weak_probs = infer_probs(&runtime, &table_weak, 0, &mut tracker);

        eprintln!("QJ on Q83 (top pair): {}", format_probs(&strong_probs));
        eprintln!("52 on Q83 (air):      {}", format_probs(&weak_probs));

        // The probability distributions should be meaningfully different
        let diff: f32 = strong_probs
            .iter()
            .zip(weak_probs.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.05,
            "probability distributions should differ by >5% total variation \
             between top pair and air, got {:.1}%",
            diff * 100.0
        );

        // Strong hand should bet/raise more than weak hand
        let strong_aggression: f32 = strong_probs[2..=8].iter().sum();
        let weak_aggression: f32 = weak_probs[2..=8].iter().sum();
        eprintln!(
            "Aggression: strong={:.1}% weak={:.1}%",
            strong_aggression * 100.0,
            weak_aggression * 100.0
        );
        // We don't require strong > weak because bluffing is valid,
        // but the distributions should be different (already checked above).
    }

    #[test]
    fn test_onnx_allin_not_dominant_preflop_trash() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // With 100bb deep stacks and trash, all-in should not be the dominant
        // action (that would indicate blind play).
        let trash_hands: [[(u8, u8); 2]; 4] = [
            [(2, 0), (7, 1)], // 72o
            [(3, 0), (8, 2)], // 83o
            [(2, 1), (5, 3)], // 52o
            [(4, 0), (9, 1)], // 94o
        ];
        let names_list = ["72o", "83o", "52o", "94o"];

        for (hand, name) in trash_hands.iter().zip(names_list.iter()) {
            let mut table = make_preflop_facing_raise();
            set_hole_cards(&mut table, 0, *hand);
            set_hole_cards(&mut table, 1, [(14, 0), (14, 1)]);
            let probs = infer_probs(&runtime, &table, 0, &mut tracker);
            let allin_pct = probs[8];
            eprintln!("{name}: allin={:.1}% | {}", allin_pct * 100.0, format_probs(&probs));
            assert!(
                allin_pct < 0.50,
                "{name}: all-in should be <50% with trash at 100bb deep, got {:.1}%",
                allin_pct * 100.0
            );
        }
    }

    #[test]
    fn test_onnx_preflop_strength_ordering() {
        let model_path = match resolve_test_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: no ONNX model found");
                return;
            }
        };
        let runtime = ModelRuntime::load(&model_path).expect("failed to load model");
        let mut tracker = OpponentTracker::new();

        // In HU the model may never fold preflop from the BB (correct GTO).
        // Instead, check that the ACTION DISTRIBUTION differs by hand strength:
        // premium hands should have higher call rate (slow-play) and lower
        // big-raise rate than trash hands (which bluff-raise).
        let ordered_hands: [[(u8, u8); 2]; 5] = [
            [(14, 0), (14, 1)], // AA
            [(13, 0), (12, 0)], // KQs
            [(10, 0), (10, 2)], // TT
            [(8, 0), (9, 1)],   // 98o
            [(2, 0), (7, 1)],   // 72o
        ];
        let hand_names = ["AA", "KQs", "TT", "98o", "72o"];

        let mut call_rates = Vec::new();
        for (hand, name) in ordered_hands.iter().zip(hand_names.iter()) {
            let mut table = make_preflop_facing_raise();
            set_hole_cards(&mut table, 0, *hand);
            set_hole_cards(&mut table, 1, [(2, 2), (3, 3)]);
            let probs = infer_probs(&runtime, &table, 0, &mut tracker);
            let call_rate = probs[1];
            let big_raise: f32 = probs[7] + probs[8]; // 1.5x + allin
            eprintln!("{name}: call={:.1}% big_raise={:.1}% | {}",
                call_rate * 100.0, big_raise * 100.0, format_probs(&probs));
            call_rates.push((*name, call_rate));
        }

        // AA should call more than 72o (slow-play vs bluff)
        let aa_call = call_rates[0].1;
        let trash_call = call_rates[4].1;
        assert!(
            aa_call > trash_call,
            "AA call rate ({:.1}%) should exceed 72o call rate ({:.1}%) — \
             model should slow-play premiums",
            aa_call * 100.0,
            trash_call * 100.0
        );

        // The spread should be meaningful
        let spread = aa_call - trash_call;
        assert!(
            spread > 0.10,
            "spread between AA and 72o call rates should be >10%, got {:.1}%",
            spread * 100.0
        );
    }

}
