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

const OBS_DIM: usize = 462;
const HISTORY_DIM: usize = 11;
const MAX_HISTORY_LEN: usize = 30;
const NUM_ACTIONS: usize = 9;

type OnnxPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>;

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
            .set_input_fact(3, InferenceFact::dt_shape(bool::datum_type(), &[1, NUM_ACTIONS]))
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
        let legal_tensor =
            tract_ndarray::Array2::from_shape_vec((1, NUM_ACTIONS), legal_mask.to_vec())
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

        let obs = encode_static_observation(table, player_idx);
        
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
    encode_hand_strength(table, player_idx, &mut obs);
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

    // Street action count / 10 (approximate from hand history)
    let street_actions = table.hand_action_history.len().min(30) as f32;
    out.push((street_actions / 10.0).min(1.0));

    // Total action count / 30
    out.push((table.hand_action_history.len() as f32 / 30.0).min(1.0));

    // Num raises this street / 4 (approximate: count raises in history)
    let raises_count = table.raises_this_round.min(10) as f32;
    out.push((raises_count / 4.0).min(1.0));

    // Last aggressor is hero
    // (approximation: check if last raiser seat matches player_idx)
    out.push(0.0); // conservative default since we don't track last raiser seat precisely

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
    // Total: 6 + 8 + 8 + 24 = 46
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
        assert_eq!(obs.len(), OBS_DIM);
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
}
