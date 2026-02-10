//! Server-side poker bot module.
//!
//! Bots run as background tasks on the server, making decisions based on
//! hand strength evaluation and configurable strategy. They interact with
//! the game engine directly (no WebSocket needed).

pub mod evaluate;
pub mod model;
pub mod strategy;

use crate::game::constants::BOT_ACTION_THINK_DELAY_MS;
use crate::game::table::{current_timestamp_ms, is_bot_identity, PokerTable};
use crate::game::{GamePhase, PlayerAction};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use strategy::{BotGameView, BotPosition, BotStrategy, SimpleStrategy};
use uuid::Uuid;

/// A bot player with its decision-making strategy.
pub struct BotPlayer {
    pub user_id: String,
    pub username: String,
    strategy: Box<dyn BotStrategy>,
}

impl BotPlayer {
    pub fn new(user_id: String, username: String, strategy: Box<dyn BotStrategy>) -> Self {
        Self {
            user_id,
            username,
            strategy,
        }
    }

    /// Decide what action to take given the current table state.
    pub fn decide(
        &self,
        view: &BotGameView,
        table: &PokerTable,
        player_idx: usize,
    ) -> PlayerAction {
        self.strategy.decide_with_table(view, table, player_idx)
    }
}

/// Manages bot players across tables.
pub struct BotManager {
    /// Maps table_id -> list of bot players at that table
    bots: HashMap<String, Vec<BotPlayer>>,
    /// Counter for generating unique bot names
    next_bot_name_idx: u32,
}

/// Bot names to cycle through.
const BOT_NAMES: &[&str] = &[
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Karen",
    "Leo",
];

impl BotManager {
    pub fn new() -> Self {
        Self {
            bots: HashMap::new(),
            next_bot_name_idx: 1,
        }
    }

    /// Add a bot to a table. Returns the bot's user_id and username.
    pub fn add_bot(
        &mut self,
        table_id: &str,
        name: Option<String>,
        strategy_name: Option<&str>,
    ) -> (String, String) {
        let idx = self.next_bot_name_idx;
        self.next_bot_name_idx += 1;

        // Use UUID to prevent collision on restart
        let user_id = format!("bot_{}", Uuid::new_v4());
        let username = name.unwrap_or_else(|| {
            let base = BOT_NAMES[(idx as usize - 1) % BOT_NAMES.len()];
            format!("{} (Bot)", base)
        });

        let strategy = build_strategy(strategy_name);

        let bot = BotPlayer::new(user_id.clone(), username.clone(), strategy);
        self.bots.entry(table_id.to_string()).or_default().push(bot);

        (user_id, username)
    }

    /// Register an existing user as a bot (for tournament bots created in DB)
    pub fn register_existing_bot(
        &mut self,
        table_id: &str,
        user_id: String,
        username: String,
        strategy_name: Option<&str>,
    ) {
        let strategy = build_strategy(strategy_name);

        let bot = BotPlayer::new(user_id, username, strategy);
        self.bots.entry(table_id.to_string()).or_default().push(bot);
    }

    /// Remove a bot from a table.
    pub fn remove_bot(&mut self, table_id: &str, user_id: &str) -> bool {
        if let Some(bots) = self.bots.get_mut(table_id) {
            let before = bots.len();
            bots.retain(|b| b.user_id != user_id);
            bots.len() < before
        } else {
            false
        }
    }

    /// Move a bot from one table to another. Returns true if the bot was found and moved.
    pub fn move_bot(&mut self, from_table_id: &str, to_table_id: &str, user_id: &str) -> bool {
        let bot = if let Some(bots) = self.bots.get_mut(from_table_id) {
            if let Some(pos) = bots.iter().position(|b| b.user_id == user_id) {
                Some(bots.remove(pos))
            } else {
                None
            }
        } else {
            None
        };

        if let Some(bot) = bot {
            self.bots
                .entry(to_table_id.to_string())
                .or_default()
                .push(bot);
            true
        } else {
            false
        }
    }

    /// Check if a user_id belongs to a bot.
    pub fn is_bot(&self, user_id: &str) -> bool {
        self.bots
            .values()
            .any(|bots| bots.iter().any(|b| b.user_id == user_id))
    }

    /// Get a bot by user_id (across all tables).
    pub fn get_bot(&self, user_id: &str) -> Option<&BotPlayer> {
        self.bots
            .values()
            .flat_map(|bots| bots.iter())
            .find(|b| b.user_id == user_id)
    }

    /// Collect actions for all bots whose turn it is.
    /// Returns (table_id, bot_user_id, action) tuples.
    pub fn collect_bot_actions(
        &self,
        tables: &HashMap<String, PokerTable>,
    ) -> Vec<(String, String, PlayerAction)> {
        let mut actions = Vec::new();
        let now = current_timestamp_ms();

        for (table_id, table) in tables {
            // Skip tables not in an active betting phase
            if !matches!(
                table.phase,
                GamePhase::PreFlop | GamePhase::Flop | GamePhase::Turn | GamePhase::River
            ) {
                continue;
            }

            // Give clients a short window to render phase transitions before
            // the next bot action mutates state again.
            if let Some(phase_changed_at) = table.last_phase_change_time {
                if now.saturating_sub(phase_changed_at) < BOT_ACTION_THINK_DELAY_MS {
                    continue;
                }
            }

            // If a round is complete, the table is waiting on timed auto-advance.
            // Do not queue a new bot action in that transition window.
            if table.is_betting_round_complete() {
                continue;
            }

            // Check if current player is a bot
            if table.current_player >= table.players.len() {
                continue;
            }

            let current = &table.players[table.current_player];
            if !current.can_act() {
                continue;
            }

            if let Some(bot) = self.get_bot(&current.user_id) {
                let view = build_bot_view(table, table.current_player);
                let action = bot.decide(&view, table, table.current_player);
                tracing::info!(
                    "Bot {} at table {} decides: {:?} (phase={:?}, pot={}, bet={})",
                    bot.username,
                    table_id,
                    action,
                    table.phase,
                    view.pot_total,
                    view.current_bet
                );
                actions.push((table_id.clone(), current.user_id.clone(), action));
            } else if is_bot_identity(&current.user_id, &current.username) {
                // Tournament/table balancing can transiently desync bot-manager
                // registration. Fall back to a balanced strategy so the table
                // cannot deadlock waiting for a bot turn.
                let view = build_bot_view(table, table.current_player);
                let fallback = SimpleStrategy::balanced();
                let action = fallback.decide(&view);
                tracing::warn!(
                    "Bot {} ({}) missing from bot manager at table {}; using fallback action {:?}",
                    current.username,
                    current.user_id,
                    table_id,
                    action
                );
                actions.push((table_id.clone(), current.user_id.clone(), action));
            }
        }

        actions
    }
}

fn build_strategy(strategy_name: Option<&str>) -> Box<dyn BotStrategy> {
    match strategy_name {
        Some("tight") => Box::new(SimpleStrategy::tight()),
        Some("aggressive") => Box::new(SimpleStrategy::aggressive()),
        Some("calling_station") => Box::new(strategy::CallingStation),
        Some("model") => match env::var("POKER_BOT_MODEL_ONNX") {
            Ok(path) if !path.trim().is_empty() => load_model_strategy(&path),
            _ => {
                tracing::warn!(
                    "Bot strategy `model` requested but POKER_BOT_MODEL_ONNX is not set; falling back to balanced strategy"
                );
                Box::new(SimpleStrategy::balanced())
            }
        },
        Some(name) if name.starts_with("model:") => {
            let path = name.trim_start_matches("model:").trim();
            if path.is_empty() {
                tracing::warn!(
                    "Bot strategy `model:` requested with empty path; falling back to balanced strategy"
                );
                Box::new(SimpleStrategy::balanced())
            } else {
                load_model_strategy(path)
            }
        }
        _ => Box::new(SimpleStrategy::balanced()),
    }
}

fn load_model_strategy(path: &str) -> Box<dyn BotStrategy> {
    match model::ModelStrategy::from_path(Path::new(path)) {
        Ok(strategy) => {
            tracing::info!("Loaded ONNX model bot strategy from {}", path);
            Box::new(strategy)
        }
        Err(err) => {
            tracing::warn!(
                "Failed to load ONNX model bot strategy from {}: {}. Falling back to balanced strategy.",
                path,
                err
            );
            Box::new(SimpleStrategy::balanced())
        }
    }
}

impl Default for BotManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a BotGameView from a table and the bot's player index.
fn build_bot_view(table: &PokerTable, player_idx: usize) -> BotGameView {
    let player = &table.players[player_idx];
    let num_active_opponents = table
        .players
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != player_idx && p.is_active_in_hand())
        .count();

    let position = compute_position(table, player_idx);
    let is_big_blind = if table.players.len() == 2 {
        player_idx == (table.dealer_seat + 1) % 2
    } else {
        let sb_idx = table.next_player_for_blind(table.dealer_seat);
        let bb_idx = table.next_player_for_blind(sb_idx);
        player_idx == bb_idx
    };

    // Heuristic: if my current bet > big blind, I was likely the preflop raiser
    let was_preflop_raiser = player.current_bet > table.big_blind;

    BotGameView {
        hole_cards: player.hole_cards.clone(),
        community_cards: table.community_cards.clone(),
        pot_total: table.pot.total(),
        current_bet: table.current_bet,
        my_current_bet: player.current_bet,
        my_stack: player.stack,
        phase: table.phase.clone(),
        big_blind: table.big_blind,
        min_raise: table.min_raise,
        num_active_opponents,
        position,
        was_preflop_raiser,
        is_big_blind,
    }
}

/// Compute the bot's position category from dealer_seat and player arrangement.
fn compute_position(table: &PokerTable, player_idx: usize) -> BotPosition {
    let num_players = table.players.len();
    if num_players <= 2 {
        // Heads-up: dealer is Late (button), other is Blind
        return if player_idx == table.dealer_seat {
            BotPosition::Late
        } else {
            BotPosition::Blind
        };
    }

    // Find SB and BB indices
    let sb_idx = table.next_player_for_blind(table.dealer_seat);
    let bb_idx = table.next_player_for_blind(sb_idx);

    if player_idx == sb_idx || player_idx == bb_idx {
        return BotPosition::Blind;
    }

    if player_idx == table.dealer_seat {
        return BotPosition::Late;
    }

    // Build the preflop acting order: UTG (after BB) ... to dealer (button)
    // and find where we sit in that order
    let mut positions = Vec::new();
    let mut idx = bb_idx;
    loop {
        idx = (idx + 1) % num_players;
        if idx == sb_idx {
            break; // full circle
        }
        positions.push(idx);
        if idx == table.dealer_seat {
            break;
        }
    }

    let my_pos = positions.iter().position(|&i| i == player_idx);
    let total = positions.len();

    match (my_pos, total) {
        (Some(pos), t) if t <= 3 => {
            // Short-handed (3-4 players): just Early and Late
            if pos + 1 == t {
                BotPosition::Late
            } else {
                BotPosition::Early
            }
        }
        (Some(pos), t) => {
            // Standard table: first ~1/3 Early, last ~1/3 Late, rest Middle
            let early_cutoff = t.div_ceil(3);
            let late_start = t - t.div_ceil(3);
            if pos < early_cutoff {
                BotPosition::Early
            } else if pos >= late_start {
                BotPosition::Late
            } else {
                BotPosition::Middle
            }
        }
        _ => BotPosition::Middle, // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::constants::BOT_ACTION_THINK_DELAY_MS;

    #[test]
    fn test_bot_manager_add_remove() {
        let mut mgr = BotManager::new();

        let (id1, name1) = mgr.add_bot("table_1", None, None);
        assert!(id1.starts_with("bot_"));
        assert!(name1.contains("Bot"));
        assert!(mgr.is_bot(&id1));

        let (id2, _) = mgr.add_bot(
            "table_1",
            Some("Custom Bot".to_string()),
            Some("aggressive"),
        );
        assert!(id2.starts_with("bot_"));
        assert!(mgr.is_bot(&id2));

        assert!(mgr.remove_bot("table_1", &id1));
        assert!(!mgr.is_bot(&id1));
        assert!(mgr.is_bot(&id2));
    }

    #[test]
    fn test_bot_manager_unique_ids() {
        let mut mgr = BotManager::new();
        let (id1, _) = mgr.add_bot("t1", None, None);
        let (id2, _) = mgr.add_bot("t1", None, None);
        let (id3, _) = mgr.add_bot("t2", None, None);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
    }

    #[test]
    fn test_collect_bot_actions_skips_completed_round() {
        let mut mgr = BotManager::new();
        let table_id = "table_1".to_string();
        let (bot_id, bot_name) = mgr.add_bot(&table_id, None, None);

        let mut table = PokerTable::new(table_id.clone(), "Test Table".to_string(), 50, 100);
        table
            .take_seat("human".to_string(), "Human".to_string(), 0, 5000)
            .unwrap();
        table.take_seat(bot_id.clone(), bot_name, 1, 5000).unwrap();

        let human_idx = table
            .players
            .iter()
            .position(|p| p.user_id == "human")
            .expect("human should exist");
        let bot_idx = table
            .players
            .iter()
            .position(|p| p.user_id == bot_id)
            .expect("bot should exist");

        table.current_player = bot_idx;
        table.current_bet = 100;
        table.players[human_idx].current_bet = 100;
        table.players[human_idx].has_acted_this_round = true;
        table.players[bot_idx].current_bet = 100;
        table.players[bot_idx].has_acted_this_round = true;
        table.last_phase_change_time =
            Some(current_timestamp_ms().saturating_sub(BOT_ACTION_THINK_DELAY_MS + 1));
        assert!(table.is_betting_round_complete());

        let mut tables = HashMap::new();
        tables.insert(table_id, table);
        let actions = mgr.collect_bot_actions(&tables);
        assert!(
            actions.is_empty(),
            "bot manager should not queue actions while waiting for street auto-advance"
        );
    }

    #[test]
    fn test_collect_bot_actions_uses_identity_fallback_when_unregistered() {
        let mgr = BotManager::new();
        let table_id = "table_fallback".to_string();

        let mut table = PokerTable::new(table_id.clone(), "Fallback Table".to_string(), 50, 100);
        table
            .take_seat(
                "bot_like_user".to_string(),
                "Alice (Bot)".to_string(),
                0,
                5000,
            )
            .unwrap();
        table
            .take_seat("human".to_string(), "Human".to_string(), 1, 5000)
            .unwrap();

        let bot_idx = table
            .players
            .iter()
            .position(|p| p.user_id == "bot_like_user")
            .expect("bot-like player should exist");
        table.current_player = bot_idx;
        table.last_phase_change_time =
            Some(current_timestamp_ms().saturating_sub(BOT_ACTION_THINK_DELAY_MS + 1));

        let mut tables = HashMap::new();
        tables.insert(table_id.clone(), table);

        let actions = mgr.collect_bot_actions(&tables);
        assert_eq!(actions.len(), 1, "fallback should queue a bot action");
        assert_eq!(actions[0].0, table_id);
        assert_eq!(actions[0].1, "bot_like_user");
    }
}
