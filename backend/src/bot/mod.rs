//! Server-side poker bot module.
//!
//! Bots run as background tasks on the server, making decisions based on
//! hand strength evaluation and configurable strategy. They interact with
//! the game engine directly (no WebSocket needed).

pub mod evaluate;
pub mod strategy;

use crate::game::table::PokerTable;
use crate::game::{GamePhase, PlayerAction};
use std::collections::HashMap;
use strategy::{BotGameView, BotStrategy, SimpleStrategy};
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
    pub fn decide(&self, view: &BotGameView) -> PlayerAction {
        self.strategy.decide(view)
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

        let strategy: Box<dyn BotStrategy> = match strategy_name {
            Some("tight") => Box::new(SimpleStrategy::tight()),
            Some("aggressive") => Box::new(SimpleStrategy::aggressive()),
            Some("calling_station") => Box::new(strategy::CallingStation),
            _ => Box::new(SimpleStrategy::balanced()),
        };

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
        let strategy: Box<dyn BotStrategy> = match strategy_name {
            Some("tight") => Box::new(SimpleStrategy::tight()),
            Some("aggressive") => Box::new(SimpleStrategy::aggressive()),
            Some("calling_station") => Box::new(strategy::CallingStation),
            _ => Box::new(SimpleStrategy::balanced()),
        };

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
            self.bots.entry(to_table_id.to_string()).or_default().push(bot);
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

        for (table_id, table) in tables {
            // Skip tables not in an active betting phase
            if !matches!(
                table.phase,
                GamePhase::PreFlop | GamePhase::Flop | GamePhase::Turn | GamePhase::River
            ) {
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
                let action = bot.decide(&view);
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
            }
        }

        actions
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
