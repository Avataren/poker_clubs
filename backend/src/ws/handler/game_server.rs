use crate::{
    auth::JwtManager,
    bot::BotManager,
    game::{
        constants::BROADCAST_CHANNEL_CAPACITY, constants::DEFAULT_MAX_SEATS, GameFormat,
        PokerTable, PokerVariant,
    },
    ws::messages::ServerMessage,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{broadcast, RwLock};

use super::{ClubBroadcast, GlobalBroadcast, TableBroadcast};

/// Grace period before a disconnected player is fully removed from their table.
const DISCONNECT_GRACE_PERIOD_SECS: u64 = 60;

/// Tracks a player who disconnected but may reconnect within the grace period.
#[derive(Debug, Clone)]
pub(super) struct DisconnectedPlayer {
    pub table_id: String,
    pub disconnected_at: tokio::time::Instant,
}

pub struct GameServer {
    pub(super) tables: Arc<RwLock<HashMap<String, PokerTable>>>,
    pub(super) jwt_manager: Arc<JwtManager>,
    pub(super) pool: Arc<sqlx::SqlitePool>,
    // Broadcast channel for each table - sends notification when table state changes
    pub(super) table_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<TableBroadcast>>>>,
    // Broadcast channel for each club - sends notification when tables/members change
    pub(super) club_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<ClubBroadcast>>>>,
    // Global broadcast - for new clubs, global events
    pub(super) global_broadcast: broadcast::Sender<GlobalBroadcast>,
    // Tournament broadcasts scoped by club
    pub(super) tournament_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<ServerMessage>>>>,
    // Bot manager - tracks server-side bot players
    pub(super) bot_manager: Arc<RwLock<BotManager>>,
    // Track disconnected players awaiting reconnection
    pub(super) disconnected_players: Arc<RwLock<HashMap<String, DisconnectedPlayer>>>,
}

impl GameServer {
    pub fn new(jwt_manager: Arc<JwtManager>, pool: Arc<sqlx::SqlitePool>) -> Self {
        let (global_tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            jwt_manager,
            pool,
            table_broadcasts: Arc::new(RwLock::new(HashMap::new())),
            club_broadcasts: Arc::new(RwLock::new(HashMap::new())),
            global_broadcast: global_tx,
            tournament_broadcasts: Arc::new(RwLock::new(HashMap::new())),
            bot_manager: Arc::new(RwLock::new(BotManager::new())),
            disconnected_players: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[allow(dead_code)] // Prepared for global event subscriptions
    pub fn subscribe_global(&self) -> broadcast::Receiver<GlobalBroadcast> {
        self.global_broadcast.subscribe()
    }

    pub async fn subscribe_tournament_club(
        &self,
        club_id: &str,
    ) -> broadcast::Receiver<ServerMessage> {
        let mut broadcasts = self.tournament_broadcasts.write().await;
        let tx = broadcasts.entry(club_id.to_string()).or_insert_with(|| {
            let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
            tx
        });
        tx.subscribe()
    }

    #[allow(dead_code)] // Prepared for club-level event subscriptions
    pub async fn subscribe_club(&self, club_id: &str) -> broadcast::Receiver<ClubBroadcast> {
        let mut broadcasts = self.club_broadcasts.write().await;
        let tx = broadcasts.entry(club_id.to_string()).or_insert_with(|| {
            let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
            tx
        });
        tx.subscribe()
    }

    pub async fn notify_global(&self, event_type: &str) {
        let _ = self.global_broadcast.send(GlobalBroadcast {
            event_type: event_type.to_string(),
        });
    }

    pub async fn notify_club(&self, club_id: &str) {
        let broadcasts = self.club_broadcasts.read().await;
        if let Some(tx) = broadcasts.get(club_id) {
            let _ = tx.send(ClubBroadcast {
                club_id: club_id.to_string(),
            });
        }
    }

    #[allow(dead_code)] // Keep for backwards compatibility, new code uses create_table_with_options
    pub async fn create_table(
        &self,
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
    ) {
        let table = PokerTable::new(table_id.clone(), name, small_blind, big_blind);
        self.tables.write().await.insert(table_id.clone(), table);

        // Create broadcast channel for this table
        let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
        self.table_broadcasts.write().await.insert(table_id, tx);
    }

    /// Create a table with a specific variant and format
    pub async fn create_table_with_options(
        &self,
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        variant: Box<dyn PokerVariant>,
        format: Box<dyn GameFormat>,
    ) {
        let table = PokerTable::with_variant_and_format(
            table_id.clone(),
            name,
            small_blind,
            big_blind,
            DEFAULT_MAX_SEATS,
            variant,
            format,
        );
        self.tables.write().await.insert(table_id.clone(), table);

        // Create broadcast channel for this table
        let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
        self.table_broadcasts.write().await.insert(table_id, tx);
    }

    #[allow(dead_code)] // Useful for REST API to get table state
    pub async fn get_table_state(
        &self,
        table_id: &str,
        user_id: Option<&str>,
    ) -> Option<crate::game::PublicTableState> {
        let tables = self.tables.read().await;
        tables.get(table_id).map(|t| t.get_public_state(user_id))
    }

    // Check all tables for auto-advance (when all players are all-in)
    pub async fn check_all_tables_auto_advance(&self) {
        let mut tables = self.tables.write().await;
        let mut tables_to_notify = Vec::new();

        for (table_id, table) in tables.iter_mut() {
            if table.check_auto_advance() {
                tables_to_notify.push(table_id.clone());
            }
        }

        drop(tables);

        // Notify after releasing the lock
        for table_id in tables_to_notify {
            self.notify_table_update(&table_id).await;
        }
    }

    pub(super) async fn get_or_create_broadcast(&self, table_id: &str) -> broadcast::Receiver<TableBroadcast> {
        let mut broadcasts = self.table_broadcasts.write().await;
        let tx = broadcasts.entry(table_id.to_string()).or_insert_with(|| {
            let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
            tx
        });
        tx.subscribe()
    }

    pub(super) async fn notify_table_update(&self, table_id: &str) {
        let broadcasts = self.table_broadcasts.read().await;
        if let Some(tx) = broadcasts.get(table_id) {
            let _ = tx.send(TableBroadcast {
                table_id: table_id.to_string(),
            });
        }
    }

    /// Fetch tournament info for a table to include in state
    pub(super) async fn get_tournament_info(
        &self,
        tournament_id: &str,
    ) -> Option<crate::game::TournamentInfo> {
        use crate::db::models::{Tournament, TournamentBlindLevel};

        // Get tournament
        let tournament: Option<Tournament> =
            sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
                .bind(tournament_id)
                .fetch_optional(self.pool.as_ref())
                .await
                .map_err(|e| tracing::warn!("Failed to fetch tournament {}: {}", tournament_id, e))
                .ok()
                .flatten();

        let tournament = tournament?;

        // Get current blind level (for ante)
        let current_level: Option<TournamentBlindLevel> = sqlx::query_as(
            "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? AND level_number = ?",
        )
        .bind(tournament_id)
        .bind(tournament.current_blind_level)
        .fetch_optional(self.pool.as_ref())
        .await
        .map_err(|e| tracing::warn!("Failed to fetch blind level for tournament {}: {}", tournament_id, e))
        .ok()
        .flatten();

        // Get next blind level
        let next_level: Option<TournamentBlindLevel> = sqlx::query_as(
            "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? AND level_number = ?",
        )
        .bind(tournament_id)
        .bind(tournament.current_blind_level + 1)
        .fetch_optional(self.pool.as_ref())
        .await
        .map_err(|e| tracing::warn!("Failed to fetch next blind level for tournament {}: {}", tournament_id, e))
        .ok()
        .flatten();

        Some(crate::game::TournamentInfo {
            tournament_id: tournament_id.to_string(),
            blind_level: tournament.current_blind_level as i64,
            ante: current_level.as_ref().map(|l| l.ante).unwrap_or(0),
            level_start_time: tournament.level_start_time,
            level_duration_secs: tournament.level_duration_secs,
            next_small_blind: next_level.as_ref().map(|l| l.small_blind),
            next_big_blind: next_level.as_ref().map(|l| l.big_blind),
        })
    }

    // Load table from database into memory
    pub(super) async fn load_table_from_db(&self, table_id: &str) -> Result<(), String> {
        // Query database for table with variant and format
        let table_data: Option<(String, String, String, i64, i64, String, String)> = sqlx::query_as(
            "SELECT id, club_id, name, small_blind, big_blind, variant_id, format_id FROM tables WHERE id = ?",
        )
        .bind(table_id)
        .fetch_optional(self.pool.as_ref())
        .await
        .map_err(|e| format!("Database error: {}", e))?;

        if let Some((id, _club_id, name, small_blind, big_blind, variant_id, format_id)) =
            table_data
        {
            // Get the variant from ID (default to holdem if not found)
            let variant = crate::game::variant_from_id(&variant_id)
                .ok_or_else(|| format!("Unknown variant: {}", variant_id))?;

            // Check if this is a tournament table
            let is_tournament: Option<(String, i64, i64)> = sqlx::query_as(
                "SELECT t.id, t.buy_in, t.starting_stack
                 FROM tournaments t
                 JOIN tournament_tables tt ON t.id = tt.tournament_id
                 WHERE tt.table_id = ?",
            )
            .bind(table_id)
            .fetch_optional(self.pool.as_ref())
            .await
            .map_err(|e| format!("Database error: {}", e))?;

            if let Some((tournament_id, buy_in, starting_stack)) = is_tournament {
                // Load tournament to get level duration
                let tournament: crate::db::models::Tournament =
                    sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
                        .bind(&tournament_id)
                        .fetch_one(self.pool.as_ref())
                        .await
                        .map_err(|e| format!("Failed to load tournament: {}", e))?;

                // Create tournament format
                let format: Box<dyn crate::game::GameFormat> = if format_id == "sng" {
                    Box::new(crate::game::SitAndGo::new(
                        buy_in,
                        starting_stack,
                        tournament.max_players as usize,
                        tournament.level_duration_secs as u64,
                    ))
                } else {
                    Box::new(crate::game::MultiTableTournament::new(
                        name.clone(),
                        buy_in,
                        starting_stack,
                        tournament.level_duration_secs as u64,
                    ))
                };

                // Create in-memory table with tournament format
                let table = crate::game::PokerTable::with_variant_and_format(
                    id.clone(),
                    name,
                    small_blind,
                    big_blind,
                    9, // DEFAULT_MAX_SEATS
                    variant,
                    format,
                );

                // Set tournament ID on the table
                let mut tables = self.tables.write().await;
                tables.insert(id.clone(), table);
                drop(tables);

                self.set_table_tournament(&id, tournament_id).await;
            } else {
                // Create regular cash game table
                let table = crate::game::PokerTable::with_variant(
                    id.clone(),
                    name,
                    small_blind,
                    big_blind,
                    9, // DEFAULT_MAX_SEATS
                    variant,
                );
                self.tables.write().await.insert(id.clone(), table);
            }

            // Create broadcast channel for this table
            let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
            self.table_broadcasts.write().await.insert(id, tx);

            Ok(())
        } else {
            Err("Table not in the database.".to_string())
        }
    }

    /// Record that a player has disconnected from a table.
    pub async fn mark_disconnected(&self, user_id: &str, table_id: &str) {
        let mut disconnected = self.disconnected_players.write().await;
        disconnected.insert(
            user_id.to_string(),
            DisconnectedPlayer {
                table_id: table_id.to_string(),
                disconnected_at: tokio::time::Instant::now(),
            },
        );
        tracing::info!(
            "Player {} marked as disconnected from table {} (grace period: {}s)",
            user_id,
            table_id,
            DISCONNECT_GRACE_PERIOD_SECS
        );
    }

    /// Clear disconnected status when a player reconnects.
    pub async fn clear_disconnected(&self, user_id: &str) {
        let mut disconnected = self.disconnected_players.write().await;
        if disconnected.remove(user_id).is_some() {
            tracing::info!("Player {} reconnected, cleared disconnected status", user_id);
        }
    }

    /// Check if a player is currently in disconnected state.
    #[allow(dead_code)] // Prepared for reconnection logic queries
    pub(super) async fn is_disconnected(&self, user_id: &str) -> Option<DisconnectedPlayer> {
        let disconnected = self.disconnected_players.read().await;
        disconnected.get(user_id).cloned()
    }

    /// Remove players whose grace period has expired. Called periodically.
    pub async fn cleanup_expired_disconnections(&self) {
        let grace = tokio::time::Duration::from_secs(DISCONNECT_GRACE_PERIOD_SECS);
        let now = tokio::time::Instant::now();

        // Collect expired entries
        let expired: Vec<(String, String)> = {
            let disconnected = self.disconnected_players.read().await;
            disconnected
                .iter()
                .filter(|(_, info)| now.duration_since(info.disconnected_at) > grace)
                .map(|(uid, info)| (uid.clone(), info.table_id.clone()))
                .collect()
        };

        if expired.is_empty() {
            return;
        }

        // Remove from tracking map
        {
            let mut disconnected = self.disconnected_players.write().await;
            for (uid, _) in &expired {
                disconnected.remove(uid);
            }
        }

        // Remove each expired player from their table
        for (uid, table_id) in &expired {
            tracing::info!(
                "Disconnect grace period expired for player {} on table {}, removing from table",
                uid,
                table_id
            );
            let mut tables = self.tables.write().await;
            if let Some(table) = tables.get_mut(table_id) {
                table.remove_player(uid);
            }
            drop(tables);
            self.notify_table_update(table_id).await;
        }
    }
}
