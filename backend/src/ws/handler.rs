use crate::{
    auth::JwtManager,
    bot::BotManager,
    game::{
        constants::BROADCAST_CHANNEL_CAPACITY, constants::DEFAULT_MAX_SEATS, GameFormat,
        PokerTable, PokerVariant,
    },
    ws::messages::{ClientMessage, ServerMessage},
};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{broadcast, RwLock};

// Broadcast message types - prepared for pub/sub broadcasting
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TableBroadcast {
    table_id: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ClubBroadcast {
    club_id: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GlobalBroadcast {
    event_type: String, // "club_created", "club_joined", etc.
}

#[derive(Deserialize)]
pub struct WsQuery {
    token: String,
}

pub struct GameServer {
    tables: Arc<RwLock<HashMap<String, PokerTable>>>,
    jwt_manager: Arc<JwtManager>,
    pool: Arc<sqlx::SqlitePool>,
    // Broadcast channel for each table - sends notification when table state changes
    table_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<TableBroadcast>>>>,
    // Broadcast channel for each club - sends notification when tables/members change
    club_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<ClubBroadcast>>>>,
    // Global broadcast - for new clubs, global events
    global_broadcast: broadcast::Sender<GlobalBroadcast>,
    // Tournament broadcasts scoped by club
    tournament_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<ServerMessage>>>>,
    // Bot manager - tracks server-side bot players
    bot_manager: Arc<RwLock<BotManager>>,
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

    /// Add a player to a table at a specific seat (for tournaments)
    pub async fn add_player_to_table(
        &self,
        table_id: &str,
        user_id: String,
        username: String,
        seat: usize,
        stack: i64,
    ) -> Result<(), crate::game::error::GameError> {
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        table.take_seat(user_id, username, seat, stack)?;

        Ok(())
    }

    /// Add a player to the next available seat (for late registration or rebuys)
    pub async fn add_player_to_table_next_seat(
        &self,
        table_id: &str,
        user_id: &str,
        username: &str,
        stack: i64,
    ) -> Result<usize, crate::game::error::GameError> {
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        if table.players.len() >= table.max_seats {
            return Err(crate::game::error::GameError::TableFull);
        }

        let mut occupied = vec![false; table.max_seats];
        for player in &table.players {
            if player.seat < table.max_seats {
                occupied[player.seat] = true;
            }
        }

        let seat = occupied
            .iter()
            .position(|taken| !*taken)
            .ok_or(crate::game::error::GameError::TableFull)?;

        table.take_seat(user_id.to_string(), username.to_string(), seat, stack)?;

        drop(tables);
        self.notify_table_update(table_id).await;

        Ok(seat)
    }

    /// Apply a tournament add-on or rebuy stack to a seated player
    pub async fn tournament_top_up(
        &self,
        table_id: &str,
        user_id: &str,
        amount: i64,
    ) -> Result<(), crate::game::error::GameError> {
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        table.top_up(user_id, amount)?;

        drop(tables);
        self.notify_table_update(table_id).await;

        Ok(())
    }

    /// Update blinds on a table (for tournaments)
    pub async fn update_table_blinds(&self, table_id: &str, small_blind: i64, big_blind: i64) {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            table.update_blinds(small_blind, big_blind);
            tracing::info!(
                "Updated blinds on table {} to {}/{}",
                table_id,
                small_blind,
                big_blind
            );
        }
    }

    /// Set tournament ID for a table
    pub async fn set_table_tournament(&self, table_id: &str, tournament_id: String) {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            table.set_tournament_id(Some(tournament_id));
        }
    }

    /// Force start a hand on a table (for tournaments after all players are seated)
    pub async fn force_start_table_hand(&self, table_id: &str) {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            table.force_start_hand();
            tracing::info!("Force-started first hand on table {}", table_id);
        }
        drop(tables);
        self.notify_table_update(table_id).await;
    }

    /// Check a table for eliminations and return (tournament_id, eliminated_users)
    pub async fn check_table_eliminations(&self, table_id: &str) -> Option<(String, Vec<String>)> {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            let eliminated = table.check_eliminations();
            if !eliminated.is_empty() {
                if let Some(tournament_id) = &table.tournament_id {
                    return Some((tournament_id.clone(), eliminated));
                }
            }
        }
        None
    }

    /// Broadcast a tournament event to all tables in a tournament
    pub async fn broadcast_tournament_event(&self, tournament_id: &str, message: ServerMessage) {
        // Get tournament to find its club_id
        let tournament_result: Result<(String,), sqlx::Error> =
            sqlx::query_as("SELECT club_id FROM tournaments WHERE id = ?")
                .bind(tournament_id)
                .fetch_one(self.pool.as_ref())
                .await;

        if let Ok((club_id,)) = tournament_result {
            // Send via tournament broadcast channel (scoped by club)
            let broadcasts = self.tournament_broadcasts.read().await;
            if let Some(tx) = broadcasts.get(&club_id) {
                let _ = tx.send(message.clone());
            }
            drop(broadcasts);
        }

        // Also find and trigger table updates for certain tournament events
        let tables = self.tables.read().await;
        let mut table_ids = Vec::new();

        for (table_id, table) in tables.iter() {
            if let Some(tid) = &table.tournament_id {
                if tid == tournament_id {
                    table_ids.push(table_id.clone());
                }
            }
        }

        drop(tables);

        // Trigger table state updates
        let table_broadcasts = self.table_broadcasts.read().await;
        for table_id in table_ids {
            if let Some(tx) = table_broadcasts.get(&table_id) {
                let _ = tx.send(TableBroadcast {
                    table_id: table_id.clone(),
                });
            }
        }

        let club_id: Option<(String,)> =
            sqlx::query_as("SELECT club_id FROM tournaments WHERE id = ?")
                .bind(tournament_id)
                .fetch_optional(self.pool.as_ref())
                .await
                .ok()
                .flatten();
        if let Some((club_id,)) = club_id {
            let broadcasts = self.tournament_broadcasts.read().await;
            if let Some(tx) = broadcasts.get(&club_id) {
                let _ = tx.send(message);
                tracing::debug!(
                    "Broadcasted tournament event for {} to club {}",
                    tournament_id,
                    club_id
                );
            }
        }
    }

    async fn get_or_create_broadcast(&self, table_id: &str) -> broadcast::Receiver<TableBroadcast> {
        let mut broadcasts = self.table_broadcasts.write().await;
        let tx = broadcasts.entry(table_id.to_string()).or_insert_with(|| {
            let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
            tx
        });
        tx.subscribe()
    }

    async fn notify_table_update(&self, table_id: &str) {
        let broadcasts = self.table_broadcasts.read().await;
        if let Some(tx) = broadcasts.get(table_id) {
            let _ = tx.send(TableBroadcast {
                table_id: table_id.to_string(),
            });
        }
    }

    /// Fetch tournament info for a table to include in state
    async fn get_tournament_info(
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
                .ok()
                .flatten();

        let tournament = tournament?;

        // Get next blind level
        let next_level: Option<TournamentBlindLevel> = sqlx::query_as(
            "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? AND level_number = ?",
        )
        .bind(tournament_id)
        .bind(tournament.current_blind_level + 1)
        .fetch_optional(self.pool.as_ref())
        .await
        .ok()
        .flatten();

        Some(crate::game::TournamentInfo {
            tournament_id: tournament_id.to_string(),
            blind_level: tournament.current_blind_level as i64,
            level_start_time: tournament.level_start_time,
            level_duration_secs: tournament.level_duration_secs,
            next_small_blind: next_level.as_ref().map(|l| l.small_blind),
            next_big_blind: next_level.as_ref().map(|l| l.big_blind),
        })
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

    /// Check all tables for bots whose turn it is and execute their actions.
    pub async fn check_bot_actions(&self) {
        // Phase 1: Check for broke bots and auto top-up
        let topups = {
            let tables = self.tables.read().await;
            let bot_mgr = self.bot_manager.read().await;
            let mut topups = Vec::new();

            for (table_id, table) in tables.iter() {
                // Only in cash games (can_top_up)
                if !table.can_top_up() {
                    continue;
                }

                for player in &table.players {
                    // Check if bot is broke or very low on chips
                    if bot_mgr.is_bot(&player.user_id) && player.stack < table.big_blind * 10 {
                        let topup_amount = table.big_blind * 100; // Top up to 100BB
                        topups.push((table_id.clone(), player.user_id.clone(), topup_amount));
                    }
                }
            }
            topups
        };

        // Execute top-ups
        for (table_id, user_id, amount) in topups {
            let mut tables = self.tables.write().await;
            if let Some(table) = tables.get_mut(&table_id) {
                if let Ok(()) = table.top_up(&user_id, amount) {
                    tracing::info!(
                        "Bot {} topped up ${} at table {}",
                        user_id,
                        amount,
                        table_id
                    );
                    // Notify clients about the top-up
                    drop(tables); // Release lock before notify
                    self.notify_table_update(&table_id).await;
                }
            }
        }

        // Phase 2: read tables + bot manager to collect actions
        let actions = {
            let tables = self.tables.read().await;
            let bot_mgr = self.bot_manager.read().await;
            bot_mgr.collect_bot_actions(&tables)
        };

        if actions.is_empty() {
            return;
        }

        // Phase 3: execute each action (needs write lock)
        for (table_id, user_id, action) in actions {
            let result = {
                let mut tables = self.tables.write().await;
                if let Some(table) = tables.get_mut(&table_id) {
                    table.handle_action(&user_id, action).ok()
                } else {
                    None
                }
            };

            if result.is_some() {
                self.notify_table_update(&table_id).await;
            }
        }
    }

    /// Add a bot to a table. Returns (bot_user_id, bot_username) or error.
    pub async fn add_bot_to_table(
        &self,
        table_id: &str,
        buyin: i64,
        name: Option<String>,
        strategy: Option<&str>,
    ) -> Result<(String, String), String> {
        // Register bot in manager
        let (bot_user_id, bot_username) = {
            let mut bot_mgr = self.bot_manager.write().await;
            bot_mgr.add_bot(table_id, name, strategy)
        };

        // Seat the bot at the table
        {
            let mut tables = self.tables.write().await;
            let table = tables
                .get_mut(table_id)
                .ok_or_else(|| "Table not found".to_string())?;
            table
                .add_player(bot_user_id.clone(), bot_username.clone(), buyin)
                .map_err(|e| {
                    // Clean up bot registration on failure
                    // (can't await here, so we'll do a blocking attempt)
                    e.to_string()
                })?;
        }

        self.notify_table_update(table_id).await;

        tracing::info!(
            "Bot {} ({}) added to table {} with {} chips",
            bot_username,
            bot_user_id,
            table_id,
            buyin
        );

        Ok((bot_user_id, bot_username))
    }

    /// Register an existing user as a bot with the bot manager (for tournament bots)
    pub async fn register_bot(
        &self,
        table_id: &str,
        user_id: String,
        username: String,
        strategy: Option<&str>,
    ) {
        let mut bot_mgr = self.bot_manager.write().await;

        // Register this user as a bot for this table
        bot_mgr.register_existing_bot(table_id, user_id.clone(), username.clone(), strategy);

        tracing::info!(
            "Registered existing user {} ({}) as bot on table {}",
            username,
            user_id,
            table_id
        );
    }

    /// Remove a bot from a table.
    pub async fn remove_bot_from_table(
        &self,
        table_id: &str,
        bot_user_id: &str,
    ) -> Result<(), String> {
        // Remove from bot manager
        {
            let mut bot_mgr = self.bot_manager.write().await;
            if !bot_mgr.remove_bot(table_id, bot_user_id) {
                return Err("Bot not found".to_string());
            }
        }

        // Remove from table
        {
            let mut tables = self.tables.write().await;
            if let Some(table) = tables.get_mut(table_id) {
                table.remove_player(bot_user_id);
            }
        }

        self.notify_table_update(table_id).await;

        tracing::info!("Bot {} removed from table {}", bot_user_id, table_id);
        Ok(())
    }

    // Load table from database into memory
    async fn load_table_from_db(&self, table_id: &str) -> Result<(), String> {
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
}

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    State(game_server): State<Arc<GameServer>>,
) -> Response {
    // Verify JWT token
    let claims = match game_server.jwt_manager.verify_token(&query.token) {
        Ok(claims) => claims,
        Err(_) => {
            return (axum::http::StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
        }
    };

    ws.on_upgrade(move |socket| handle_socket(socket, claims.sub, claims.username, game_server))
}

async fn handle_socket(
    socket: WebSocket,
    user_id: String,
    username: String,
    game_server: Arc<GameServer>,
) {
    let (mut sender, mut receiver) = socket.split();

    // Send connected message
    let _ = sender
        .send(Message::Text(
            serde_json::to_string(&ServerMessage::Connected).unwrap(),
        ))
        .await;

    let mut current_table_id: Option<String> = None;
    let mut broadcast_rx: Option<broadcast::Receiver<TableBroadcast>> = None;
    let mut current_club_id: Option<String> = None;
    let mut club_broadcast_rx: Option<broadcast::Receiver<ClubBroadcast>> = None;
    let mut global_broadcast_rx: Option<broadcast::Receiver<GlobalBroadcast>> = None;
    let mut tournament_broadcast_rx: Option<broadcast::Receiver<ServerMessage>> = None;

    //Check if user was already at a table (reconnection after server restart)
    // For now, we don't persist this, but we could add it later

    loop {
        tokio::select! {
            // Handle incoming WebSocket messages from client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(client_msg) = serde_json::from_str::<ClientMessage>(&text) {
                            let response = handle_client_message(
                                client_msg,
                                &user_id,
                                &username,
                                &game_server,
                                &mut current_table_id,
                                &mut broadcast_rx,
                                &mut current_club_id,
                                &mut club_broadcast_rx,
                                &mut global_broadcast_rx,
                                &mut tournament_broadcast_rx,
                            ).await;

                            // Send response
                            if let Ok(response_text) = serde_json::to_string(&response) {
                                let _ = sender.send(Message::Text(response_text)).await;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        // Client disconnected - clean up
                        if let Some(table_id) = &current_table_id {
                            let mut tables = game_server.tables.write().await;
                            if let Some(table) = tables.get_mut(table_id) {
                                table.remove_player(&user_id);
                                game_server.notify_table_update(table_id).await;
                            }
                        }
                        break;
                    }
                    _ => {}
                }
            }

            // Handle broadcast notifications from table
            broadcast_msg = async {
                match &mut broadcast_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Ok(_update) = broadcast_msg {
                    // Send updated table state to this client
                    if let Some(table_id) = &current_table_id {
                        let tables = game_server.tables.read().await;
                        if let Some(table) = tables.get(table_id) {
                            let tournament_info = if let Some(tid) = &table.tournament_id {
                                game_server.get_tournament_info(tid).await
                            } else {
                                None
                            };
                            drop(tables);

                            let tables = game_server.tables.read().await;
                            if let Some(table) = tables.get(table_id) {
                                let state = table.get_public_state_with_tournament(Some(&user_id), tournament_info);
                                let msg = ServerMessage::TableState(state);
                                if let Ok(msg_text) = serde_json::to_string(&msg) {
                                    let _ = sender.send(Message::Text(msg_text)).await;
                                }
                            }
                        }
                    }
                }
            }

            // Handle broadcast notifications from club
            club_msg = async {
                match &mut club_broadcast_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Ok(_update) = club_msg {
                    // Notify client that club data has changed (tables list, members, etc.)
                    tracing::debug!("Club broadcast received for club_id: {:?}", current_club_id);
                    let msg = ServerMessage::ClubUpdate;
                    if let Ok(msg_text) = serde_json::to_string(&msg) {
                        let _ = sender.send(Message::Text(msg_text)).await;
                    }
                }
            }

            // Handle global broadcast notifications
            global_msg = async {
                match &mut global_broadcast_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Ok(_update) = global_msg {
                    // Notify client of global events (new clubs, etc.)
                    tracing::debug!("Global broadcast received");
                    let msg = ServerMessage::GlobalUpdate;
                    if let Ok(msg_text) = serde_json::to_string(&msg) {
                        let _ = sender.send(Message::Text(msg_text)).await;
                    }
                }
            }

            // Handle tournament broadcast notifications
            tournament_msg = async {
                match &mut tournament_broadcast_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Ok(msg) = tournament_msg {
                    if let Ok(msg_text) = serde_json::to_string(&msg) {
                        let _ = sender.send(Message::Text(msg_text)).await;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)] // These parameters represent connection state
async fn handle_client_message(
    msg: ClientMessage,
    user_id: &str,
    username: &str,
    game_server: &Arc<GameServer>,
    current_table_id: &mut Option<String>,
    broadcast_rx: &mut Option<broadcast::Receiver<TableBroadcast>>,
    current_club_id: &mut Option<String>,
    club_broadcast_rx: &mut Option<broadcast::Receiver<ClubBroadcast>>,
    global_broadcast_rx: &mut Option<broadcast::Receiver<GlobalBroadcast>>,
    tournament_broadcast_rx: &mut Option<broadcast::Receiver<ServerMessage>>,
) -> ServerMessage {
    // Helper to get table state with tournament info
    async fn get_table_state_with_tournament(
        game_server: &GameServer,
        table_id: &str,
        user_id: &str,
    ) -> Option<crate::game::PublicTableState> {
        // First, get tournament ID if present
        let tournament_id = {
            let tables = game_server.tables.read().await;
            tables.get(table_id)?.tournament_id.clone()
        };

        // Fetch tournament info if needed
        let tournament_info = if let Some(tid) = tournament_id {
            game_server.get_tournament_info(&tid).await
        } else {
            None
        };

        // Get table state with tournament info
        let tables = game_server.tables.read().await;
        let table = tables.get(table_id)?;
        Some(table.get_public_state_with_tournament(Some(user_id), tournament_info))
    }

    match msg {
        ClientMessage::ViewingClubsList => {
            // User is viewing the global clubs list
            // Subscribe to global broadcasts for new clubs
            *global_broadcast_rx = Some(game_server.subscribe_global());
            *current_club_id = None;
            *club_broadcast_rx = None;
            *tournament_broadcast_rx = None;
            tracing::debug!("User {} subscribed to global broadcasts", username);
            ServerMessage::Connected
        }

        ClientMessage::ViewingClub { club_id } => {
            // User is viewing a specific club's lobby
            // Subscribe to club broadcasts for new tables
            *current_club_id = Some(club_id.clone());
            *club_broadcast_rx = Some(game_server.subscribe_club(&club_id).await);
            *tournament_broadcast_rx = Some(game_server.subscribe_tournament_club(&club_id).await);
            tracing::debug!(
                "User {} subscribed to club broadcasts for club {}",
                username,
                club_id
            );
            ServerMessage::Connected
        }

        ClientMessage::LeavingView => {
            // User left the current view
            // Unsubscribe from club/global broadcasts
            *current_club_id = None;
            *club_broadcast_rx = None;
            *global_broadcast_rx = None;
            *tournament_broadcast_rx = None;
            tracing::debug!("User {} unsubscribed from all view broadcasts", username);
            ServerMessage::Connected
        }
        ClientMessage::JoinTable { table_id, buyin } => {
            // Check if table exists, load from DB if needed
            let table_exists = game_server.tables.read().await.contains_key(&table_id);

            if !table_exists {
                if let Err(e) = game_server.load_table_from_db(&table_id).await {
                    return ServerMessage::Error {
                        message: format!("Table not found: {}", e),
                    };
                }
            }

            // Subscribe to table as observer or join with seat
            *current_table_id = Some(table_id.clone());
            *broadcast_rx = Some(game_server.get_or_create_broadcast(&table_id).await);

            // If this is a tournament table, subscribe to tournament broadcasts
            let is_tournament = {
                let tables = game_server.tables.read().await;
                tables
                    .get(&table_id)
                    .and_then(|t| t.tournament_id.as_ref())
                    .is_some()
            };

            if is_tournament {
                // Get club_id for the table/tournament
                let club_result: Result<(String,), sqlx::Error> =
                    sqlx::query_as("SELECT club_id FROM tables WHERE id = ?")
                        .bind(&table_id)
                        .fetch_one(game_server.pool.as_ref())
                        .await;

                if let Ok((club_id,)) = club_result {
                    *tournament_broadcast_rx =
                        Some(game_server.subscribe_tournament_club(&club_id).await);
                    tracing::info!(
                        "User {} subscribed to tournament broadcasts for club {}",
                        username,
                        club_id
                    );
                }
            }

            // If buyin is 0, just observe without taking a seat
            if buyin == 0 {
                if let Some(state) =
                    get_table_state_with_tournament(game_server, &table_id, user_id).await
                {
                    ServerMessage::TableState(state)
                } else {
                    ServerMessage::Error {
                        message: "Table not found".to_string(),
                    }
                }
            } else {
                // Join with a seat (legacy auto-seat behavior)
                let mut tables = game_server.tables.write().await;
                if let Some(table) = tables.get_mut(&table_id) {
                    match table.add_player(user_id.to_string(), username.to_string(), buyin) {
                        Ok(_seat) => {
                            drop(tables); // Release lock before notifying

                            // Notify all players at table
                            game_server.notify_table_update(&table_id).await;

                            // Get and return current state
                            if let Some(state) =
                                get_table_state_with_tournament(game_server, &table_id, user_id)
                                    .await
                            {
                                ServerMessage::TableState(state)
                            } else {
                                ServerMessage::Error {
                                    message: "Table disappeared".to_string(),
                                }
                            }
                        }
                        Err(e) => ServerMessage::Error {
                            message: e.to_string(),
                        },
                    }
                } else {
                    ServerMessage::Error {
                        message: "Table not found".to_string(),
                    }
                }
            }
        }

        ClientMessage::TakeSeat {
            table_id,
            seat,
            buyin,
        } => {
            // Check if table exists, load from DB if needed
            let table_exists = game_server.tables.read().await.contains_key(&table_id);

            if !table_exists {
                if let Err(e) = game_server.load_table_from_db(&table_id).await {
                    return ServerMessage::Error {
                        message: format!("Table not found: {}", e),
                    };
                }
            }

            // Take the specific seat at the table
            let mut tables = game_server.tables.write().await;
            if let Some(table) = tables.get_mut(&table_id) {
                match table.take_seat(user_id.to_string(), username.to_string(), seat, buyin) {
                    Ok(_seat_num) => {
                        *current_table_id = Some(table_id.clone());
                        drop(tables); // Release lock before subscribing

                        // Subscribe to table broadcasts
                        *broadcast_rx = Some(game_server.get_or_create_broadcast(&table_id).await);

                        // Notify all players at table
                        game_server.notify_table_update(&table_id).await;

                        // Get and return current state
                        if let Some(state) =
                            get_table_state_with_tournament(game_server, &table_id, user_id).await
                        {
                            ServerMessage::TableState(state)
                        } else {
                            ServerMessage::Error {
                                message: "Table disappeared".to_string(),
                            }
                        }
                    }
                    Err(e) => ServerMessage::Error {
                        message: e.to_string(),
                    },
                }
            } else {
                ServerMessage::Error {
                    message: "Table not found".to_string(),
                }
            }
        }

        ClientMessage::StandUp => {
            if let Some(ref table_id) = current_table_id {
                let mut tables = game_server.tables.write().await;
                if let Some(table) = tables.get_mut(table_id) {
                    match table.stand_up(user_id) {
                        Ok(_) => {
                            // Check if player is still at table (might have been removed)
                            let still_seated = table.players.iter().any(|p| p.user_id == user_id);

                            drop(tables);
                            game_server.notify_table_update(table_id).await;

                            // If player was removed, clear their table subscription
                            if !still_seated {
                                *current_table_id = None;
                                *broadcast_rx = None;
                            }

                            ServerMessage::Connected
                        }
                        Err(e) => ServerMessage::Error {
                            message: e.to_string(),
                        },
                    }
                } else {
                    ServerMessage::Error {
                        message: "Table not found".to_string(),
                    }
                }
            } else {
                ServerMessage::Error {
                    message: "Not at a table".to_string(),
                }
            }
        }

        ClientMessage::TopUp { amount } => {
            if let Some(ref table_id) = current_table_id {
                let mut tables = game_server.tables.write().await;
                if let Some(table) = tables.get_mut(table_id) {
                    match table.top_up(user_id, amount) {
                        Ok(_) => {
                            drop(tables);
                            game_server.notify_table_update(table_id).await;
                            ServerMessage::Connected
                        }
                        Err(e) => ServerMessage::Error {
                            message: e.to_string(),
                        },
                    }
                } else {
                    ServerMessage::Error {
                        message: "Table not found".to_string(),
                    }
                }
            } else {
                ServerMessage::Error {
                    message: "Not at a table".to_string(),
                }
            }
        }

        ClientMessage::LeaveTable => {
            if let Some(ref table_id) = current_table_id {
                let mut tables = game_server.tables.write().await;
                if let Some(table) = tables.get_mut(table_id) {
                    table.remove_player(user_id);
                    drop(tables);
                    game_server.notify_table_update(table_id).await;
                }
                *current_table_id = None;
                *broadcast_rx = None;
            }
            ServerMessage::Connected
        }

        ClientMessage::PlayerAction { action } => {
            if let Some(ref table_id) = current_table_id {
                let mut tables = game_server.tables.write().await;
                if let Some(table) = tables.get_mut(table_id) {
                    match table.handle_action(user_id, action) {
                        Ok(_) => {
                            drop(tables); // Release lock before notifying
                                          // Notify all players at table
                            game_server.notify_table_update(table_id).await;
                            // Return success - table state will be sent via broadcast
                            ServerMessage::Connected
                        }
                        Err(e) => ServerMessage::Error {
                            message: e.to_string(),
                        },
                    }
                } else {
                    ServerMessage::Error {
                        message: "Table not found".to_string(),
                    }
                }
            } else {
                ServerMessage::Error {
                    message: "Not at a table".to_string(),
                }
            }
        }

        ClientMessage::GetTableState => {
            if let Some(ref table_id) = current_table_id {
                if let Some(state) =
                    get_table_state_with_tournament(game_server, table_id, user_id).await
                {
                    ServerMessage::TableState(state)
                } else {
                    ServerMessage::Error {
                        message: "Table not found".to_string(),
                    }
                }
            } else {
                ServerMessage::Error {
                    message: "Not at a table".to_string(),
                }
            }
        }

        ClientMessage::Ping => ServerMessage::Pong,

        ClientMessage::AddBot {
            table_id,
            name,
            strategy,
        } => {
            // Default buyin: use table's big blind * 100
            let buyin = {
                let tables = game_server.tables.read().await;
                tables
                    .get(&table_id)
                    .map(|t| t.big_blind * 100)
                    .unwrap_or(1000)
            };

            match game_server
                .add_bot_to_table(&table_id, buyin, name, strategy.as_deref())
                .await
            {
                Ok((bot_id, bot_name)) => {
                    tracing::info!(
                        "User {} added bot {} ({}) to table {}",
                        username,
                        bot_name,
                        bot_id,
                        table_id
                    );
                    // Return current table state
                    if let Some(state) =
                        get_table_state_with_tournament(game_server, &table_id, user_id).await
                    {
                        ServerMessage::TableState(state)
                    } else {
                        ServerMessage::Connected
                    }
                }
                Err(e) => ServerMessage::Error { message: e },
            }
        }

        ClientMessage::RemoveBot {
            table_id,
            bot_user_id,
        } => {
            match game_server
                .remove_bot_from_table(&table_id, &bot_user_id)
                .await
            {
                Ok(()) => {
                    if let Some(state) =
                        get_table_state_with_tournament(game_server, &table_id, user_id).await
                    {
                        ServerMessage::TableState(state)
                    } else {
                        ServerMessage::Connected
                    }
                }
                Err(e) => ServerMessage::Error { message: e },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GameServer;
    use crate::auth::JwtManager;
    use crate::ws::messages::ServerMessage;
    use std::sync::Arc;
    use uuid::Uuid;

    #[tokio::test]
    async fn tournament_broadcast_reaches_subscribers() {
        let pool = crate::create_test_db().await;
        let jwt_manager = Arc::new(JwtManager::new("test-secret".to_string()));
        let server = GameServer::new(jwt_manager, Arc::new(pool));
        let club_id = Uuid::new_v4().to_string();
        let tournament_id = Uuid::new_v4().to_string();
        let admin_id = Uuid::new_v4().to_string();

        sqlx::query("INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)")
            .bind(&admin_id)
            .bind("admin")
            .bind("admin@example.com")
            .bind("hashed")
            .execute(server.pool.as_ref())
            .await
            .expect("insert user");

        sqlx::query("INSERT INTO clubs (id, name, admin_id) VALUES (?, ?, ?)")
            .bind(&club_id)
            .bind("Test Club")
            .bind(&admin_id)
            .execute(server.pool.as_ref())
            .await
            .expect("insert club");

        sqlx::query(
            "INSERT INTO tournaments (
                id, club_id, name, format_id, variant_id, buy_in, starting_stack, prize_pool,
                max_players, min_players, registered_players, remaining_players, current_blind_level,
                level_duration_secs, status, pre_seat_secs, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&tournament_id)
        .bind(&club_id)
        .bind("Test Tourney")
        .bind("sng")
        .bind("holdem")
        .bind(100)
        .bind(1500)
        .bind(0)
        .bind(9)
        .bind(2)
        .bind(0)
        .bind(0)
        .bind(0)
        .bind(300)
        .bind("registering")
        .bind(0)
        .execute(server.pool.as_ref())
        .await
        .expect("insert tournament");

        let mut rx = server.subscribe_tournament_club(&club_id).await;

        server
            .broadcast_tournament_event(
                &tournament_id,
                ServerMessage::TournamentStarted {
                    tournament_id: tournament_id.clone(),
                    tournament_name: "Test Tourney".to_string(),
                    table_id: None,
                },
            )
            .await;

        let expected_tournament_id = tournament_id.clone();
        let message = rx.recv().await.expect("broadcast message");
        match message {
            ServerMessage::TournamentStarted { tournament_id, .. } => {
                assert_eq!(tournament_id, expected_tournament_id);
            }
            other => panic!("unexpected message: {:?}", other),
        }
    }
}
