use crate::{
    auth::JwtManager,
    game::{constants::BROADCAST_CHANNEL_CAPACITY, PokerTable},
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
use std::{
    collections::HashMap,
    sync::Arc,
};
use tokio::sync::{RwLock, broadcast};

// Broadcast message types - prepared for pub/sub broadcasting
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TableBroadcast {
    table_id: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ClubBroadcast {
    club_id: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct GlobalBroadcast {
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
        }
    }

    #[allow(dead_code)] // Prepared for global event subscriptions
    pub fn subscribe_global(&self) -> broadcast::Receiver<GlobalBroadcast> {
        self.global_broadcast.subscribe()
    }

    #[allow(dead_code)] // Prepared for club-level event subscriptions
    pub async fn subscribe_club(&self, club_id: &str) -> broadcast::Receiver<ClubBroadcast> {
        let mut broadcasts = self.club_broadcasts.write().await;
        let tx = broadcasts.entry(club_id.to_string())
            .or_insert_with(|| {
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

    pub async fn create_table(&self, table_id: String, name: String, small_blind: i64, big_blind: i64) {
        let table = PokerTable::new(table_id.clone(), name, small_blind, big_blind);
        self.tables.write().await.insert(table_id.clone(), table);

        // Create broadcast channel for this table
        let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
        self.table_broadcasts.write().await.insert(table_id, tx);
    }

    #[allow(dead_code)] // Useful for REST API to get table state
    pub async fn get_table_state(&self, table_id: &str, user_id: Option<&str>) -> Option<crate::game::PublicTableState> {
        let tables = self.tables.read().await;
        tables.get(table_id).map(|t| t.get_public_state(user_id))
    }

    async fn get_or_create_broadcast(&self, table_id: &str) -> broadcast::Receiver<TableBroadcast> {
        let mut broadcasts = self.table_broadcasts.write().await;
        let tx = broadcasts.entry(table_id.to_string())
            .or_insert_with(|| {
                let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
                tx
            });
        tx.subscribe()
    }

    async fn notify_table_update(&self, table_id: &str) {
        let broadcasts = self.table_broadcasts.read().await;
        if let Some(tx) = broadcasts.get(table_id) {
            let _ = tx.send(TableBroadcast { table_id: table_id.to_string() });
        }
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

    // Load table from database into memory
    async fn load_table_from_db(&self, table_id: &str) -> Result<(), String> {
        // Query database for table
        let table_data: Option<(String, String, String, i64, i64)> = sqlx::query_as(
            "SELECT id, club_id, name, small_blind, big_blind FROM tables WHERE id = ?",
        )
        .bind(table_id)
        .fetch_optional(self.pool.as_ref())
        .await
        .map_err(|e| format!("Database error: {}", e))?;

        if let Some((id, _club_id, name, small_blind, big_blind)) = table_data {
            // Create in-memory table
            let table = PokerTable::new(id.clone(), name, small_blind, big_blind);
            self.tables.write().await.insert(id.clone(), table);

            // Create broadcast channel for this table
            let (tx, _rx) = broadcast::channel(BROADCAST_CHANNEL_CAPACITY);
            self.table_broadcasts.write().await.insert(id, tx);

            Ok(())
        } else {
            Err("Table not found in database".to_string())
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
                            let state = table.get_public_state(Some(&user_id));
                            let msg = ServerMessage::TableState(state);
                            if let Ok(msg_text) = serde_json::to_string(&msg) {
                                let _ = sender.send(Message::Text(msg_text)).await;
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
        }
    }
}

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
) -> ServerMessage {
    match msg {
        ClientMessage::ViewingClubsList => {
            // User is viewing the global clubs list
            // Subscribe to global broadcasts for new clubs
            *global_broadcast_rx = Some(game_server.subscribe_global());
            *current_club_id = None;
            *club_broadcast_rx = None;
            tracing::debug!("User {} subscribed to global broadcasts", username);
            ServerMessage::Connected
        }

        ClientMessage::ViewingClub { club_id } => {
            // User is viewing a specific club's lobby
            // Subscribe to club broadcasts for new tables
            *current_club_id = Some(club_id.clone());
            *club_broadcast_rx = Some(game_server.subscribe_club(&club_id).await);
            tracing::debug!("User {} subscribed to club broadcasts for club {}", username, club_id);
            ServerMessage::Connected
        }

        ClientMessage::LeavingView => {
            // User left the current view
            // Unsubscribe from club/global broadcasts
            *current_club_id = None;
            *club_broadcast_rx = None;
            *global_broadcast_rx = None;
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

            // If buyin is 0, just observe without taking a seat
            if buyin == 0 {
                let tables = game_server.tables.read().await;
                if let Some(table) = tables.get(&table_id) {
                    let state = table.get_public_state(Some(user_id));
                    ServerMessage::TableState(state)
                } else {
                    ServerMessage::Error { message: "Table not found".to_string() }
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
                            let tables = game_server.tables.read().await;
                            if let Some(table) = tables.get(&table_id) {
                                let state = table.get_public_state(Some(user_id));
                                ServerMessage::TableState(state)
                            } else {
                                ServerMessage::Error { message: "Table disappeared".to_string() }
                            }
                        }
                        Err(e) => ServerMessage::Error { message: e.to_string() },
                    }
                } else {
                    ServerMessage::Error { message: "Table not found".to_string() }
                }
            }
        }

        ClientMessage::TakeSeat { table_id, seat, buyin } => {
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
                        let tables = game_server.tables.read().await;
                        if let Some(table) = tables.get(&table_id) {
                            let state = table.get_public_state(Some(user_id));
                            ServerMessage::TableState(state)
                        } else {
                            ServerMessage::Error { message: "Table disappeared".to_string() }
                        }
                    }
                    Err(e) => ServerMessage::Error { message: e.to_string() },
                }
            } else {
                ServerMessage::Error { message: "Table not found".to_string() }
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
                        Err(e) => ServerMessage::Error { message: e.to_string() },
                    }
                } else {
                    ServerMessage::Error { message: "Table not found".to_string() }
                }
            } else {
                ServerMessage::Error { message: "Not at a table".to_string() }
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
                        Err(e) => ServerMessage::Error { message: e.to_string() },
                    }
                } else {
                    ServerMessage::Error { message: "Table not found".to_string() }
                }
            } else {
                ServerMessage::Error { message: "Not at a table".to_string() }
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
                        Err(e) => ServerMessage::Error { message: e.to_string() },
                    }
                } else {
                    ServerMessage::Error { message: "Table not found".to_string() }
                }
            } else {
                ServerMessage::Error { message: "Not at a table".to_string() }
            }
        }

        ClientMessage::GetTableState => {
            if let Some(ref table_id) = current_table_id {
                let tables = game_server.tables.read().await;
                if let Some(table) = tables.get(table_id) {
                    let state = table.get_public_state(Some(user_id));
                    ServerMessage::TableState(state)
                } else {
                    ServerMessage::Error { message: "Table not found".to_string() }
                }
            } else {
                ServerMessage::Error { message: "Not at a table".to_string() }
            }
        }

        ClientMessage::Ping => ServerMessage::Pong,
    }
}
