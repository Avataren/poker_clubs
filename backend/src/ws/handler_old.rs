use crate::{
    auth::JwtManager,
    game::PokerTable,
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

#[derive(Deserialize)]
pub struct WsQuery {
    token: String,
}

// Broadcast message that includes the table_id and the state update
#[derive(Debug, Clone)]
struct TableBroadcast {
    table_id: String,
}

pub struct GameServer {
    tables: Arc<RwLock<HashMap<String, PokerTable>>>,
    jwt_manager: Arc<JwtManager>,
    pool: Arc<sqlx::SqlitePool>,
    // Broadcast channel for each table - sends notification when table state changes
    table_broadcasts: Arc<RwLock<HashMap<String, broadcast::Sender<TableBroadcast>>>>,
}

impl GameServer {
    pub fn new(jwt_manager: Arc<JwtManager>, pool: Arc<sqlx::SqlitePool>) -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            jwt_manager,
            pool,
            table_broadcasts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_table(&self, table_id: String, name: String, small_blind: i64, big_blind: i64) {
        let table = PokerTable::new(table_id.clone(), name, small_blind, big_blind);
        self.tables.write().await.insert(table_id.clone(), table);

        // Create broadcast channel for this table (capacity 100)
        let (tx, _rx) = broadcast::channel(100);
        self.table_broadcasts.write().await.insert(table_id, tx);
    }

    async fn get_or_create_broadcast(&self, table_id: &str) -> broadcast::Receiver<TableBroadcast> {
        let mut broadcasts = self.table_broadcasts.write().await;
        let tx = broadcasts.entry(table_id.to_string())
            .or_insert_with(|| {
                let (tx, _rx) = broadcast::channel(100);
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

    pub async fn get_table_state(&self, table_id: &str, user_id: Option<&str>) -> Option<crate::game::PublicTableState> {
        let tables = self.tables.read().await;
        tables.get(table_id).map(|t| t.get_public_state(user_id))
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
            let (tx, _rx) = broadcast::channel(100);
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

    // Handle incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            let client_msg: Result<ClientMessage, _> = serde_json::from_str(&text);

            let response = match client_msg {
                Ok(ClientMessage::JoinTable { table_id, buyin }) => {
                    // Check if table exists in memory, if not try loading from database
                    let table_exists = {
                        let tables = game_server.tables.read().await;
                        tables.contains_key(&table_id)
                    };

                    if !table_exists {
                        // Try to load from database
                        if let Err(e) = game_server.load_table_from_db(&table_id).await {
                            ServerMessage::Error {
                                message: format!("Table not found: {}", e),
                            }
                        } else {
                            // Table loaded successfully, continue to join
                            let mut tables = game_server.tables.write().await;
                            if let Some(table) = tables.get_mut(&table_id) {
                                match table.add_player(user_id.clone(), username.clone(), buyin) {
                                    Ok(_seat) => {
                                        current_table_id = Some(table_id.clone());
                                        let state = table.get_public_state(Some(&user_id));
                                        ServerMessage::TableState(state)
                                    }
                                    Err(e) => ServerMessage::Error { message: e },
                                }
                            } else {
                                ServerMessage::Error {
                                    message: "Table not found after loading".to_string(),
                                }
                            }
                        }
                    } else {
                        // Table exists in memory, join it
                        let mut tables = game_server.tables.write().await;
                        if let Some(table) = tables.get_mut(&table_id) {
                            match table.add_player(user_id.clone(), username.clone(), buyin) {
                                Ok(_seat) => {
                                    current_table_id = Some(table_id.clone());
                                    let state = table.get_public_state(Some(&user_id));
                                    ServerMessage::TableState(state)
                                }
                                Err(e) => ServerMessage::Error { message: e },
                            }
                        } else {
                            ServerMessage::Error {
                                message: "Table not found".to_string(),
                            }
                        }
                    }
                }

                Ok(ClientMessage::LeaveTable) => {
                    if let Some(ref table_id) = current_table_id {
                        let mut tables = game_server.tables.write().await;
                        if let Some(table) = tables.get_mut(table_id) {
                            table.remove_player(&user_id);
                        }
                        current_table_id = None;
                    }
                    ServerMessage::Connected
                }

                Ok(ClientMessage::PlayerAction { action }) => {
                    if let Some(ref table_id) = current_table_id {
                        let mut tables = game_server.tables.write().await;

                        if let Some(table) = tables.get_mut(table_id) {
                            match table.handle_action(&user_id, action) {
                                Ok(_) => {
                                    // Send updated state
                                    let state = table.get_public_state(Some(&user_id));
                                    ServerMessage::TableState(state)
                                }
                                Err(e) => ServerMessage::Error { message: e },
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

                Ok(ClientMessage::GetTableState) => {
                    if let Some(ref table_id) = current_table_id {
                        let tables = game_server.tables.read().await;
                        if let Some(table) = tables.get(table_id) {
                            let state = table.get_public_state(Some(&user_id));
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

                Ok(ClientMessage::Ping) => ServerMessage::Pong,

                Err(_) => ServerMessage::Error {
                    message: "Invalid message format".to_string(),
                },
            };

            // Send response
            if let Ok(response_text) = serde_json::to_string(&response) {
                let _ = sender.send(Message::Text(response_text)).await;
            }
        }
    }

    // Cleanup on disconnect
    if let Some(table_id) = current_table_id {
        let mut tables = game_server.tables.write().await;
        if let Some(table) = tables.get_mut(&table_id) {
            table.remove_player(&user_id);
        }
    }
}
