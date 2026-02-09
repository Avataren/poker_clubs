mod bot_integration;
mod game_server;
mod message_router;
mod tournament_ops;

pub use game_server::GameServer;

use crate::{game::player::PlayerState, ws::messages::ServerMessage, ws::rate_limit::RateLimiter};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::broadcast;

use message_router::handle_client_message;

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

    let token_expires_at = DateTime::from_timestamp(claims.exp as i64, 0)
        .unwrap_or_else(|| Utc::now() + chrono::Duration::hours(1));

    ws.max_message_size(8 * 1024) // 8KB max message size
        .on_upgrade(move |socket| {
            handle_socket(
                socket,
                claims.sub,
                claims.username,
                game_server,
                token_expires_at,
            )
        })
}

async fn handle_socket(
    socket: WebSocket,
    user_id: String,
    username: String,
    game_server: Arc<GameServer>,
    token_expires_at: DateTime<Utc>,
) {
    let (mut sender, mut receiver) = socket.split();

    // Send connected message
    if let Ok(json) = serde_json::to_string(&ServerMessage::Connected) {
        let _ = sender.send(Message::Text(json)).await;
    }

    let mut current_table_id: Option<String> = None;
    let mut broadcast_rx: Option<broadcast::Receiver<TableBroadcast>> = None;
    let mut current_club_id: Option<String> = None;
    let mut club_broadcast_rx: Option<broadcast::Receiver<ClubBroadcast>> = None;
    let mut global_broadcast_rx: Option<broadcast::Receiver<GlobalBroadcast>> = None;
    let mut tournament_broadcast_rx: Option<broadcast::Receiver<ServerMessage>> = None;
    let mut rate_limiter = RateLimiter::new(10.0, 20.0);
    let mut token_check_interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
    let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
    let mut last_pong = tokio::time::Instant::now();

    loop {
        tokio::select! {
            // Handle incoming WebSocket messages from client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Rate limit check
                        if !rate_limiter.allow() {
                            let err = ServerMessage::Error {
                                message: "Rate limited: too many messages".to_string(),
                            };
                            if let Ok(err_text) = serde_json::to_string(&err) {
                                let _ = sender.send(Message::Text(err_text)).await;
                            }
                            continue;
                        }

                        if let Ok(client_msg) = serde_json::from_str::<crate::ws::messages::ClientMessage>(&text) {
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
                    Some(Ok(Message::Pong(_))) => {
                        last_pong = tokio::time::Instant::now();
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        // Client disconnected - clean up
                        if let Some(table_id) = &current_table_id {
                            // Check if the player is actually seated at the table
                            let is_seated = {
                                let tables = game_server.tables.read().await;
                                tables
                                    .get(table_id)
                                    .map(|t| t.players.iter().any(|p| p.user_id == user_id))
                                    .unwrap_or(false)
                            };

                            if is_seated {
                                // Player is seated: mark as disconnected instead of removing
                                game_server.mark_disconnected(&user_id, table_id).await;

                                // Set their PlayerState to Disconnected
                                let mut tables = game_server.tables.write().await;
                                if let Some(table) = tables.get_mut(table_id) {
                                    if let Some(player) = table.players.iter_mut().find(|p| p.user_id == user_id) {
                                        player.state = PlayerState::Disconnected;
                                        tracing::info!(
                                            "Player {} disconnected from table {}, entering grace period",
                                            user_id,
                                            table_id
                                        );
                                    }
                                }
                                drop(tables);
                                game_server.notify_table_update(table_id).await;
                            }
                            // If not seated (just observing), do nothing special
                        }
                        break;
                    }
                    _ => {}
                }
            }

            // Server-side heartbeat: send ping every 30 seconds
            _ = ping_interval.tick() => {
                // Check if we received a pong recently (within 10 seconds of the last ping)
                if last_pong.elapsed() > tokio::time::Duration::from_secs(40) {
                    tracing::warn!("No pong from user {} in 40s, closing connection", user_id);
                    if let Some(table_id) = &current_table_id {
                        // Check if the player is actually seated at the table
                        let is_seated = {
                            let tables = game_server.tables.read().await;
                            tables
                                .get(table_id)
                                .map(|t| t.players.iter().any(|p| p.user_id == user_id))
                                .unwrap_or(false)
                        };

                        if is_seated {
                            // Player is seated: mark as disconnected instead of removing
                            game_server.mark_disconnected(&user_id, table_id).await;

                            let mut tables = game_server.tables.write().await;
                            if let Some(table) = tables.get_mut(table_id) {
                                if let Some(player) = table.players.iter_mut().find(|p| p.user_id == user_id) {
                                    player.state = PlayerState::Disconnected;
                                }
                            }
                            drop(tables);
                            game_server.notify_table_update(table_id).await;
                        }
                    }
                    break;
                }
                if sender.send(Message::Ping(vec![])).await.is_err() {
                    break;
                }
            }

            // Periodic token expiry check
            _ = token_check_interval.tick() => {
                if Utc::now() >= token_expires_at {
                    let err = ServerMessage::Error {
                        message: "Token expired, please reconnect".to_string(),
                    };
                    if let Ok(err_text) = serde_json::to_string(&err) {
                        let _ = sender.send(Message::Text(err_text)).await;
                    }
                    // Clean up and close - mark as disconnected instead of removing
                    if let Some(table_id) = &current_table_id {
                        let is_seated = {
                            let tables = game_server.tables.read().await;
                            tables
                                .get(table_id)
                                .map(|t| t.players.iter().any(|p| p.user_id == user_id))
                                .unwrap_or(false)
                        };

                        if is_seated {
                            game_server.mark_disconnected(&user_id, table_id).await;

                            let mut tables = game_server.tables.write().await;
                            if let Some(table) = tables.get_mut(table_id) {
                                if let Some(player) = table.players.iter_mut().find(|p| p.user_id == user_id) {
                                    player.state = PlayerState::Disconnected;
                                }
                            }
                            drop(tables);
                            game_server.notify_table_update(table_id).await;
                        }
                    }
                    break;
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

#[cfg(test)]
mod tests {
    use super::GameServer;
    use crate::auth::JwtManager;
    use crate::game::format::MultiTableTournament;
    use crate::game::variant::TexasHoldem;
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

    #[tokio::test]
    async fn tournament_move_rejects_mid_hand_player_transfer() {
        let pool = crate::create_test_db().await;
        let jwt_manager = Arc::new(JwtManager::new("test-secret".to_string()));
        let server = GameServer::new(jwt_manager, Arc::new(pool));
        let source_table = format!("table-{}", Uuid::new_v4());
        let dest_table = format!("table-{}", Uuid::new_v4());
        let moved_user = Uuid::new_v4().to_string();

        let table_format = || {
            Box::new(MultiTableTournament::new(
                "Move Test".to_string(),
                100,
                1500,
                300,
            ))
        };
        server
            .create_table_with_options(
                source_table.clone(),
                "Source".to_string(),
                50,
                100,
                Box::new(TexasHoldem),
                table_format(),
            )
            .await;
        server
            .create_table_with_options(
                dest_table.clone(),
                "Destination".to_string(),
                50,
                100,
                Box::new(TexasHoldem),
                table_format(),
            )
            .await;

        server
            .add_player_to_table(
                &source_table,
                moved_user.clone(),
                "moved".to_string(),
                0,
                1000,
            )
            .await
            .unwrap();
        server
            .add_player_to_table(
                &source_table,
                Uuid::new_v4().to_string(),
                "other".to_string(),
                1,
                1000,
            )
            .await
            .unwrap();
        server
            .add_player_to_table(
                &dest_table,
                Uuid::new_v4().to_string(),
                "dest".to_string(),
                0,
                1000,
            )
            .await
            .unwrap();

        server.force_start_table_hand(&source_table).await;
        assert!(server.is_table_mid_hand(&source_table).await);

        let err = server
            .move_tournament_player(&source_table, &dest_table, &moved_user, None)
            .await
            .expect_err("move should be rejected while source table is mid-hand");
        assert!(err.contains("hand is in progress"));

        let source_players = server.get_all_player_ids_at_table(&source_table).await;
        let dest_players = server.get_all_player_ids_at_table(&dest_table).await;
        assert!(source_players.contains(&moved_user));
        assert!(!dest_players.contains(&moved_user));
    }
}
