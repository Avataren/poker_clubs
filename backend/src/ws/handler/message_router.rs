use crate::game::player::PlayerState;
use crate::game::table::GamePhase;
use crate::ws::messages::{ClientMessage, ServerMessage};
use std::sync::Arc;
use tokio::sync::broadcast;

use super::game_server::GameServer;
use super::{ClubBroadcast, GlobalBroadcast, TableBroadcast};

#[allow(clippy::too_many_arguments)] // These parameters represent connection state
pub(super) async fn handle_client_message(
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

    // Input validation helper
    fn validate_table_id(table_id: &str) -> Result<(), String> {
        if table_id.is_empty() || table_id.len() > 128 {
            return Err("Invalid table ID".to_string());
        }
        Ok(())
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
            if club_id.is_empty() || club_id.len() > 128 {
                return ServerMessage::Error {
                    message: "Invalid club ID".to_string(),
                };
            }
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
            if let Err(e) = validate_table_id(&table_id) {
                return ServerMessage::Error { message: e };
            }
            if buyin < 0 {
                return ServerMessage::Error {
                    message: "Buy-in cannot be negative".to_string(),
                };
            }

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

            // Check if the player is reconnecting (already seated but disconnected)
            let reconnecting = {
                let tables = game_server.tables.read().await;
                if let Some(table) = tables.get(&table_id) {
                    table.players.iter().any(|p| p.user_id == user_id && p.state == PlayerState::Disconnected)
                } else {
                    false
                }
            };

            if reconnecting {
                // Player is reconnecting to their existing seat
                game_server.clear_disconnected(user_id).await;

                // Restore their state based on whether a hand is in progress
                {
                    let mut tables = game_server.tables.write().await;
                    if let Some(table) = tables.get_mut(&table_id) {
                        if let Some(player) = table.players.iter_mut().find(|p| p.user_id == user_id) {
                            if table.phase == GamePhase::Waiting {
                                player.state = PlayerState::Active;
                            } else {
                                player.state = PlayerState::WaitingForHand;
                            }
                            tracing::info!(
                                "Player {} reconnected to table {}, state set to {:?}",
                                user_id,
                                table_id,
                                player.state
                            );
                        }
                    }
                }

                // Re-subscribe to broadcasts
                *broadcast_rx = Some(game_server.get_or_create_broadcast(&table_id).await);

                // Notify all players at table
                game_server.notify_table_update(&table_id).await;

                // Send them the current table state
                if let Some(state) =
                    get_table_state_with_tournament(game_server, &table_id, user_id).await
                {
                    ServerMessage::TableState(state)
                } else {
                    ServerMessage::Error {
                        message: "Table disappeared".to_string(),
                    }
                }
            } else if buyin == 0 {
                // Just observe without taking a seat
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
            if let Err(e) = validate_table_id(&table_id) {
                return ServerMessage::Error { message: e };
            }
            if buyin <= 0 {
                return ServerMessage::Error {
                    message: "Buy-in must be positive".to_string(),
                };
            }
            if seat > 20 {
                return ServerMessage::Error {
                    message: "Invalid seat number".to_string(),
                };
            }

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
            if amount <= 0 {
                return ServerMessage::Error {
                    message: "Top-up amount must be positive".to_string(),
                };
            }

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
            // Validate raise amount is positive
            if let crate::game::PlayerAction::Raise(amount) = &action {
                if *amount <= 0 {
                    return ServerMessage::Error {
                        message: "Raise amount must be positive".to_string(),
                    };
                }
            }

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
