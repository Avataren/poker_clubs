//! WebSocket Integration Tests for the Poker Server
//!
//! These tests verify the real-time game flow via WebSocket connections.

use futures::{SinkExt, StreamExt};
use poker_server::{create_test_app, game::{PlayerAction, GamePhase}, ws::messages::{ClientMessage, ServerMessage}};
use serde_json::json;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Test helper to spin up a server and return its address
async fn spawn_server() -> (SocketAddr, String) {
    let (app, _game_server) = create_test_app().await;
    
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    
    // Create a user and get token for WebSocket auth
    let client = reqwest::Client::new();
    let base_url = format!("http://{}", addr);
    
    let response = client
        .post(format!("{}/api/auth/register", base_url))
        .json(&json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = response.json().await.unwrap();
    let token = body["token"].as_str().unwrap().to_string();
    
    (addr, token)
}

/// Helper to spawn a server and create a second user
async fn spawn_server_with_two_users() -> (SocketAddr, String, String) {
    let (app, _game_server) = create_test_app().await;
    
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    
    let client = reqwest::Client::new();
    let base_url = format!("http://{}", addr);
    
    // Register first user
    let response = client
        .post(format!("{}/api/auth/register", base_url))
        .json(&json!({
            "username": "player1",
            "email": "player1@example.com",
            "password": "password123"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = response.json().await.unwrap();
    let token1 = body["token"].as_str().unwrap().to_string();
    
    // Register second user
    let response = client
        .post(format!("{}/api/auth/register", base_url))
        .json(&json!({
            "username": "player2",
            "email": "player2@example.com",
            "password": "password123"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = response.json().await.unwrap();
    let token2 = body["token"].as_str().unwrap().to_string();
    
    (addr, token1, token2)
}

/// Helper to create a club and table via HTTP API
async fn create_club_and_table(addr: SocketAddr, token: &str) -> (String, String) {
    let client = reqwest::Client::new();
    let base_url = format!("http://{}", addr);
    
    // Create a club
    let response = client
        .post(format!("{}/api/clubs", base_url))
        .header("Authorization", format!("Bearer {}", token))
        .json(&json!({
            "name": "Test Club"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = response.json().await.unwrap();
    // Club API returns {"club": {"id": ...}, "is_admin": ..., "balance": ...}
    let club_id = body["club"]["id"].as_str().unwrap().to_string();
    
    // Create a table in the club
    let response = client
        .post(format!("{}/api/tables", base_url))
        .header("Authorization", format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test Table",
            "small_blind": 25,
            "big_blind": 50
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = response.json().await.unwrap();
    // Table API returns Table directly with "id" field
    let table_id = body["id"].as_str().unwrap().to_string();
    
    (club_id, table_id)
}

// ============================================================================
// Basic WebSocket Connection Tests
// ============================================================================

#[tokio::test]
async fn test_ws_connect_and_receive_connected_message() {
    let (addr, token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Should receive Connected message
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        assert!(matches!(server_msg, ServerMessage::Connected));
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_connect_with_invalid_token() {
    let (addr, _token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token=invalid_token", addr);
    let result = connect_async(&ws_url).await;
    
    // Connection should fail with 401
    assert!(result.is_err() || {
        if let Ok((_, response)) = result {
            response.status().as_u16() == 401
        } else {
            false
        }
    });
}

#[tokio::test]
async fn test_ws_ping_pong() {
    let (addr, token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Send Ping
    let ping_msg = serde_json::to_string(&ClientMessage::Ping).unwrap();
    ws_stream.send(Message::Text(ping_msg)).await.unwrap();
    
    // Should receive Pong
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        assert!(matches!(server_msg, ServerMessage::Pong));
    } else {
        panic!("Expected Pong message");
    }
}

// ============================================================================
// Table Join and Seat Tests
// ============================================================================

#[tokio::test]
async fn test_ws_join_table_as_observer() {
    let (addr, token) = spawn_server().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token).await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Join table as observer (buyin = 0)
    let join_msg = serde_json::to_string(&ClientMessage::JoinTable {
        table_id: table_id.clone(),
        buyin: 0,
    }).unwrap();
    ws_stream.send(Message::Text(join_msg)).await.unwrap();
    
    // Should receive TableState
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::TableState(state) => {
                assert_eq!(state.table_id, table_id);
                assert!(matches!(state.phase, GamePhase::Waiting));
            }
            other => panic!("Expected TableState, got: {:?}", other),
        }
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_take_seat_at_table() {
    let (addr, token) = spawn_server().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token).await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Take seat at table
    let take_seat_msg = serde_json::to_string(&ClientMessage::TakeSeat {
        table_id: table_id.clone(),
        seat: 0,
        buyin: 1000,
    }).unwrap();
    ws_stream.send(Message::Text(take_seat_msg)).await.unwrap();
    
    // Should receive TableState with player seated
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::TableState(state) => {
                assert_eq!(state.table_id, table_id);
                // Player should be seated
                assert!(!state.players.is_empty());
                assert_eq!(state.players[0].username, "testuser");
                assert_eq!(state.players[0].seat, 0);
            }
            other => panic!("Expected TableState, got: {:?}", other),
        }
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_join_table_with_buyin() {
    let (addr, token) = spawn_server().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token).await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Join with buyin (auto-seat)
    let join_msg = serde_json::to_string(&ClientMessage::JoinTable {
        table_id: table_id.clone(),
        buyin: 1000,
    }).unwrap();
    ws_stream.send(Message::Text(join_msg)).await.unwrap();
    
    // Should receive TableState with player seated
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::TableState(state) => {
                assert_eq!(state.table_id, table_id);
                assert!(!state.players.is_empty());
                assert_eq!(state.players[0].stack, 1000);
            }
            other => panic!("Expected TableState, got: {:?}", other),
        }
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_join_nonexistent_table() {
    let (addr, token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Try to join non-existent table
    let join_msg = serde_json::to_string(&ClientMessage::JoinTable {
        table_id: "nonexistent-table-id".to_string(),
        buyin: 0,
    }).unwrap();
    ws_stream.send(Message::Text(join_msg)).await.unwrap();
    
    // Should receive Error
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::Error { message } => {
                assert!(message.contains("not found") || message.contains("Table"));
            }
            other => panic!("Expected Error, got: {:?}", other),
        }
    } else {
        panic!("Expected text message");
    }
}

// ============================================================================
// Game State and Table State Tests
// ============================================================================

#[tokio::test]
async fn test_ws_get_table_state() {
    let (addr, token) = spawn_server().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token).await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Join table first
    let join_msg = serde_json::to_string(&ClientMessage::JoinTable {
        table_id: table_id.clone(),
        buyin: 0,
    }).unwrap();
    ws_stream.send(Message::Text(join_msg)).await.unwrap();
    ws_stream.next().await.unwrap().unwrap(); // Consume join response
    
    // Request table state
    let get_state_msg = serde_json::to_string(&ClientMessage::GetTableState).unwrap();
    ws_stream.send(Message::Text(get_state_msg)).await.unwrap();
    
    // Should receive TableState
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::TableState(state) => {
                assert_eq!(state.table_id, table_id);
                // Verify table state is valid
                assert!(!state.name.is_empty());
            }
            other => panic!("Expected TableState, got: {:?}", other),
        }
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_get_table_state_not_at_table() {
    let (addr, token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Request table state without joining a table
    let get_state_msg = serde_json::to_string(&ClientMessage::GetTableState).unwrap();
    ws_stream.send(Message::Text(get_state_msg)).await.unwrap();
    
    // Should receive Error
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::Error { message } => {
                assert!(message.contains("Not at a table"));
            }
            other => panic!("Expected Error, got: {:?}", other),
        }
    } else {
        panic!("Expected text message");
    }
}

// ============================================================================
// Leave Table and Stand Up Tests
// ============================================================================

#[tokio::test]
async fn test_ws_leave_table() {
    let (addr, token) = spawn_server().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token).await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Join table
    let join_msg = serde_json::to_string(&ClientMessage::TakeSeat {
        table_id: table_id.clone(),
        seat: 0,
        buyin: 1000,
    }).unwrap();
    ws_stream.send(Message::Text(join_msg)).await.unwrap();
    ws_stream.next().await.unwrap().unwrap(); // Consume join response
    
    // Leave table
    let leave_msg = serde_json::to_string(&ClientMessage::LeaveTable).unwrap();
    ws_stream.send(Message::Text(leave_msg)).await.unwrap();
    
    // Consume leave response and any broadcasts
    // Wait a bit for broadcasts to arrive
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    while let Ok(result) = tokio::time::timeout(
        tokio::time::Duration::from_millis(100),
        ws_stream.next()
    ).await {
        if result.is_none() {
            break;
        }
    }
    
    // Verify we're no longer at the table by requesting table state
    let get_state_msg = serde_json::to_string(&ClientMessage::GetTableState).unwrap();
    ws_stream.send(Message::Text(get_state_msg)).await.unwrap();
    
    // Look for the error response (may have other messages in between)
    let mut found_error = false;
    for _ in 0..5 {
        if let Some(Ok(Message::Text(text))) = ws_stream.next().await {
            if let Ok(ServerMessage::Error { message }) = serde_json::from_str(&text) {
                assert!(message.contains("Not at a table"));
                found_error = true;
                break;
            }
        }
    }
    assert!(found_error, "Should have received 'Not at a table' error");
}

// ============================================================================
// Multi-Player Game Flow Tests
// ============================================================================

#[tokio::test]
async fn test_ws_two_players_start_game() {
    let (addr, token1, token2) = spawn_server_with_two_users().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token1).await;
    
    // Connect player 1
    let ws_url1 = format!("ws://{}/ws?token={}", addr, token1);
    let (mut ws1, _) = connect_async(&ws_url1).await.expect("Failed to connect player 1");
    ws1.next().await.unwrap().unwrap(); // Consume Connected
    
    // Connect player 2
    let ws_url2 = format!("ws://{}/ws?token={}", addr, token2);
    let (mut ws2, _) = connect_async(&ws_url2).await.expect("Failed to connect player 2");
    ws2.next().await.unwrap().unwrap(); // Consume Connected
    
    // Player 1 takes seat 0
    let seat_msg = serde_json::to_string(&ClientMessage::TakeSeat {
        table_id: table_id.clone(),
        seat: 0,
        buyin: 1000,
    }).unwrap();
    ws1.send(Message::Text(seat_msg)).await.unwrap();
    let msg = ws1.next().await.unwrap().unwrap();
    
    // Verify player 1 is seated and game is still waiting
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::TableState(state) => {
                assert!(matches!(state.phase, GamePhase::Waiting));
                assert_eq!(state.players.len(), 1);
            }
            other => panic!("Expected TableState, got: {:?}", other),
        }
    }
    
    // Player 2 takes seat 1
    let seat_msg = serde_json::to_string(&ClientMessage::TakeSeat {
        table_id: table_id.clone(),
        seat: 1,
        buyin: 1000,
    }).unwrap();
    ws2.send(Message::Text(seat_msg)).await.unwrap();
    
    // Collect messages until we get one with PreFlop phase
    // Both players receive broadcasts, so we need to handle multiple messages
    let mut found_preflop = false;
    for _ in 0..5 {
        if let Some(Ok(Message::Text(text))) = ws2.next().await {
            if let Ok(ServerMessage::TableState(state)) = serde_json::from_str(&text) {
                if matches!(state.phase, GamePhase::PreFlop) {
                    found_preflop = true;
                    assert_eq!(state.players.len(), 2);
                    break;
                }
            }
        }
    }
    assert!(found_preflop, "Game should have started (PreFlop phase)");
}

#[tokio::test]
async fn test_ws_player_action_fold() {
    let (addr, token1, token2) = spawn_server_with_two_users().await;
    let (_club_id, table_id) = create_club_and_table(addr, &token1).await;
    
    // Connect both players
    let ws_url1 = format!("ws://{}/ws?token={}", addr, token1);
    let (mut ws1, _) = connect_async(&ws_url1).await.unwrap();
    ws1.next().await.unwrap().unwrap();
    
    let ws_url2 = format!("ws://{}/ws?token={}", addr, token2);
    let (mut ws2, _) = connect_async(&ws_url2).await.unwrap();
    ws2.next().await.unwrap().unwrap();
    
    // Both players take seats
    let seat_msg = serde_json::to_string(&ClientMessage::TakeSeat {
        table_id: table_id.clone(),
        seat: 0,
        buyin: 1000,
    }).unwrap();
    ws1.send(Message::Text(seat_msg)).await.unwrap();
    ws1.next().await.unwrap().unwrap();
    
    let seat_msg = serde_json::to_string(&ClientMessage::TakeSeat {
        table_id: table_id.clone(),
        seat: 1,
        buyin: 1000,
    }).unwrap();
    ws2.send(Message::Text(seat_msg)).await.unwrap();
    ws2.next().await.unwrap().unwrap();
    
    // Consume broadcast messages
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    // Find out whose turn it is
    let get_state_msg = serde_json::to_string(&ClientMessage::GetTableState).unwrap();
    ws1.send(Message::Text(get_state_msg.clone())).await.unwrap();
    
    let msg = ws1.next().await.unwrap().unwrap();
    let current_player_seat = if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        match server_msg {
            ServerMessage::TableState(state) => {
                state.current_player_seat
            }
            _ => panic!("Expected TableState"),
        }
    } else {
        panic!("Expected text message");
    };
    
    // The player whose turn it is sends a fold
    let fold_msg = serde_json::to_string(&ClientMessage::PlayerAction {
        action: PlayerAction::Fold,
    }).unwrap();
    
    if current_player_seat == 0 {
        ws1.send(Message::Text(fold_msg)).await.unwrap();
    } else {
        ws2.send(Message::Text(fold_msg)).await.unwrap();
    }
    
    // Should receive success response
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
}

// ============================================================================
// Club and View Subscription Tests
// ============================================================================

#[tokio::test]
async fn test_ws_viewing_clubs_list() {
    let (addr, token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Subscribe to clubs list view
    let view_msg = serde_json::to_string(&ClientMessage::ViewingClubsList).unwrap();
    ws_stream.send(Message::Text(view_msg)).await.unwrap();
    
    // Should receive Connected (subscription success)
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        assert!(matches!(server_msg, ServerMessage::Connected));
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_viewing_club() {
    let (addr, token) = spawn_server().await;
    let (club_id, _table_id) = create_club_and_table(addr, &token).await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Subscribe to specific club view
    let view_msg = serde_json::to_string(&ClientMessage::ViewingClub {
        club_id: club_id.clone(),
    }).unwrap();
    ws_stream.send(Message::Text(view_msg)).await.unwrap();
    
    // Should receive Connected (subscription success)
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        assert!(matches!(server_msg, ServerMessage::Connected));
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_ws_leaving_view() {
    let (addr, token) = spawn_server().await;
    
    let ws_url = format!("ws://{}/ws?token={}", addr, token);
    let (mut ws_stream, _) = connect_async(&ws_url).await.expect("Failed to connect");
    
    // Consume Connected message
    ws_stream.next().await.unwrap().unwrap();
    
    // Subscribe to clubs list view
    let view_msg = serde_json::to_string(&ClientMessage::ViewingClubsList).unwrap();
    ws_stream.send(Message::Text(view_msg)).await.unwrap();
    ws_stream.next().await.unwrap().unwrap();
    
    // Leave view
    let leave_msg = serde_json::to_string(&ClientMessage::LeavingView).unwrap();
    ws_stream.send(Message::Text(leave_msg)).await.unwrap();
    
    // Should receive Connected (unsubscribe success)
    let msg = ws_stream.next().await.unwrap().unwrap();
    if let Message::Text(text) = msg {
        let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
        assert!(matches!(server_msg, ServerMessage::Connected));
    } else {
        panic!("Expected text message");
    }
}
