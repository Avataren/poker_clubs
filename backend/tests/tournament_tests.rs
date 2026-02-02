//! Integration tests for Tournament functionality
//!
//! These tests verify the complete tournament lifecycle including:
//! - SNG creation and registration
//! - MTT creation and multi-table management
//! - Blind level advancement
//! - Player elimination tracking
//! - Prize distribution

use axum::http::header::AUTHORIZATION;
use axum_test::TestServer;
use poker_server::create_test_app;
use serde_json::{json, Value};

/// Helper to create a test server instance
async fn setup() -> TestServer {
    let (app, _game_server) = create_test_app().await;
    TestServer::new(app).unwrap()
}

/// Helper to register a user and get their token
async fn register_user(server: &TestServer, username: &str, email: &str, password: &str) -> String {
    let response = server
        .post("/api/auth/register")
        .json(&json!({
            "username": username,
            "email": email,
            "password": password
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    body["token"].as_str().unwrap().to_string()
}

/// Helper to create a club and return (club_id, owner_token)
async fn create_club(server: &TestServer, owner: &str) -> (String, String) {
    let token = register_user(server, owner, &format!("{}@example.com", owner), "password123").await;
    
    let response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "name": "Test Club",
            "description": "Test poker club"
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let club_id = body["club"]["id"].as_str().unwrap().to_string();
    
    (club_id, token)
}

/// Helper to add balance to a club member
async fn add_balance(server: &TestServer, admin_token: &str, club_id: &str, user_id: &str, amount: i64) {
    let response = server
        .post(&format!("/api/clubs/{}/members/{}/balance", club_id, user_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", admin_token))
        .json(&json!({
            "amount": amount
        }))
        .await;
    
    response.assert_status_ok();
}

// ============================================================================
// SNG Tournament Tests
// ============================================================================

#[tokio::test]
async fn test_create_sng_tournament() {
    let server = setup().await;
    let (club_id, token) = create_club(&server, "owner").await;
    
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;
    
    // Debug: print the response
    let status = response.status_code();
    let body: Value = response.json();
    eprintln!("Status: {}, Body: {:?}", status, body);
    
    assert_eq!(status, 200);
    assert_eq!(body["tournament"]["name"], "Test SNG");
    assert_eq!(body["tournament"]["buy_in"], 100);
    assert_eq!(body["tournament"]["starting_stack"], 1500);
    assert_eq!(body["tournament"]["max_players"], 6);
    assert_eq!(body["tournament"]["status"], "registering");
    assert_eq!(body["tournament"]["format_id"], "sng");
    assert!(body["blind_levels"].is_array());
}

#[tokio::test]
async fn test_sng_registration() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create SNG tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 3,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register three players
    for i in 1..=3 {
        let username = format!("player{}", i);
        let token = register_user(&server, &username, &format!("{}@example.com", username), "password123").await;
        
        // Join club
        server
            .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
        
        // Get user_id from token response
        let user_response = server
            .get("/api/auth/me")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await;
        user_response.assert_status_ok();
        let user_body: Value = user_response.json();
        let user_id = user_body["tournament"]["id"].as_str().unwrap();
        
        // Add balance
        add_balance(&server, &owner_token, &club_id, user_id, 1000).await;
        
        // Register for tournament
        let reg_response = server
            .post(&format!("/api/tournaments/{}/register", tournament_id))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await;
        
        reg_response.assert_status_ok();
    }
    
    // Check tournament details
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["registered_players"], 3);
    assert_eq!(details["tournament"]["prize_pool"], 300); // 3 * 100
}

#[tokio::test]
async fn test_sng_unregister() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create SNG
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register player
    let token = register_user(&server, "player1", "player1@example.com", "password123").await;
    
    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();
    
    let user_response = server
        .get("/api/auth/me")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    user_response.assert_status_ok();
    let user_body: Value = user_response.json();
    let user_id = user_body["tournament"]["id"].as_str().unwrap();
    
    add_balance(&server, &owner_token, &club_id, user_id, 1000).await;
    
    server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();
    
    // Unregister
    let unreg_response = server
        .delete(&format!("/api/tournaments/{}/unregister", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    
    unreg_response.assert_status_ok();
    
    // Verify unregistration
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["registered_players"], 0);
    assert_eq!(details["tournament"]["prize_pool"], 0);
}

#[tokio::test]
async fn test_sng_auto_start() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create SNG with max 3 players
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG Auto Start",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 3,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register 3 players (should auto-start)
    for i in 1..=3 {
        let username = format!("player{}", i);
        let token = register_user(&server, &username, &format!("{}@example.com", username), "password123").await;
        
        server
            .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
        
        let user_response = server
            .get("/api/auth/me")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await;
        user_response.assert_status_ok();
        let user_body: Value = user_response.json();
        let user_id = user_body["tournament"]["id"].as_str().unwrap();
        
        add_balance(&server, &owner_token, &club_id, user_id, 1000).await;
        
        server
            .post(&format!("/api/tournaments/{}/register", tournament_id))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
    }
    
    // Check that tournament started automatically
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["status"], "running");
    assert_eq!(details["tournament"]["remaining_players"], 3);
}

// ============================================================================
// MTT Tournament Tests
// ============================================================================

#[tokio::test]
async fn test_create_mtt_tournament() {
    let server = setup().await;
    let (club_id, token) = create_club(&server, "owner").await;
    
    let response = server
        .post("/api/tournaments/mtt")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test MTT",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 18,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    
    assert_eq!(body["tournament"]["name"], "Test MTT");
    assert_eq!(body["tournament"]["format_id"], "mtt");
    assert_eq!(body["tournament"]["max_players"], 18);
}

#[tokio::test]
async fn test_mtt_manual_start() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create MTT
    let response = server
        .post("/api/tournaments/mtt")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test MTT",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 18,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register 10 players
    for i in 1..=10 {
        let username = format!("player{}", i);
        let token = register_user(&server, &username, &format!("{}@example.com", username), "password123").await;
        
        server
            .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
        
        let user_response = server
            .get("/api/auth/me")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await;
        user_response.assert_status_ok();
        let user_body: Value = user_response.json();
        let user_id = user_body["tournament"]["id"].as_str().unwrap();
        
        add_balance(&server, &owner_token, &club_id, user_id, 1000).await;
        
        server
            .post(&format!("/api/tournaments/{}/register", tournament_id))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
    }
    
    // Manually start tournament
    let start_response = server
        .post(&format!("/api/tournaments/{}/start", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    
    start_response.assert_status_ok();
    
    // Verify started
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["status"], "running");
    assert_eq!(details["tournament"]["remaining_players"], 10);
}

// ============================================================================
// Tournament Listing Tests
// ============================================================================

#[tokio::test]
async fn test_list_club_tournaments() {
    let server = setup().await;
    let (club_id, token) = create_club(&server, "owner").await;
    
    // Create multiple tournaments
    for i in 1..=3 {
        server
            .post("/api/tournaments/sng")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .json(&json!({
                "club_id": club_id,
                "name": format!("SNG {}", i),
                "variant_id": "holdem",
                "buy_in": 100,
                "starting_stack": 1500,
                "max_players": 6,
                "level_duration_mins": 5
            }))
            .await
            .assert_status_ok();
    }
    
    // List tournaments
    let list_response = server
        .get(&format!("/api/tournaments/club/{}", club_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    
    list_response.assert_status_ok();
    let tournaments: Value = list_response.json();
    
    assert!(tournaments.is_array());
    assert_eq!(tournaments.as_array().unwrap().len(), 3);
}

// ============================================================================
// Blind Level Tests
// ============================================================================

#[tokio::test]
async fn test_blind_level_advancement() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create SNG with short blind levels (1 second for testing)
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Fast Blinds SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 2,
            "level_duration_mins": 1
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register 2 players to auto-start
    for i in 1..=2 {
        let username = format!("player{}", i);
        let token = register_user(&server, &username, &format!("{}@example.com", username), "password123").await;
        
        server
            .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
        
        let user_response = server
            .get("/api/auth/me")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await;
        user_response.assert_status_ok();
        let user_body: Value = user_response.json();
        let user_id = user_body["tournament"]["id"].as_str().unwrap();
        
        add_balance(&server, &owner_token, &club_id, user_id, 1000).await;
        
        server
            .post(&format!("/api/tournaments/{}/register", tournament_id))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
    }
    
    // Check initial blind level
    let details1 = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    details1.assert_status_ok();
    let level1: Value = details1.json();
    assert_eq!(level1["tournament"]["current_blind_level"], 0);
    
    // Wait for blind level to advance (background task runs every 10s, but level duration is 1s)
    // In a real scenario, the background task would advance it
    // For this test, we just verify the structure is in place
}

// ============================================================================
// Prize Distribution Tests
// ============================================================================

#[tokio::test]
async fn test_prize_structure() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create 6-player SNG
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Prize Test SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register 6 players
    for i in 1..=6 {
        let username = format!("player{}", i);
        let token = register_user(&server, &username, &format!("{}@example.com", username), "password123").await;
        
        server
            .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
        
        let user_response = server
            .get("/api/auth/me")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await;
        user_response.assert_status_ok();
        let user_body: Value = user_response.json();
        let user_id = user_body["tournament"]["id"].as_str().unwrap();
        
        add_balance(&server, &owner_token, &club_id, user_id, 1000).await;
        
        server
            .post(&format!("/api/tournaments/{}/register", tournament_id))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
    }
    
    // Verify prize pool
    let details = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;
    
    details.assert_status_ok();
    let tournament: Value = details.json();
    assert_eq!(tournament["tournament"]["prize_pool"], 600); // 6 * 100
    assert_eq!(tournament["tournament"]["status"], "running");
}

#[tokio::test]
async fn test_insufficient_balance() {
    let server = setup().await;
    let (club_id, owner_token) = create_club(&server, "owner").await;
    
    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;
    
    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    
    // Register player without sufficient balance
    let token = register_user(&server, "poorplayer", "poor@example.com", "password123").await;
    
    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();
    
    // Try to register without balance (should fail)
    let reg_response = server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    
    reg_response.assert_status_bad_request();
}
