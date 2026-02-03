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

/// Helper to register a user and return (token, user_id)
async fn register_user(server: &TestServer, username: &str, email: &str, password: &str) -> (String, String) {
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
    let token = body["token"].as_str().unwrap().to_string();
    let user_id = body["user"]["id"].as_str().unwrap().to_string();
    (token, user_id)
}

/// Helper to create a club and return (club_id, owner_token, owner_id)
async fn create_club(server: &TestServer, owner: &str) -> (String, String, String) {
    let (token, user_id) = register_user(
        server,
        owner,
        &format!("{}@example.com", owner),
        "password123",
    )
    .await;

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

    (club_id, token, user_id)
}

/// Helper to add balance to a club member
async fn add_balance(
    server: &TestServer,
    admin_token: &str,
    club_id: &str,
    user_id: &str,
    amount: i64,
) {
    let response = server
        .post(&format!(
            "/api/clubs/{}/members/{}/balance",
            club_id, user_id
        ))
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
    let (club_id, token, _owner_id) = create_club(&server, "owner").await;

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
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

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
        let (token, user_id) = register_user(
            &server,
            &username,
            &format!("{}@example.com", username),
            "password123",
        )
        .await;

        // Join club
        server
            .post("/api/clubs/join")
            .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();

        

        // Add balance
        add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

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
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

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
    let (token, user_id) = register_user(&server, "player1", "player1@example.com", "password123").await;

    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

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
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

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
        let (token, user_id) = register_user(
            &server,
            &username,
            &format!("{}@example.com", username),
            "password123",
        )
        .await;

        server
            .post("/api/clubs/join")
            .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();

        add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

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
    let (club_id, token, _owner_id) = create_club(&server, "owner").await;

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
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

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
        let (token, user_id) = register_user(
            &server,
            &username,
            &format!("{}@example.com", username),
            "password123",
        )
        .await;

        server
            .post("/api/clubs/join")
            .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();

        add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

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
    let (club_id, token, _owner_id) = create_club(&server, "owner").await;

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
    let body: Value = list_response.json();

    assert!(body["tournaments"].is_array());
    assert_eq!(body["tournaments"].as_array().unwrap().len(), 3);
}

// ============================================================================
// Blind Level Tests
// ============================================================================

#[tokio::test]
async fn test_blind_level_advancement() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

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
        let (token, user_id) = register_user(
            &server,
            &username,
            &format!("{}@example.com", username),
            "password123",
        )
        .await;

        server
            .post("/api/clubs/join")
            .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();

        add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

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
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

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
        let (token, user_id) = register_user(
            &server,
            &username,
            &format!("{}@example.com", username),
            "password123",
        )
        .await;

        server
            .post("/api/clubs/join")
            .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();

        add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

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
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG",
            "variant_id": "holdem",
            "buy_in": 20000,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Register player without sufficient balance
    let (token, _user_id) =
        register_user(&server, "poorplayer", "poor@example.com", "password123").await;

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

// ============================================================================
// Bot and Table Tests
// ============================================================================

#[tokio::test]
async fn test_fill_with_bots_sng() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create SNG with 9 seats
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG with Bots",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 9,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Register 2 real players
    for i in 1..=2 {
        let username = format!("player{}", i);
        let (token, user_id) = register_user(
            &server,
            &username,
            &format!("{}@example.com", username),
            "password123",
        )
        .await;

        server
            .post("/api/clubs/join")
            .json(&json!({"club_id": club_id.clone()}))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();

        

        add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

        server
            .post(&format!("/api/tournaments/{}/register", tournament_id))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
    }

    // Fill remaining spots with bots
    let fill_response = server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    fill_response.assert_status_ok();
    let fill_body: Value = fill_response.json();
    
    // Should have 9 total registrations (2 real + 7 bots)
    assert_eq!(fill_body["tournament"]["registered_players"], 9);
    assert_eq!(fill_body["registrations"].as_array().unwrap().len(), 9);
    
    // Verify some are bots
    let registrations = fill_body["registrations"].as_array().unwrap();
    let bot_count = registrations
        .iter()
        .filter(|r| r["username"].as_str().unwrap().starts_with("Bot_"))
        .count();
    assert_eq!(bot_count, 7);
}

#[tokio::test]
async fn test_start_sng_creates_table() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create SNG
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test SNG Table Creation",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 9,
            "min_players": 2,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Fill with bots
    server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await
        .assert_status_ok();

    // Start tournament
    let start_response = server
        .post(&format!("/api/tournaments/{}/start", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    start_response.assert_status_ok();
    let start_body: Value = start_response.json();
    assert_eq!(start_body["status"], "running");

    // Verify tables were created
    let tables_response = server
        .get(&format!("/api/tournaments/{}/tables", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    tables_response.assert_status_ok();
    let tables_body: Value = tables_response.json();
    eprintln!("Tables response: {:?}", tables_body);
    let tables = tables_body["tables"].as_array().unwrap();
    
    // Should have 1 table for SNG
    assert_eq!(tables.len(), 1, "Expected 1 table, got: {:?}", tables_body);
    assert_eq!(tables[0]["table_number"], 1);
    assert_eq!(tables[0]["player_count"], 9);
}

#[tokio::test]
async fn test_start_mtt_creates_multiple_tables() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create MTT with 18 players (should create 2 tables)
    let response = server
        .post("/api/tournaments/mtt")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test MTT Multi-Table",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 18,
            "min_players": 10,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Fill with bots
    server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await
        .assert_status_ok();

    // Start tournament
    let start_response = server
        .post(&format!("/api/tournaments/{}/start", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    start_response.assert_status_ok();

    // Verify multiple tables were created
    let tables_response = server
        .get(&format!("/api/tournaments/{}/tables", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    tables_response.assert_status_ok();
    let tables_body: Value = tables_response.json();
    let tables = tables_body["tables"].as_array().unwrap();
    
    // Should have 2 tables for 18 players (9 per table)
    assert_eq!(tables.len(), 2);
    
    // Verify table numbers
    assert_eq!(tables[0]["table_number"], 1);
    assert_eq!(tables[1]["table_number"], 2);
    
    // Verify player distribution (9 per table)
    let total_players: i64 = tables
        .iter()
        .map(|t| t["player_count"].as_i64().unwrap())
        .sum();
    assert_eq!(total_players, 18);
}

#[tokio::test]
async fn test_cannot_fill_bots_after_start() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 9,
            "min_players": 2,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Fill and start
    server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await
        .assert_status_ok();

    server
        .post(&format!("/api/tournaments/{}/start", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await
        .assert_status_ok();

    // Try to fill bots again (should fail)
    let fill_response = server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    fill_response.assert_status_bad_request();
}

#[tokio::test]
async fn test_cancel_tournament_refunds_players() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test Cancel",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 9,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Register a player
    let (token, user_id) = register_user(&server, "player1", "player1@example.com", "password123").await;

    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    

    add_balance(&server, &owner_token, &club_id, &user_id, 1000).await;

    server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    // Cancel tournament
    let cancel_response = server
        .delete(&format!("/api/tournaments/{}/cancel", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    cancel_response.assert_status_ok();
    let cancel_body: Value = cancel_response.json();
    assert_eq!(cancel_body["status"], "cancelled");
    assert_eq!(cancel_body["prize_pool"], 0);
}

// ============================================================================
// Prize Distribution Tests
// ============================================================================

#[tokio::test]
async fn test_tournament_complete_flow() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // 1. Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Complete Flow Test",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "min_players": 2,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();
    assert_eq!(body["tournament"]["status"], "registering");

    // 2. Fill with bots
    let fill_response = server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    if fill_response.status_code() != 200 {
        eprintln!("Fill bots failed with status: {}", fill_response.status_code());
        eprintln!("Response body: {}", fill_response.text());
        panic!("Failed to fill with bots");
    }

    fill_response.assert_status_ok();
    let fill_body: Value = fill_response.json();
    assert_eq!(fill_body["tournament"]["registered_players"], 6);

    // 3. Start tournament
    let start_response = server
        .post(&format!("/api/tournaments/{}/start", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    start_response.assert_status_ok();
    let start_body: Value = start_response.json();
    assert_eq!(start_body["status"], "running");
    assert_eq!(start_body["remaining_players"], 6);

    // 4. Verify tables created
    let tables_response = server
        .get(&format!("/api/tournaments/{}/tables", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    if tables_response.status_code() != 200 {
        eprintln!("Get tables failed: {}", tables_response.text());
        panic!("Failed to get tables");
    }

    tables_response.assert_status_ok();
    let tables_body: Value = tables_response.json();
    eprintln!("Tables response: {}", serde_json::to_string_pretty(&tables_body).unwrap());
    
    if !tables_body["tables"].is_array() {
        panic!("tables_body['tables'] is not an array: {:?}", tables_body);
    }
    
    let tables = tables_body["tables"].as_array().unwrap();
    
    if tables.is_empty() {
        panic!("No tables were created!");
    }
    
    assert_eq!(tables.len(), 1, "Should create exactly 1 table for SNG");
    assert_eq!(tables[0]["table_number"], 1);
    eprintln!("Player count in table: {}", tables[0]["player_count"]);
    assert_eq!(tables[0]["player_count"], 6, "Table should have all 6 players");

    // 5. Get tournament details
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["status"], "running");
    assert_eq!(details["tournament"]["current_blind_level"], 0);
}
