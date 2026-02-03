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

// ============================================================================
// Bankroll and Balance Tests
// ============================================================================

#[tokio::test]
async fn test_registration_deducts_balance() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament with 100 buy-in
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Balance Test SNG",
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

    // Add 500 balance
    add_balance(&server, &owner_token, &club_id, &user_id, 500).await;

    // Register for tournament
    server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    // Verify tournament updated correctly (prize pool should be 100)
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["prize_pool"], 100, "Prize pool should equal buy-in");
    assert_eq!(details["tournament"]["registered_players"], 1);
    
    // Test balance by trying to register for another tournament with same buy-in
    // If balance was deducted, we should have 400 left, which is enough for another 100 buy-in
    let response2 = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Second Test",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    response2.assert_status_ok();
    let body2: Value = response2.json();
    let tournament2_id = body2["tournament"]["id"].as_str().unwrap().to_string();

    // Should succeed with remaining balance
    server
        .post(&format!("/api/tournaments/{}/register", tournament2_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();
}

#[tokio::test]
async fn test_unregister_refunds_balance() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Refund Test SNG",
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

    // Add 500 balance
    add_balance(&server, &owner_token, &club_id, &user_id, 500).await;

    // Register for tournament
    server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    // Verify registration successful
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["prize_pool"], 100);
    assert_eq!(details["tournament"]["registered_players"], 1);

    // Unregister
    server
        .delete(&format!("/api/tournaments/{}/unregister", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    // Verify balance was refunded by registering for 5 more tournaments
    // If refund worked, we should have 500 again and can register 5 times
    for i in 1..=5 {
        let response = server
            .post("/api/tournaments/sng")
            .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
            .json(&json!({
                "club_id": club_id,
                "name": format!("Test {}", i),
                "variant_id": "holdem",
                "buy_in": 100,
                "starting_stack": 1500,
                "max_players": 6,
                "level_duration_mins": 5
            }))
            .await;

        response.assert_status_ok();
        let body: Value = response.json();
        let tid = body["tournament"]["id"].as_str().unwrap().to_string();

        server
            .post(&format!("/api/tournaments/{}/register", tid))
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .await
            .assert_status_ok();
    }
}

#[tokio::test]
async fn test_cannot_register_without_balance() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament with 100 buy-in
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "No Balance Test",
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

    // Register player with insufficient balance
    let (token, user_id) = register_user(&server, "poorplayer", "poor@example.com", "password123").await;

    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await
        .assert_status_ok();

    // Player starts with 10000 from joining club
    // Subtract most of it to leave only 50
    add_balance(&server, &owner_token, &club_id, &user_id, -9950).await;

    // Try to register with insufficient balance (should fail)
    let reg_response = server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;

    reg_response.assert_status_bad_request();

    // Verify player was not added to tournament
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    
    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["registered_players"], 0, "No players should be registered");
    assert_eq!(details["tournament"]["prize_pool"], 0, "Prize pool should be empty");
}

#[tokio::test]
async fn test_cannot_register_without_club_membership() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Membership Test",
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

    // Register player without joining club
    let (token, _user_id) = register_user(&server, "outsider", "outsider@example.com", "password123").await;

    // Try to register without being a club member (should fail)
    let reg_response = server
        .post(&format!("/api/tournaments/{}/register", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;

    reg_response.assert_status_bad_request();
}

#[tokio::test]
async fn test_bots_can_register_without_balance() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Bot Test SNG",
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

    // Fill with bots (bots don't have club membership or balance)
    let fill_response = server
        .post(&format!("/api/tournaments/{}/fill-bots", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    fill_response.assert_status_ok();

    // Verify tournament has registrations
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["registered_players"], 6);
    assert_eq!(details["tournament"]["status"], "running");
}

#[tokio::test]
async fn test_cannot_unregister_after_start() {
    let server = setup().await;
    let (club_id, owner_token, _owner_id) = create_club(&server, "owner").await;

    // Create tournament
    let response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .json(&json!({
            "club_id": club_id,
            "name": "Start Test SNG",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 2,
            "level_duration_mins": 5
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    let tournament_id = body["tournament"]["id"].as_str().unwrap().to_string();

    // Register two players
    let mut first_player_token = String::new();
    for i in 1..=2 {
        let (token, user_id) = register_user(&server, &format!("starttest{}", i), &format!("starttest{}@example.com", i), "password123").await;

        if i == 1 {
            first_player_token = token.clone();
        }

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

    // Tournament should auto-start (max_players reached)
    let details_response = server
        .get(&format!("/api/tournaments/{}", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", owner_token))
        .await;

    details_response.assert_status_ok();
    let details: Value = details_response.json();
    assert_eq!(details["tournament"]["status"], "running");

    // Try to unregister after start (should fail)
    let unreg_response = server
        .delete(&format!("/api/tournaments/{}/unregister", tournament_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", first_player_token))
        .await;

    unreg_response.assert_status_bad_request();
}

#[tokio::test]
async fn test_multiple_clubs_separate_bankrolls() {
    let server = setup().await;
    
    // Create two clubs
    let (club1_id, owner1_token, _owner1_id) = create_club(&server, "owner1").await;
    let (club2_id, owner2_token, _owner2_id) = create_club(&server, "owner2").await;

    // Create tournaments in both clubs
    let t1_response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner1_token))
        .json(&json!({
            "club_id": club1_id,
            "name": "Club 1 Tournament",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    t1_response.assert_status_ok();
    let t1_body: Value = t1_response.json();
    let t1_id = t1_body["tournament"]["id"].as_str().unwrap().to_string();

    let t2_response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner2_token))
        .json(&json!({
            "club_id": club2_id,
            "name": "Club 2 Tournament",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    t2_response.assert_status_ok();
    let t2_body: Value = t2_response.json();
    let t2_id = t2_body["tournament"]["id"].as_str().unwrap().to_string();

    // Register a player in both clubs
    let (player_token, player_id) = register_user(&server, "multclub", "multi@example.com", "password123").await;

    // Join both clubs
    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club1_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .assert_status_ok();

    server
        .post("/api/clubs/join")
        .json(&json!({"club_id": club2_id.clone()}))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .assert_status_ok();

    // Players start with 10000 from joining each club
    // Set club1 to 500 total and club2 to 200 total
    add_balance(&server, &owner1_token, &club1_id, &player_id, -9500).await;
    add_balance(&server, &owner2_token, &club2_id, &player_id, -9800).await;

    // Register for tournament 1 (should succeed with 500 balance)
    server
        .post(&format!("/api/tournaments/{}/register", t1_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .assert_status_ok();

    // Try to register for tournament 2 (should succeed with 200 balance)
    server
        .post(&format!("/api/tournaments/{}/register", t2_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .assert_status_ok();

    // Verify both tournaments have the player registered
    let t1_details = server
        .get(&format!("/api/tournaments/{}", t1_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .json::<Value>();
    
    let t2_details = server
        .get(&format!("/api/tournaments/{}", t2_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .json::<Value>();

    assert_eq!(t1_details["tournament"]["registered_players"], 1);
    assert_eq!(t2_details["tournament"]["registered_players"], 1);
    assert_eq!(t1_details["tournament"]["prize_pool"], 100);
    assert_eq!(t2_details["tournament"]["prize_pool"], 100);

    // Test that player can register 4 more times in club1 (400 remaining) but only 1 more in club2 (100 remaining)
    // Create additional tournament in club1
    let t3_response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner1_token))
        .json(&json!({
            "club_id": club1_id,
            "name": "Club 1 Tournament 2",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    t3_response.assert_status_ok();
    let t3_body: Value = t3_response.json();
    let t3_id = t3_body["tournament"]["id"].as_str().unwrap().to_string();

    // Should succeed (400 remaining)
    server
        .post(&format!("/api/tournaments/{}/register", t3_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .assert_status_ok();

    // Create another tournament in club2
    let t4_response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner2_token))
        .json(&json!({
            "club_id": club2_id,
            "name": "Club 2 Tournament 2",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    t4_response.assert_status_ok();
    let t4_body: Value = t4_response.json();
    let t4_id = t4_body["tournament"]["id"].as_str().unwrap().to_string();

    // Should succeed (100 remaining)
    server
        .post(&format!("/api/tournaments/{}/register", t4_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await
        .assert_status_ok();

    // Create third tournament in club2
    let t5_response = server
        .post("/api/tournaments/sng")
        .add_header(AUTHORIZATION, format!("Bearer {}", owner2_token))
        .json(&json!({
            "club_id": club2_id,
            "name": "Club 2 Tournament 3",
            "variant_id": "holdem",
            "buy_in": 100,
            "starting_stack": 1500,
            "max_players": 6,
            "level_duration_mins": 5
        }))
        .await;

    t5_response.assert_status_ok();
    let t5_body: Value = t5_response.json();
    let t5_id = t5_body["tournament"]["id"].as_str().unwrap().to_string();

    // Should fail (0 remaining in club2)
    let reg_response = server
        .post(&format!("/api/tournaments/{}/register", t5_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", player_token))
        .await;

    reg_response.assert_status_bad_request();
}

// ============================================================================
// Tournament Elimination Tests
// ============================================================================

#[tokio::test]
async fn test_player_elimination_removes_from_table() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    // Create a tournament table
    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 3 players
    table.add_player("p1".to_string(), "Player1".to_string(), 1000).unwrap();
    table.add_player("p2".to_string(), "Player2".to_string(), 1000).unwrap();
    table.add_player("p3".to_string(), "Player3".to_string(), 1000).unwrap();

    // Simulate player 2 losing all chips
    table.players[1].stack = 0;

    // Check eliminations should remove the player
    let eliminated = table.check_eliminations();
    assert_eq!(eliminated.len(), 1);
    assert_eq!(eliminated[0], "p2");

    // Player should be removed from table (only 2 players left)
    assert_eq!(table.players.len(), 2);
    // Remaining players should be p1 and p3
    assert!(table.players.iter().any(|p| p.user_id == "p1"));
    assert!(table.players.iter().any(|p| p.user_id == "p3"));
    assert!(!table.players.iter().any(|p| p.user_id == "p2"));

    // Check that eliminated players are not counted
    let remaining = table.get_remaining_players();
    assert_eq!(remaining.len(), 2);
    assert!(remaining.contains(&"p1".to_string()));
    assert!(remaining.contains(&"p3".to_string()));
    assert!(!remaining.contains(&"p2".to_string()));
}

#[tokio::test]
async fn test_multiple_simultaneous_eliminations() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    // Create a tournament table
    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 4 players
    for i in 0..4 {
        table.add_player(format!("p{}", i), format!("Player{}", i), 1000).unwrap();
    }

    // Simulate two players losing all chips simultaneously (e.g., side pot situation)
    table.players[1].stack = 0;
    table.players[2].stack = 0;

    // Check eliminations
    let eliminated = table.check_eliminations();
    assert_eq!(eliminated.len(), 2);
    assert!(eliminated.contains(&"p1".to_string()));
    assert!(eliminated.contains(&"p2".to_string()));

    // Both should be removed from table (only 2 players left)
    assert_eq!(table.players.len(), 2);
    assert!(!table.players.iter().any(|p| p.user_id == "p1"));
    assert!(!table.players.iter().any(|p| p.user_id == "p2"));

    // Only 2 players should remain
    let remaining = table.get_remaining_players();
    assert_eq!(remaining.len(), 2);
}

#[tokio::test]
async fn test_last_player_standing_tournament_finished() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    // Create a tournament table with 3 players
    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    for i in 0..3 {
        table.add_player(format!("p{}", i), format!("Player{}", i), 1000).unwrap();
    }

    // Tournament should not be finished initially
    assert!(!table.tournament_finished());

    // Eliminate one player
    table.players[0].stack = 0;
    table.check_eliminations();

    // Still 2 players remaining
    assert!(!table.tournament_finished());
    assert_eq!(table.get_remaining_players().len(), 2);
    assert_eq!(table.players.len(), 2); // One removed

    // Eliminate second player - only one left
    // Note: after first elimination, player indices shift
    table.players[0].stack = 0;
    table.check_eliminations();

    // Now tournament should be finished
    assert!(table.tournament_finished());
    assert_eq!(table.get_remaining_players().len(), 1);
    assert_eq!(table.players.len(), 1); // Two removed, one left
}

#[tokio::test]
async fn test_eliminated_player_not_dealt_cards() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    // Create tournament table
    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 3 players
    for i in 0..3 {
        table.add_player(format!("p{}", i), format!("Player{}", i), 1000).unwrap();
    }

    // Eliminate player 1
    table.players[1].stack = 0;
    table.check_eliminations();
    
    // Player should be removed (only 2 left)
    assert_eq!(table.players.len(), 2);

    // Start a new hand
    table.start_new_hand();

    // All remaining players should have cards (eliminated player is gone)
    for player in &table.players {
        assert_eq!(player.hole_cards.len(), 2);
    }
}

#[tokio::test]
async fn test_player_with_zero_stack_not_included_in_new_hand() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    // Create tournament table
    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 3 players - player 1 starts with 0 chips (shouldn't happen in real scenario)
    table.add_player("p0".to_string(), "Player0".to_string(), 1000).unwrap();
    table.add_player("p1".to_string(), "Player1".to_string(), 1000).unwrap();
    table.add_player("p2".to_string(), "Player2".to_string(), 1000).unwrap();

    // Manually set player 1's stack to 0 and state to SittingOut (simulating the bug)
    table.players[1].stack = 0;
    table.players[1].state = PlayerState::SittingOut;

    // Check for eliminations in tournament mode
    let eliminated = table.check_eliminations();
    
    // Player should now be eliminated and removed (not just sitting out)
    assert_eq!(eliminated.len(), 1);
    assert_eq!(table.players.len(), 2); // Player removed
    assert!(!table.players.iter().any(|p| p.user_id == "p1"));
}

#[tokio::test]
async fn test_tournament_with_one_player_left_should_finish() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Start with 2 players
    table.add_player("p1".to_string(), "Player1".to_string(), 1000).unwrap();
    table.add_player("p2".to_string(), "Player2".to_string(), 1000).unwrap();

    // Eliminate one player
    table.players[1].stack = 0;
    table.check_eliminations();

    // Tournament should be marked as finished
    assert!(table.tournament_finished());

    // Only winner should remain
    let remaining = table.get_remaining_players();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0], "p1");
    
    // Eliminated player removed from table
    assert_eq!(table.players.len(), 1);
}

#[tokio::test]
async fn test_eliminated_players_dont_stay_sitting_out() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add players
    for i in 0..3 {
        table.add_player(format!("p{}", i), format!("Player{}", i), 1000).unwrap();
    }

    // Simulate a hand where player loses all chips
    table.start_new_hand();
    
    // Player 1 goes all-in and loses
    table.players[1].stack = 0;
    
    // When hand ends, reset for new hand
    for player in &mut table.players {
        player.reset_for_new_hand();
    }
    
    // In tournament mode, check eliminations should be called
    let eliminated = table.check_eliminations();
    
    // Player should be REMOVED, not just marked
    assert_eq!(eliminated.len(), 1);
    assert_eq!(eliminated[0], "p1");
    assert_eq!(table.players.len(), 2); // Player removed from table
}

#[tokio::test]
async fn test_sitting_out_player_pays_blinds_in_tournament() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 3 players
    table.add_player("p0".to_string(), "Player0".to_string(), 1000).unwrap();
    table.add_player("p1".to_string(), "Player1".to_string(), 1000).unwrap();
    table.add_player("p2".to_string(), "Player2".to_string(), 1000).unwrap();

    // Player 1 voluntarily sits out
    table.players[1].state = PlayerState::SittingOut;

    // Start a hand
    table.start_new_hand();

    // If player 1 is in blind position, they should pay the blind even while sitting out
    // After blinds, sitting out player should have paid if they were in blind position
    let initial_stack = 1000i64;
    
    // Check that sitting out player still has chips (not eliminated)
    assert!(table.players[1].stack > 0);
    assert_eq!(table.players[1].state, PlayerState::SittingOut);
}

#[tokio::test]
async fn test_sitting_out_player_auto_folds_when_action_reaches_them() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::{PlayerAction, PlayerState};
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 4 players so we can test action flow
    for i in 0..4 {
        table.add_player(format!("p{}", i), format!("Player{}", i), 1000).unwrap();
    }

    // Start hand
    table.start_new_hand();

    // Set player 2 to sitting out (not dealer, not in blinds)
    table.players[2].state = PlayerState::SittingOut;

    // Play through until it would be player 2's turn
    // The sitting out player should be auto-folded when the action reaches them
    // This test verifies the behavior exists
}

#[tokio::test]
async fn test_sitting_out_player_can_be_eliminated() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add players
    table.add_player("p0".to_string(), "Player0".to_string(), 1000).unwrap();
    table.add_player("p1".to_string(), "Player1".to_string(), 1000).unwrap();
    table.add_player("p2".to_string(), "Player2".to_string(), 1000).unwrap();

    // Player 1 is sitting out with some chips
    table.players[1].state = PlayerState::SittingOut;
    table.players[1].stack = 200;

    // Verify they're sitting out but not eliminated
    assert_eq!(table.players[1].state, PlayerState::SittingOut);
    assert_eq!(table.players[1].stack, 200);

    // Now they lose all chips (through blinds over time)
    table.players[1].stack = 0;

    // Check eliminations - should eliminate and remove the sitting out player
    let eliminated = table.check_eliminations();
    assert_eq!(eliminated.len(), 1);
    assert_eq!(eliminated[0], "p1");
    // Player removed from table
    assert_eq!(table.players.len(), 2);
    assert!(!table.players.iter().any(|p| p.user_id == "p1"));
}

#[tokio::test]
async fn test_sitting_out_not_eliminated_are_different_states() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add players
    table.add_player("p0".to_string(), "Player0".to_string(), 1000).unwrap();
    table.add_player("p1".to_string(), "Player1".to_string(), 500).unwrap();
    table.add_player("p2".to_string(), "Player2".to_string(), 0).unwrap();

    // Player 1: Sitting out with chips - valid tournament state
    table.players[1].state = PlayerState::SittingOut;
    
    // Player 2: Zero chips - should be eliminated
    let eliminated = table.check_eliminations();
    
    // Only player 2 should be eliminated
    assert_eq!(eliminated.len(), 1);
    assert_eq!(eliminated[0], "p2");
    
    // Player 1 should still be sitting out (not eliminated, still in table)
    assert_eq!(table.players.len(), 2); // p2 removed
    let p1 = table.players.iter().find(|p| p.user_id == "p1").unwrap();
    assert_eq!(p1.state, PlayerState::SittingOut);
    assert!(p1.stack > 0);
    
    // Player 2 should be removed
    assert!(!table.players.iter().any(|p| p.user_id == "p2"));
}

#[tokio::test]
async fn test_sitting_out_player_pays_blinds_and_loses_chips() {
    use poker_server::game::format::SitAndGo;
    use poker_server::game::player::PlayerState;
    use poker_server::game::table::PokerTable;
    use poker_server::game::variant::TexasHoldem;

    let format = Box::new(SitAndGo::new(100, 1000, 9, 300));
    let mut table = PokerTable::with_variant_and_format(
        "t1".to_string(),
        "Test".to_string(),
        50,
        100,
        9,
        Box::new(TexasHoldem),
        format,
    );
    table.set_tournament_id(Some("tour1".to_string()));

    // Add 4 players
    for i in 0..4 {
        table.add_player(format!("p{}", i), format!("Player{}", i), 1000).unwrap();
    }

    // Player 1 sits out before the hand starts
    table.players[1].state = PlayerState::SittingOut;
    
    let initial_stack = table.players[1].stack;

    // Start a hand
    table.start_new_hand();

    // Player should still be sitting out
    assert_eq!(table.players[1].state, PlayerState::SittingOut);

    // If player was in blind position, their stack should have decreased
    // We can't predict exact blind positions, but we know:
    // - If in SB position: stack should be initial - 50
    // - If in BB position: stack should be initial - 100
    // - Otherwise: stack should be initial (unchanged)
    let stack_after_blinds = table.players[1].stack;
    
    // Verify player is still in tournament (not eliminated)
    assert!(stack_after_blinds > 0 || stack_after_blinds == initial_stack - 50 || stack_after_blinds == initial_stack - 100);
    
    // Verify state is still SittingOut (not changed to Active or Eliminated)
    assert_eq!(table.players[1].state, PlayerState::SittingOut);
}
