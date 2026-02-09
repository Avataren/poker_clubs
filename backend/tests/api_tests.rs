//! Integration tests for the Poker Server API
//!
//! These tests verify that the HTTP API endpoints work correctly
//! with a real database and authentication flow.

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

// ============================================================================
// Health Check Tests
// ============================================================================

#[tokio::test]
async fn test_health_endpoint() {
    let server = setup().await;

    let response = server.get("/health").await;

    response.assert_status_ok();
    response.assert_text("OK");
}

#[tokio::test]
async fn test_root_endpoint() {
    let server = setup().await;

    let response = server.get("/").await;

    response.assert_status_ok();
    response.assert_text("Poker Server");
}

// ============================================================================
// Authentication Tests
// ============================================================================

#[tokio::test]
async fn test_register_new_user() {
    let server = setup().await;

    let response = server
        .post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "Password123"
        }))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["token"].is_string());
    assert_eq!(body["user"]["username"], "testuser");
    assert_eq!(body["user"]["email"], "test@example.com");
}

#[tokio::test]
async fn test_register_duplicate_username() {
    let server = setup().await;

    // Register first user
    server
        .post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test1@example.com",
            "password": "Password123"
        }))
        .await
        .assert_status_ok();

    // Try to register with same username
    let response = server
        .post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test2@example.com",
            "password": "Password123"
        }))
        .await;

    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_register_short_password() {
    let server = setup().await;

    let response = server
        .post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "short"
        }))
        .await;

    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_login_success() {
    let server = setup().await;

    // Register a user first
    server
        .post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "Password123"
        }))
        .await
        .assert_status_ok();

    // Login
    let response = server
        .post("/api/auth/login")
        .json(&json!({
            "username": "testuser",
            "password": "Password123"
        }))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["token"].is_string());
    assert_eq!(body["user"]["username"], "testuser");
}

#[tokio::test]
async fn test_login_wrong_password() {
    let server = setup().await;

    // Register a user first
    server
        .post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "Password123"
        }))
        .await
        .assert_status_ok();

    // Login with wrong password
    let response = server
        .post("/api/auth/login")
        .json(&json!({
            "username": "testuser",
            "password": "wrongpassword"
        }))
        .await;

    response.assert_status_unauthorized();
}

#[tokio::test]
async fn test_login_nonexistent_user() {
    let server = setup().await;

    let response = server
        .post("/api/auth/login")
        .json(&json!({
            "username": "nonexistent",
            "password": "Password123"
        }))
        .await;

    response.assert_status_unauthorized();
}

// ============================================================================
// Club Tests
// ============================================================================

#[tokio::test]
async fn test_create_club() {
    let server = setup().await;
    let token = register_user(&server, "clubowner", "owner@example.com", "Password123").await;

    let response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "name": "Test Club"
        }))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["club"]["name"], "Test Club");
}

#[tokio::test]
async fn test_create_club_unauthorized() {
    let server = setup().await;

    let response = server
        .post("/api/clubs")
        .json(&json!({
            "name": "Test Club"
        }))
        .await;

    response.assert_status_unauthorized();
}

#[tokio::test]
async fn test_get_my_clubs() {
    let server = setup().await;
    let token = register_user(&server, "clubowner", "owner@example.com", "Password123").await;

    // Create a club
    server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({ "name": "My Club" }))
        .await
        .assert_status_ok();

    // Get my clubs
    let response = server
        .get("/api/clubs/my")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    let clubs = body.as_array().unwrap();
    assert_eq!(clubs.len(), 1);
    assert_eq!(clubs[0]["club"]["name"], "My Club");
}

// ============================================================================
// Profile Tests
// ============================================================================

#[tokio::test]
async fn test_get_profile_defaults() {
    let server = setup().await;
    let token = register_user(
        &server,
        "profile_user",
        "profile@example.com",
        "Password123",
    )
    .await;

    let response = server
        .get("/api/profile/me")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert_eq!(body["username"], "profile_user");
    assert_eq!(body["avatar_index"], 0);
    assert_eq!(body["deck_style"], "classic");
}

#[tokio::test]
async fn test_update_profile_settings() {
    let server = setup().await;
    let token = register_user(
        &server,
        "settings_user",
        "settings@example.com",
        "Password123",
    )
    .await;

    let response = server
        .put("/api/profile/me")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "avatar_index": 7,
            "deck_style": "multi_color"
        }))
        .await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert_eq!(body["avatar_index"], 7);
    assert_eq!(body["deck_style"], "multi_color");

    // Verify persisted value
    let read_back = server
        .get("/api/profile/me")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;
    read_back.assert_status_ok();
    let profile: Value = read_back.json();
    assert_eq!(profile["avatar_index"], 7);
    assert_eq!(profile["deck_style"], "multi_color");
}

#[tokio::test]
async fn test_update_profile_rejects_invalid_avatar_index() {
    let server = setup().await;
    let token = register_user(
        &server,
        "bad_avatar_user",
        "bad_avatar@example.com",
        "Password123",
    )
    .await;

    let response = server
        .put("/api/profile/me")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "avatar_index": 25
        }))
        .await;

    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_update_profile_rejects_invalid_deck_style() {
    let server = setup().await;
    let token = register_user(
        &server,
        "bad_deck_user",
        "bad_deck@example.com",
        "Password123",
    )
    .await;

    let response = server
        .put("/api/profile/me")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "deck_style": "neon_rainbow"
        }))
        .await;

    response.assert_status_bad_request();
}

// ============================================================================
// Table Tests
// ============================================================================

#[tokio::test]
async fn test_create_table_with_default_variant() {
    let server = setup().await;
    let token = register_user(&server, "tableowner", "table@example.com", "Password123").await;

    // Create a club first
    let club_response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({ "name": "Table Club" }))
        .await;

    club_response.assert_status_ok();
    let club: Value = club_response.json();
    let club_id = club["club"]["id"].as_str().unwrap();

    // Create a table
    let response = server
        .post("/api/tables")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "Test Table",
            "small_blind": 50,
            "big_blind": 100
        }))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["name"], "Test Table");
    assert_eq!(body["small_blind"], 50);
    assert_eq!(body["big_blind"], 100);
}

#[tokio::test]
async fn test_create_table_with_omaha_variant() {
    let server = setup().await;
    let token = register_user(&server, "omahaowner", "omaha@example.com", "Password123").await;

    // Create a club first
    let club_response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({ "name": "Omaha Club" }))
        .await;

    club_response.assert_status_ok();
    let club: Value = club_response.json();
    let club_id = club["club"]["id"].as_str().unwrap();

    // Create an Omaha table
    let response = server
        .post("/api/tables")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "Omaha Table",
            "small_blind": 25,
            "big_blind": 50,
            "variant_id": "omaha"
        }))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["name"], "Omaha Table");
}

#[tokio::test]
async fn test_create_table_with_plo_variant() {
    let server = setup().await;
    let token = register_user(&server, "ploowner", "plo@example.com", "Password123").await;

    let club_response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({ "name": "PLO Club" }))
        .await;

    club_response.assert_status_ok();
    let club: Value = club_response.json();
    let club_id = club["club"]["id"].as_str().unwrap();

    let response = server
        .post("/api/tables")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "PLO Table",
            "small_blind": 25,
            "big_blind": 50,
            "variant_id": "plo"
        }))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["name"], "PLO Table");
}

#[tokio::test]
async fn test_create_table_with_invalid_variant() {
    let server = setup().await;
    let token = register_user(&server, "badvariant", "bad@example.com", "Password123").await;

    // Create a club first
    let club_response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({ "name": "Bad Club" }))
        .await;

    club_response.assert_status_ok();
    let club: Value = club_response.json();
    let club_id = club["club"]["id"].as_str().unwrap();

    // Try to create table with invalid variant
    let response = server
        .post("/api/tables")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({
            "club_id": club_id,
            "name": "Bad Table",
            "small_blind": 50,
            "big_blind": 100,
            "variant_id": "invalid_variant"
        }))
        .await;

    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_list_variants() {
    let server = setup().await;

    let response = server.get("/api/tables/variants").await;

    response.assert_status_ok();

    let body: Value = response.json();
    let variants = body["variants"].as_array().unwrap();

    // Should have at least holdem and omaha
    let variant_ids: Vec<&str> = variants.iter().map(|v| v["id"].as_str().unwrap()).collect();

    assert!(variant_ids.contains(&"holdem"));
    assert!(variant_ids.contains(&"omaha"));
    assert!(variant_ids.contains(&"plo"));
}

#[tokio::test]
async fn test_list_formats() {
    let server = setup().await;

    let response = server.get("/api/tables/formats").await;

    response.assert_status_ok();

    let body: Value = response.json();
    let formats = body["formats"].as_array().unwrap();

    // Should have at least cash and sng
    let format_ids: Vec<&str> = formats.iter().map(|f| f["id"].as_str().unwrap()).collect();

    assert!(format_ids.contains(&"cash"));
    assert!(format_ids.contains(&"sng"));
}

#[tokio::test]
async fn test_get_club_tables() {
    let server = setup().await;
    let token = register_user(&server, "tablelister", "list@example.com", "Password123").await;

    // Create a club
    let club_response = server
        .post("/api/clubs")
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .json(&json!({ "name": "List Club" }))
        .await;

    club_response.assert_status_ok();
    let club: Value = club_response.json();
    let club_id = club["club"]["id"].as_str().unwrap();

    // Create two tables
    for i in 1..=2 {
        server
            .post("/api/tables")
            .add_header(AUTHORIZATION, format!("Bearer {}", token))
            .json(&json!({
                "club_id": club_id,
                "name": format!("Table {}", i),
                "small_blind": 50,
                "big_blind": 100
            }))
            .await
            .assert_status_ok();
    }

    // Get club tables
    let response = server
        .get(&format!("/api/tables/club/{}", club_id))
        .add_header(AUTHORIZATION, format!("Bearer {}", token))
        .await;

    response.assert_status_ok();

    let body: Value = response.json();
    let tables = body["tables"].as_array().unwrap();
    assert_eq!(tables.len(), 2);
}
