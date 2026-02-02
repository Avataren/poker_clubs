//! Poker Server Library
//!
//! This module exposes the server components for integration testing.

pub mod api;
pub mod auth;
pub mod bot;
pub mod config;
pub mod db;
pub mod error;
pub mod game;
pub mod tournament;
pub mod ws;

use axum::{routing::get, Router};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

/// Creates the application router with all endpoints
pub fn create_app(
    auth_state: Arc<api::AppState>,
    table_state: Arc<api::TableAppState>,
    tournament_state: Arc<api::TournamentAppState>,
    game_server: Arc<ws::GameServer>,
) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/", get(|| async { "Poker Server" }))
        .route("/health", get(|| async { "OK" }))
        .nest(
            "/api/auth",
            api::auth_router().with_state(auth_state.clone()),
        )
        .nest(
            "/api/clubs",
            api::clubs_router().with_state(auth_state.clone()),
        )
        .nest("/api/tables", api::tables_router().with_state(table_state))
        .nest(
            "/api/tournaments",
            api::tournaments_router().with_state(tournament_state),
        )
        .route("/ws", get(ws::ws_handler).with_state(game_server))
        .layer(cors)
}

/// Test helper to create an in-memory database and run migrations
pub async fn create_test_db() -> db::DbPool {
    let pool = sqlx::sqlite::SqlitePool::connect(":memory:")
        .await
        .expect("Failed to create in-memory database");

    db::run_migrations(&pool)
        .await
        .expect("Failed to run migrations");

    pool
}

/// Test helper to create a fully configured test app
pub async fn create_test_app() -> (Router, Arc<ws::GameServer>) {
    let pool = create_test_db().await;
    let jwt_manager = Arc::new(auth::JwtManager::new("test_secret_key".to_string()));
    let game_server = Arc::new(ws::GameServer::new(
        jwt_manager.clone(),
        Arc::new(pool.clone()),
    ));

    let auth_state = Arc::new(api::AppState {
        pool: pool.clone(),
        jwt_manager: jwt_manager.clone(),
        game_server: game_server.clone(),
    });

    let table_state = Arc::new(api::TableAppState {
        pool: pool.clone(),
        game_server: game_server.clone(),
        jwt_manager: jwt_manager.clone(),
    });

    let tournament_manager = Arc::new(tournament::TournamentManager::new(
        Arc::new(pool.clone()),
        game_server.clone(),
    ));

    let tournament_state = Arc::new(api::TournamentAppState {
        pool: pool.clone(),
        jwt_manager: jwt_manager.clone(),
        game_server: game_server.clone(),
        tournament_manager,
    });

    let app = create_app(
        auth_state,
        table_state,
        tournament_state,
        game_server.clone(),
    );
    (app, game_server)
}
