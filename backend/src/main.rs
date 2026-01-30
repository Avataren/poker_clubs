mod api;
mod auth;
mod config;
mod db;
mod error;
mod game;
mod ws;

use axum::{
    routing::get,
    Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load config
    let config = config::Config::from_env();
    tracing::info!("Starting poker server on {}", config.server_addr());

    // Create database pool
    let pool = db::create_pool(&config.database_url).await?;
    tracing::info!("Database connected");

    // Run migrations
    db::run_migrations(&pool).await?;

    // Create JWT manager
    let jwt_manager = Arc::new(auth::JwtManager::new(config.jwt_secret.clone()));

    // Create game server
    let game_server = Arc::new(ws::GameServer::new(jwt_manager.clone(), Arc::new(pool.clone())));

    // Create shared state for auth/clubs endpoints
    let auth_state = Arc::new(api::AppState {
        pool: pool.clone(),
        jwt_manager: jwt_manager.clone(),
        game_server: game_server.clone(),
    });

    // Create shared state for tables endpoint
    let table_state = Arc::new(api::TableAppState {
        pool: pool.clone(),
        game_server: game_server.clone(),
        jwt_manager: jwt_manager.clone(),
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/", get(|| async { "Poker Server" }))
        .route("/health", get(|| async { "OK" }))
        .nest("/api/auth", api::auth_router().with_state(auth_state.clone()))
        .nest("/api/clubs", api::clubs_router().with_state(auth_state.clone()))
        .nest("/api/tables", api::tables_router().with_state(table_state))
        .route("/ws", get(ws::ws_handler).with_state(game_server.clone()))
        .layer(cors);

    // Spawn background task to check for auto-advances
    let game_server_clone = game_server.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
        loop {
            interval.tick().await;
            game_server_clone.check_all_tables_auto_advance().await;
        }
    });

    // Start server
    let listener = tokio::net::TcpListener::bind(&config.server_addr()).await?;
    tracing::info!("Server listening on {}", config.server_addr());

    axum::serve(listener, app).await?;

    Ok(())
}
