use poker_server::{api, auth, config, db, tournament, ws};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create cancellation token for graceful shutdown
    let shutdown_token = CancellationToken::new();

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
    let game_server = Arc::new(ws::GameServer::new(
        jwt_manager.clone(),
        Arc::new(pool.clone()),
    ));

    // Create tournament manager
    let tournament_manager = Arc::new(tournament::TournamentManager::new(
        Arc::new(pool.clone()),
        game_server.clone(),
    ));

    // Create shared state for auth/clubs endpoints
    let auth_state = Arc::new(api::AppState {
        pool: pool.clone(),
        jwt_manager: jwt_manager.clone(),
        game_server: game_server.clone(),
        oauth_config: config.oauth.clone(),
    });

    // Create shared state for tables endpoint
    let table_state = Arc::new(api::TableAppState {
        pool: pool.clone(),
        game_server: game_server.clone(),
        jwt_manager: jwt_manager.clone(),
    });

    // Create shared state for tournaments endpoint
    let tournament_state = Arc::new(api::TournamentAppState {
        pool: pool.clone(),
        jwt_manager: jwt_manager.clone(),
        game_server: game_server.clone(),
        tournament_manager: tournament_manager.clone(),
    });

    // Build router using lib function with CORS origins from config
    let app = poker_server::create_app_with_cors(
        auth_state,
        table_state,
        tournament_state,
        game_server.clone(),
        &config.cors_allowed_origins,
    );

    // Spawn background task to check for auto-advances
    let game_server_clone = game_server.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    game_server_clone.check_all_tables_auto_advance().await;
                }
                _ = token.cancelled() => {
                    tracing::info!("Auto-advance task shutting down");
                    break;
                }
            }
        }
    });

    // Spawn background task for bot actions (check every 500ms)
    let game_server_bots = game_server.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    game_server_bots.check_bot_actions().await;
                }
                _ = token.cancelled() => {
                    tracing::info!("Bot actions task shutting down");
                    break;
                }
            }
        }
    });

    // Spawn background task for tournament blind level advancement (check every 1 second)
    let tournament_mgr_blinds = tournament_manager.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        tracing::info!("Tournament blind level check task started (runs every 1s)");
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = tournament_mgr_blinds.check_all_blind_levels().await {
                        tracing::error!("Error checking blind levels: {:?}", e);
                    }
                }
                _ = token.cancelled() => {
                    tracing::info!("Blind level check task shutting down");
                    break;
                }
            }
        }
    });

    // Spawn background task for tournament player eliminations (check every 5 seconds)
    let tournament_mgr_eliminations = tournament_manager.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = tournament_mgr_eliminations
                        .check_tournament_eliminations()
                        .await
                    {
                        tracing::error!("Error checking tournament eliminations: {:?}", e);
                    }
                }
                _ = token.cancelled() => {
                    tracing::info!("Tournament elimination task shutting down");
                    break;
                }
            }
        }
    });

    // Spawn background task for broadcasting tournament info (every 1 second)
    let tournament_mgr_info = tournament_manager.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        tracing::info!("Tournament info broadcast task started (runs every 1s)");
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let mgr = tournament_mgr_info.clone();
                    tokio::spawn(async move {
                        if let Err(e) = mgr.broadcast_tournament_info().await {
                            tracing::error!("Error broadcasting tournament info: {:?}", e);
                        }
                    });
                }
                _ = token.cancelled() => {
                    tracing::info!("Tournament info broadcast task shutting down");
                    break;
                }
            }
        }
    });

    // Spawn background task to clean up finished tournaments (check every 5 minutes)
    let tournament_mgr_cleanup = tournament_manager.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    tournament_mgr_cleanup.cleanup_finished_tournaments().await;
                }
                _ = token.cancelled() => {
                    tracing::info!("Tournament cleanup task shutting down");
                    break;
                }
            }
        }
    });

    // Spawn background task to clean up expired disconnections (check every 10 seconds)
    let game_server_dc = game_server.clone();
    let token = shutdown_token.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    game_server_dc.cleanup_expired_disconnections().await;
                }
                _ = token.cancelled() => {
                    tracing::info!("Disconnection cleanup task shutting down");
                    break;
                }
            }
        }
    });

    // Start server with graceful shutdown
    let listener = tokio::net::TcpListener::bind(&config.server_addr()).await?;
    tracing::info!("Server listening on {}", config.server_addr());

    let shutdown_signal = shutdown_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C handler");
        tracing::info!("Received CTRL+C, initiating graceful shutdown...");
        shutdown_signal.cancel();
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_token.cancelled().await;
            tracing::info!("Server shutting down gracefully");
        })
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}
