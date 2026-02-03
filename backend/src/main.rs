use poker_server::{api, auth, config, create_app, db, tournament, ws};
use std::sync::Arc;

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

    // Build router using lib function
    let app = create_app(
        auth_state,
        table_state,
        tournament_state,
        game_server.clone(),
    );

    // Spawn background task to check for auto-advances
    let game_server_clone = game_server.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
        loop {
            interval.tick().await;
            game_server_clone.check_all_tables_auto_advance().await;
        }
    });

    // Spawn background task for bot actions (check every 500ms)
    let game_server_bots = game_server.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
        loop {
            interval.tick().await;
            game_server_bots.check_bot_actions().await;
        }
    });

    // Spawn background task for tournament blind level advancement (check every 1 second)
    let tournament_mgr_blinds = tournament_manager.clone();
    tokio::spawn(async move {
        tracing::info!("Tournament blind level check task started (runs every 1s)");
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            if let Err(e) = tournament_mgr_blinds.check_all_blind_levels().await {
                tracing::error!("Error checking blind levels: {:?}", e);
            }
        }
    });

    // Spawn background task for tournament player eliminations (check every 5 seconds)
    let tournament_mgr_eliminations = tournament_manager.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            if let Err(e) = tournament_mgr_eliminations
                .check_tournament_eliminations()
                .await
            {
                tracing::error!("Error checking tournament eliminations: {:?}", e);
            }
        }
    });

    // Spawn background task for broadcasting tournament info (every 1 second)
    let tournament_mgr_info = tournament_manager.clone();
    tokio::spawn(async move {
        tracing::info!("Tournament info broadcast task started (runs every 1s)");
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            // Spawn each broadcast in its own task so it never blocks the timer
            let mgr = tournament_mgr_info.clone();
            tokio::spawn(async move {
                if let Err(e) = mgr.broadcast_tournament_info().await {
                    tracing::error!("Error broadcasting tournament info: {:?}", e);
                }
            });
        }
    });

    // Start server
    let listener = tokio::net::TcpListener::bind(&config.server_addr()).await?;
    tracing::info!("Server listening on {}", config.server_addr());

    axum::serve(listener, app).await?;

    Ok(())
}
