use crate::{
    auth::AuthUser,
    db::models::{Tournament, TournamentBlindLevel},
    error::{AppError, Result},
    tournament::manager::{MttConfig, SngConfig, TournamentManager},
    tournament::prizes::PrizeWinner,
};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    routing::{delete, get, post},
    Json, Router,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ==================== Request/Response Types ====================

#[derive(Debug, Deserialize)]
pub struct CreateSngRequest {
    pub club_id: String,
    pub name: String,
    pub buy_in: i64,
    pub max_players: i32,
    pub min_players: Option<i32>,
    pub starting_stack: i64,
    pub level_duration_mins: i32,
    pub variant_id: Option<String>,
    pub allow_rebuys: Option<bool>,
    pub max_rebuys: Option<i32>,
    pub rebuy_amount: Option<i64>,
    pub rebuy_stack: Option<i64>,
    pub allow_addons: Option<bool>,
    pub max_addons: Option<i32>,
    pub addon_amount: Option<i64>,
    pub addon_stack: Option<i64>,
    pub late_registration_mins: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct CreateMttRequest {
    pub club_id: String,
    pub name: String,
    pub buy_in: i64,
    pub max_players: i32,
    pub min_players: Option<i32>,
    pub starting_stack: i64,
    pub level_duration_mins: i32,
    pub scheduled_start: Option<String>, // ISO 8601 timestamp
    pub variant_id: Option<String>,
    pub allow_rebuys: Option<bool>,
    pub max_rebuys: Option<i32>,
    pub rebuy_amount: Option<i64>,
    pub rebuy_stack: Option<i64>,
    pub allow_addons: Option<bool>,
    pub max_addons: Option<i32>,
    pub addon_amount: Option<i64>,
    pub addon_stack: Option<i64>,
    pub late_registration_mins: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct TournamentResponse {
    pub tournament: Tournament,
    pub blind_levels: Vec<TournamentBlindLevel>,
}

#[derive(Debug, Serialize)]
pub struct TournamentDetailResponse {
    pub tournament: Tournament,
    pub blind_levels: Vec<TournamentBlindLevel>,
    pub registrations: Vec<PlayerRegistration>,
    pub is_registered: bool,
    pub can_register: bool,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct PlayerRegistration {
    pub user_id: String,
    pub username: String,
    pub registered_at: String,
    pub finish_position: Option<i32>,
    pub prize_amount: i64,
}

#[derive(Debug, Serialize)]
pub struct TournamentListResponse {
    pub tournaments: Vec<TournamentWithStats>,
}

#[derive(Debug, Serialize)]
pub struct TournamentWithStats {
    pub tournament: Tournament,
    pub registered_count: i32,
    pub is_registered: bool,
}

#[derive(Debug, Serialize)]
pub struct TournamentResultsResponse {
    pub tournament: Tournament,
    pub results: Vec<PlayerResult>,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct PlayerResult {
    pub user_id: String,
    pub username: String,
    pub finish_position: i32,
    pub prize_amount: i64,
    pub eliminated_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RegistrationResponse {
    pub success: bool,
    pub tournament: Tournament,
}

#[derive(Debug, Serialize)]
pub struct PrizesResponse {
    pub tournament: Tournament,
    pub prizes: Vec<PrizeWinner>,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct TournamentTableInfo {
    pub table_id: String,
    pub table_number: i32,
    pub table_name: String,
    pub player_count: i32,
}

#[derive(Debug, Serialize)]
pub struct TournamentTablesResponse {
    pub tables: Vec<TournamentTableInfo>,
}

// ==================== Extended AppState for Tournaments ====================

pub struct TournamentAppState {
    pub pool: crate::db::DbPool,
    pub jwt_manager: Arc<crate::auth::JwtManager>,
    pub game_server: Arc<crate::ws::GameServer>,
    pub tournament_manager: Arc<TournamentManager>,
}

// ==================== Router ====================

pub fn router() -> Router<Arc<TournamentAppState>> {
    Router::new()
        // Tournament management
        .route("/sng", post(create_sng))
        .route("/mtt", post(create_mtt))
        .route("/club/:club_id", get(list_club_tournaments))
        .route("/:id", get(get_tournament_details))
        .route("/:id/cancel", delete(cancel_tournament))
        // Registration
        .route("/:id/register", post(register_for_tournament))
        .route("/:id/unregister", delete(unregister_from_tournament))
        .route("/:id/rebuy", post(rebuy_tournament_entry))
        .route("/:id/addon", post(addon_tournament_entry))
        .route("/:id/players", get(get_tournament_players))
        .route("/:id/tables", get(get_tournament_tables))
        // Administration
        .route("/:id/start", post(start_tournament))
        .route("/:id/fill-bots", post(fill_with_bots))
        // Results
        .route("/:id/results", get(get_tournament_results))
        .route("/:id/prizes", get(get_tournament_prizes))
}

// ==================== Handlers ====================

async fn create_sng(
    State(state): State<Arc<TournamentAppState>>,
    headers: HeaderMap,
    Json(req): Json<CreateSngRequest>,
) -> Result<Json<TournamentResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    // Verify user is club admin
    verify_club_admin(&state.pool, &req.club_id, &auth_user.user_id).await?;

    // Input validation
    if req.buy_in < 0 {
        return Err(AppError::Validation("Buy-in must be non-negative".to_string()));
    }
    if req.starting_stack <= 0 {
        return Err(AppError::Validation("Starting stack must be positive".to_string()));
    }

    let config = SngConfig {
        name: req.name,
        variant_id: req.variant_id.unwrap_or_else(|| "holdem".to_string()),
        buy_in: req.buy_in,
        starting_stack: req.starting_stack,
        max_players: req.max_players,
        min_players: req.min_players.unwrap_or(2),
        level_duration_secs: (req.level_duration_mins * 60) as i64,
        allow_rebuys: req.allow_rebuys.unwrap_or(false),
        max_rebuys: req.max_rebuys.unwrap_or(0),
        rebuy_amount: req.rebuy_amount.unwrap_or(0),
        rebuy_stack: req.rebuy_stack.unwrap_or(0),
        allow_addons: req.allow_addons.unwrap_or(false),
        max_addons: req.max_addons.unwrap_or(0),
        addon_amount: req.addon_amount.unwrap_or(0),
        addon_stack: req.addon_stack.unwrap_or(0),
        late_registration_secs: req.late_registration_mins.unwrap_or(0).saturating_mul(60) as i64,
    };

    let tournament = state
        .tournament_manager
        .create_sng(&req.club_id, config)
        .await?;

    let blind_levels = load_blind_levels(&state.pool, &tournament.id).await?;

    // Notify club members
    state.game_server.notify_club(&req.club_id).await;

    Ok(Json(TournamentResponse {
        tournament,
        blind_levels,
    }))
}

async fn create_mtt(
    State(state): State<Arc<TournamentAppState>>,
    headers: HeaderMap,
    Json(req): Json<CreateMttRequest>,
) -> Result<Json<TournamentResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    // Verify user is club admin
    verify_club_admin(&state.pool, &req.club_id, &auth_user.user_id).await?;

    // Input validation
    if req.buy_in < 0 {
        return Err(AppError::Validation("Buy-in must be non-negative".to_string()));
    }
    if req.starting_stack <= 0 {
        return Err(AppError::Validation("Starting stack must be positive".to_string()));
    }

    let config = MttConfig {
        name: req.name,
        variant_id: req.variant_id.unwrap_or_else(|| "holdem".to_string()),
        buy_in: req.buy_in,
        starting_stack: req.starting_stack,
        max_players: req.max_players,
        min_players: req.min_players.unwrap_or(2),
        level_duration_secs: (req.level_duration_mins * 60) as i64,
        scheduled_start: req.scheduled_start,
        pre_seat_secs: 60, // 1 minute to find seat
        allow_rebuys: req.allow_rebuys.unwrap_or(false),
        max_rebuys: req.max_rebuys.unwrap_or(0),
        rebuy_amount: req.rebuy_amount.unwrap_or(0),
        rebuy_stack: req.rebuy_stack.unwrap_or(0),
        allow_addons: req.allow_addons.unwrap_or(false),
        max_addons: req.max_addons.unwrap_or(0),
        addon_amount: req.addon_amount.unwrap_or(0),
        addon_stack: req.addon_stack.unwrap_or(0),
        late_registration_secs: req.late_registration_mins.unwrap_or(0).saturating_mul(60) as i64,
    };

    let tournament = state
        .tournament_manager
        .create_mtt(&req.club_id, config)
        .await?;

    let blind_levels = load_blind_levels(&state.pool, &tournament.id).await?;

    // Notify club members
    state.game_server.notify_club(&req.club_id).await;

    Ok(Json(TournamentResponse {
        tournament,
        blind_levels,
    }))
}

async fn list_club_tournaments(
    State(state): State<Arc<TournamentAppState>>,
    Path(club_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TournamentListResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    // Verify user is club member
    verify_club_member(&state.pool, &club_id, &auth_user.user_id).await?;

    // Get tournaments (active and recent, limited to 50)
    let tournaments: Vec<Tournament> = sqlx::query_as(
        "SELECT * FROM tournaments
         WHERE club_id = ?
         AND (status != 'finished' OR finished_at > datetime('now', '-1 day'))
         ORDER BY
           CASE status
             WHEN 'registering' THEN 1
             WHEN 'seating' THEN 2
             WHEN 'running' THEN 3
             ELSE 4
           END,
           created_at DESC
         LIMIT 50",
    )
    .bind(&club_id)
    .fetch_all(&state.pool)
    .await?;

    let mut results = Vec::new();
    for tournament in tournaments {
        // Check if user is registered
        let is_registered: Option<(String,)> = sqlx::query_as(
            "SELECT user_id FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(&tournament.id)
        .bind(&auth_user.user_id)
        .fetch_optional(&state.pool)
        .await?;

        results.push(TournamentWithStats {
            registered_count: tournament.registered_players,
            is_registered: is_registered.is_some(),
            tournament,
        });
    }

    Ok(Json(TournamentListResponse {
        tournaments: results,
    }))
}

async fn get_tournament_tables(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TournamentTablesResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    // Get tournament to verify club membership
    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    verify_club_member(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    // Get active tournament tables with player counts from registrations
    // Note: For accurate counts, we query based on which table each player was assigned to
    let mut tables: Vec<TournamentTableInfo> = sqlx::query_as(
        "SELECT tt.table_id, tt.table_number, 
                '' as table_name,
                0 as player_count
         FROM tournament_tables tt
         WHERE tt.tournament_id = ? AND tt.is_active = 1
         ORDER BY tt.table_number",
    )
    .bind(&id)
    .fetch_all(&state.pool)
    .await?;

    // Get player count for each table
    for table in tables.iter_mut() {
        let count: Option<(i32,)> = sqlx::query_as(
            "SELECT COUNT(*) FROM tournament_registrations 
             WHERE tournament_id = ? AND starting_table_id = ? AND eliminated_at IS NULL",
        )
        .bind(&id)
        .bind(&table.table_id)
        .fetch_optional(&state.pool)
        .await?;

        table.player_count = count.map(|(c,)| c).unwrap_or(0);
    }

    // If no players are assigned yet (starting_table_id not set), but we have one table,
    // count all active registrations
    if tables.len() == 1 && tables[0].player_count == 0 {
        let total_players: Option<(i32,)> = sqlx::query_as(
            "SELECT COUNT(*) FROM tournament_registrations WHERE tournament_id = ? AND eliminated_at IS NULL"
        )
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?;

        if let Some((count,)) = total_players {
            tables[0].player_count = count;
        }
    }

    Ok(Json(TournamentTablesResponse { tables }))
}

async fn get_tournament_details(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TournamentDetailResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    // Verify user is club member
    verify_club_member(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    let blind_levels = load_blind_levels(&state.pool, &id).await?;

    // Get registrations with usernames
    let registrations: Vec<PlayerRegistration> = sqlx::query_as(
        "SELECT tr.user_id, u.username, tr.registered_at, tr.finish_position, tr.prize_amount
         FROM tournament_registrations tr
         JOIN users u ON tr.user_id = u.id
         WHERE tr.tournament_id = ?
         ORDER BY tr.registered_at",
    )
    .bind(&id)
    .fetch_all(&state.pool)
    .await?;

    // Check if current user is registered
    let is_registered = registrations.iter().any(|r| r.user_id == auth_user.user_id);

    let can_register = can_register_for_tournament(&tournament)
        && tournament.registered_players < tournament.max_players
        && !is_registered;

    Ok(Json(TournamentDetailResponse {
        tournament,
        blind_levels,
        registrations,
        is_registered,
        can_register,
    }))
}

async fn register_for_tournament(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<RegistrationResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    state
        .tournament_manager
        .register_player(&id, &auth_user.user_id, &auth_user.username)
        .await?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    // Notify club members of registration
    state.game_server.notify_club(&tournament.club_id).await;

    Ok(Json(RegistrationResponse {
        success: true,
        tournament,
    }))
}

async fn unregister_from_tournament(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<RegistrationResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    state
        .tournament_manager
        .unregister_player(&id, &auth_user.user_id)
        .await?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    // Notify club members
    state.game_server.notify_club(&tournament.club_id).await;

    Ok(Json(RegistrationResponse {
        success: true,
        tournament,
    }))
}

async fn rebuy_tournament_entry(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<RegistrationResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    state
        .tournament_manager
        .rebuy_player(&id, &auth_user.user_id, &auth_user.username)
        .await?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    state.game_server.notify_club(&tournament.club_id).await;

    Ok(Json(RegistrationResponse {
        success: true,
        tournament,
    }))
}

async fn addon_tournament_entry(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<RegistrationResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    state
        .tournament_manager
        .addon_player(&id, &auth_user.user_id)
        .await?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    state.game_server.notify_club(&tournament.club_id).await;

    Ok(Json(RegistrationResponse {
        success: true,
        tournament,
    }))
}

async fn get_tournament_players(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Vec<PlayerRegistration>>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    verify_club_member(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    let registrations: Vec<PlayerRegistration> = sqlx::query_as(
        "SELECT tr.user_id, u.username, tr.registered_at, tr.finish_position, tr.prize_amount
         FROM tournament_registrations tr
         JOIN users u ON tr.user_id = u.id
         WHERE tr.tournament_id = ?
         ORDER BY 
           CASE WHEN tr.finish_position IS NULL THEN 0 ELSE 1 END,
           tr.finish_position",
    )
    .bind(&id)
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(registrations))
}

async fn start_tournament(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Tournament>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    // Verify user is club admin
    verify_club_admin(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    state.tournament_manager.start_tournament(&id).await?;

    let updated_tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    // Notify all registered players
    state.game_server.notify_club(&tournament.club_id).await;

    Ok(Json(updated_tournament))
}

async fn cancel_tournament(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Tournament>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    // Verify user is club admin
    verify_club_admin(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    state
        .tournament_manager
        .cancel_tournament(&id, Some("Cancelled by admin"))
        .await?;

    let updated_tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    // Notify club members
    state.game_server.notify_club(&tournament.club_id).await;

    Ok(Json(updated_tournament))
}

async fn get_tournament_results(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TournamentResultsResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    verify_club_member(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    let results: Vec<PlayerResult> = sqlx::query_as(
        "SELECT tr.user_id, u.username, tr.finish_position, tr.prize_amount, tr.eliminated_at
         FROM tournament_registrations tr
         JOIN users u ON tr.user_id = u.id
         WHERE tr.tournament_id = ? AND tr.finish_position IS NOT NULL
         ORDER BY tr.finish_position",
    )
    .bind(&id)
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(TournamentResultsResponse {
        tournament,
        results,
    }))
}

async fn get_tournament_prizes(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<PrizesResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    verify_club_member(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    // Get actual prizes distributed
    let prizes: Vec<PrizeWinner> = sqlx::query_as(
        "SELECT tr.user_id, u.username, tr.finish_position as position, tr.prize_amount as amount
         FROM tournament_registrations tr
         JOIN users u ON tr.user_id = u.id
         WHERE tr.tournament_id = ? AND tr.prize_amount > 0
         ORDER BY tr.finish_position",
    )
    .bind(&id)
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(PrizesResponse { tournament, prizes }))
}

fn can_register_for_tournament(tournament: &Tournament) -> bool {
    if tournament.status == "registering" || tournament.status == "seating" {
        return true;
    }

    if tournament.status != "running" || tournament.late_registration_secs <= 0 {
        return false;
    }

    let actual_start = match tournament.actual_start.as_ref() {
        Some(start) => start,
        None => return false,
    };

    let start_time = match DateTime::parse_from_rfc3339(actual_start) {
        Ok(value) => value.with_timezone(&Utc),
        Err(_) => return false,
    };

    let deadline = start_time + Duration::seconds(tournament.late_registration_secs);
    Utc::now() <= deadline
}

async fn fill_with_bots(
    State(state): State<Arc<TournamentAppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TournamentDetailResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Tournament not found".to_string()))?;

    // Only allow during registration phase
    if tournament.status != "registering" {
        return Err(AppError::BadRequest(
            "Can only add bots during registration".to_string(),
        ));
    }

    verify_club_admin(&state.pool, &tournament.club_id, &auth_user.user_id).await?;

    // Get current registration count
    let (current_count,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM tournament_registrations WHERE tournament_id = ?")
            .bind(&id)
            .fetch_one(&state.pool)
            .await?;

    let current_count = current_count as i32;
    let spots_remaining = tournament.max_players - current_count;

    if spots_remaining <= 0 {
        return Err(AppError::BadRequest("Tournament is full".to_string()));
    }

    // Add bots to fill remaining spots
    for i in 0..spots_remaining {
        let bot_username = format!("Bot_{}", current_count + i + 1);

        tracing::info!("Creating bot: {}", bot_username);

        // Create bot user with is_bot flag
        let bot_id = uuid::Uuid::new_v4().to_string();
        let bot_password_hash = "$2b$12$BOTACCOUNT_NO_PASSWORD";

        let insert_result = sqlx::query(
            "INSERT INTO users (id, username, email, password_hash, is_bot) 
             VALUES (?, ?, ?, ?, 1)
             ON CONFLICT(username) DO NOTHING",
        )
        .bind(&bot_id)
        .bind(&bot_username)
        .bind(format!("{}@bot.local", bot_username))
        .bind(bot_password_hash)
        .execute(&state.pool)
        .await;

        if let Err(e) = insert_result {
            tracing::error!("Failed to insert bot user {}: {:?}", bot_username, e);
            return Err(e.into());
        }

        // Get the bot's actual ID (in case it already existed)
        let actual_bot_id =
            match sqlx::query_as::<_, (String,)>("SELECT id FROM users WHERE username = ?")
                .bind(&bot_username)
                .fetch_one(&state.pool)
                .await
            {
                Ok((id,)) => {
                    tracing::info!("Bot {} has ID: {}", bot_username, id);
                    id
                }
                Err(e) => {
                    tracing::error!("Failed to fetch bot ID for {}: {:?}", bot_username, e);
                    return Err(e.into());
                }
            };

        // Register bot for tournament (bots don't need club membership or balance)
        tracing::info!("Registering bot {} for tournament", bot_username);
        if let Err(e) = state
            .tournament_manager
            .register_player(&id, &actual_bot_id, &bot_username)
            .await
        {
            tracing::error!("Failed to register bot {}: {:?}", bot_username, e);
            return Err(e);
        }
        tracing::info!("Successfully registered bot {}", bot_username);
    }

    // Notify club members
    state.game_server.notify_club(&tournament.club_id).await;

    // Reload tournament to get updated counts
    let tournament: Tournament = sqlx::query_as("SELECT * FROM tournaments WHERE id = ?")
        .bind(&id)
        .fetch_one(&state.pool)
        .await?;

    // Return updated tournament details
    let blind_levels = load_blind_levels(&state.pool, &id).await?;

    let registrations: Vec<PlayerRegistration> = sqlx::query_as(
        "SELECT tr.user_id, u.username, tr.registered_at, tr.finish_position, tr.prize_amount
         FROM tournament_registrations tr
         JOIN users u ON tr.user_id = u.id
         WHERE tr.tournament_id = ?
         ORDER BY tr.registered_at",
    )
    .bind(&id)
    .fetch_all(&state.pool)
    .await?;

    let is_registered = registrations.iter().any(|r| r.user_id == auth_user.user_id);

    let can_register = can_register_for_tournament(&tournament)
        && registrations.len() < tournament.max_players as usize;

    Ok(Json(TournamentDetailResponse {
        tournament,
        blind_levels,
        registrations,
        is_registered,
        can_register,
    }))
}

// ==================== Helper Functions ====================

async fn verify_club_admin(pool: &crate::db::DbPool, club_id: &str, user_id: &str) -> Result<()> {
    let club: Option<(String,)> =
        sqlx::query_as("SELECT id FROM clubs WHERE id = ? AND admin_id = ?")
            .bind(club_id)
            .bind(user_id)
            .fetch_optional(pool)
            .await?;

    if club.is_none() {
        return Err(AppError::Forbidden);
    }

    Ok(())
}

async fn verify_club_member(pool: &crate::db::DbPool, club_id: &str, user_id: &str) -> Result<()> {
    let member: Option<(String,)> =
        sqlx::query_as("SELECT user_id FROM club_members WHERE club_id = ? AND user_id = ?")
            .bind(club_id)
            .bind(user_id)
            .fetch_optional(pool)
            .await?;

    if member.is_none() {
        return Err(AppError::Forbidden);
    }

    Ok(())
}

async fn load_blind_levels(
    pool: &crate::db::DbPool,
    tournament_id: &str,
) -> Result<Vec<TournamentBlindLevel>> {
    let levels: Vec<TournamentBlindLevel> = sqlx::query_as(
        "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? ORDER BY level_number",
    )
    .bind(tournament_id)
    .fetch_all(pool)
    .await?;

    Ok(levels)
}
