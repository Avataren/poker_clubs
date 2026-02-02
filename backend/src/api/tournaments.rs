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
    Json,
    Router,
    routing::{delete, get, post},
};
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
        .route("/:id/players", get(get_tournament_players))
        // Administration
        .route("/:id/start", post(start_tournament))
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

    let config = SngConfig {
        name: req.name,
        variant_id: req.variant_id.unwrap_or_else(|| "holdem".to_string()),
        buy_in: req.buy_in,
        starting_stack: req.starting_stack,
        max_players: req.max_players,
        min_players: req.min_players.unwrap_or(2),
        level_duration_secs: (req.level_duration_mins * 60) as i64,
    };

    let tournament = state
        .tournament_manager
        .create_sng(&req.club_id, config)
        .await?;

    let blind_levels = load_blind_levels(&state.pool, &tournament.id).await?;

    // Notify club members
    state
        .game_server
        .notify_club(&req.club_id)
        .await;

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
    };

    let tournament = state
        .tournament_manager
        .create_mtt(&req.club_id, config)
        .await?;

    let blind_levels = load_blind_levels(&state.pool, &tournament.id).await?;

    // Notify club members
    state
        .game_server
        .notify_club(&req.club_id)
        .await;

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

    // Get tournaments (active and recent)
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
           created_at DESC",
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
    let is_registered = registrations
        .iter()
        .any(|r| r.user_id == auth_user.user_id);

    let can_register = tournament.status == "registering"
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
    state
        .game_server
        .notify_club(&tournament.club_id)
        .await;

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
    state
        .game_server
        .notify_club(&tournament.club_id)
        .await;

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
    state
        .game_server
        .notify_club(&tournament.club_id)
        .await;

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
    state
        .game_server
        .notify_club(&tournament.club_id)
        .await;

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

    Ok(Json(PrizesResponse {
        tournament,
        prizes,
    }))
}

// ==================== Helper Functions ====================

async fn verify_club_admin(
    pool: &crate::db::DbPool,
    club_id: &str,
    user_id: &str,
) -> Result<()> {
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

async fn verify_club_member(
    pool: &crate::db::DbPool,
    club_id: &str,
    user_id: &str,
) -> Result<()> {
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
