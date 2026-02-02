use crate::{
    api::auth::AppState,
    auth::AuthUser,
    db::models::{Club, ClubMember},
    error::Result,
    game::constants::DEFAULT_STARTING_BALANCE,
};
use axum::{
    extract::State,
    http::HeaderMap,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct CreateClubRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct JoinClubRequest {
    pub club_id: String,
}

#[derive(Debug, Serialize)]
pub struct ClubResponse {
    pub club: Club,
    pub is_admin: bool,
    pub balance: i64,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", post(create_club))
        .route("/my", get(get_my_clubs))
        .route("/all", get(get_all_clubs))
        .route("/join", post(join_club))
}

async fn create_club(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CreateClubRequest>,
) -> Result<Json<ClubResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;
    let club = Club::new(req.name, auth_user.user_id.clone());

    // Insert club
    sqlx::query("INSERT INTO clubs (id, name, admin_id, created_at) VALUES (?, ?, ?, ?)")
        .bind(&club.id)
        .bind(&club.name)
        .bind(&club.admin_id)
        .bind(&club.created_at)
        .execute(&state.pool)
        .await?;

    // Auto-join as member with starting balance
    let member = ClubMember::new(club.id.clone(), auth_user.user_id.clone());

    sqlx::query(
        "INSERT INTO club_members (club_id, user_id, balance, joined_at) VALUES (?, ?, ?, ?)",
    )
    .bind(&member.club_id)
    .bind(&member.user_id)
    .bind(DEFAULT_STARTING_BALANCE)
    .bind(&member.joined_at)
    .execute(&state.pool)
    .await?;

    // Notify all users viewing clubs list that a new club was created
    state.game_server.notify_global("club_created").await;

    Ok(Json(ClubResponse {
        club,
        is_admin: true,
        balance: DEFAULT_STARTING_BALANCE,
    }))
}

async fn get_my_clubs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Vec<ClubResponse>>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;
    let clubs: Vec<(String, String, String, String, i64)> = sqlx::query_as(
        r#"
        SELECT c.id, c.name, c.admin_id, c.created_at, cm.balance
        FROM clubs c
        JOIN club_members cm ON c.id = cm.club_id
        WHERE cm.user_id = ?
        "#,
    )
    .bind(&auth_user.user_id)
    .fetch_all(&state.pool)
    .await?;

    let response: Vec<ClubResponse> = clubs
        .into_iter()
        .map(|(id, name, admin_id, created_at, balance)| ClubResponse {
            club: Club {
                id: id.clone(),
                name,
                admin_id: admin_id.clone(),
                created_at,
            },
            is_admin: admin_id == auth_user.user_id,
            balance,
        })
        .collect();

    Ok(Json(response))
}

async fn get_all_clubs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Vec<ClubResponse>>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    let clubs: Vec<(String, String, String, String, Option<i64>)> = sqlx::query_as(
        r#"
        SELECT c.id, c.name, c.admin_id, c.created_at, cm.balance
        FROM clubs c
        LEFT JOIN club_members cm ON c.id = cm.club_id AND cm.user_id = ?
        "#,
    )
    .bind(&auth_user.user_id)
    .fetch_all(&state.pool)
    .await?;

    let response: Vec<ClubResponse> = clubs
        .into_iter()
        .map(|(id, name, admin_id, created_at, balance)| ClubResponse {
            club: Club {
                id: id.clone(),
                name,
                admin_id: admin_id.clone(),
                created_at,
            },
            is_admin: admin_id == auth_user.user_id,
            balance: balance.unwrap_or(0),
        })
        .collect();

    Ok(Json(response))
}

async fn join_club(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<JoinClubRequest>,
) -> Result<Json<ClubResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;
    // Check if club exists
    let club: Club = sqlx::query_as("SELECT * FROM clubs WHERE id = ?")
        .bind(&req.club_id)
        .fetch_one(&state.pool)
        .await?;

    // Check if already a member
    let existing: Option<(i64,)> =
        sqlx::query_as("SELECT balance FROM club_members WHERE club_id = ? AND user_id = ?")
            .bind(&req.club_id)
            .bind(&auth_user.user_id)
            .fetch_optional(&state.pool)
            .await?;

    if let Some((balance,)) = existing {
        return Ok(Json(ClubResponse {
            club,
            is_admin: false,
            balance,
        }));
    }

    // Join as new member
    let member = ClubMember::new(req.club_id.clone(), auth_user.user_id.clone());

    sqlx::query(
        "INSERT INTO club_members (club_id, user_id, balance, joined_at) VALUES (?, ?, ?, ?)",
    )
    .bind(&member.club_id)
    .bind(&member.user_id)
    .bind(DEFAULT_STARTING_BALANCE)
    .bind(&member.joined_at)
    .execute(&state.pool)
    .await?;

    Ok(Json(ClubResponse {
        club,
        is_admin: false,
        balance: DEFAULT_STARTING_BALANCE,
    }))
}
