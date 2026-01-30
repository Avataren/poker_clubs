use crate::{
    auth::{AuthUser, JwtManager},
    db::models::Table,
    error::Result,
    game::constants::{DEFAULT_MAX_SEATS, DEFAULT_MAX_BUYIN_BB, DEFAULT_MIN_BUYIN_BB},
    ws::GameServer,
};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    Json, Router,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct TableAppState {
    pub pool: sqlx::SqlitePool,
    pub game_server: Arc<GameServer>,
    pub jwt_manager: Arc<JwtManager>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTableRequest {
    pub club_id: String,
    pub name: String,
    pub small_blind: i64,
    pub big_blind: i64,
}

#[derive(Debug, Serialize)]
pub struct TableListResponse {
    pub tables: Vec<Table>,
}

pub fn router() -> Router<Arc<TableAppState>> {
    Router::new()
        .route("/", post(create_table))
        .route("/club/:club_id", get(list_tables))
}

async fn create_table(
    State(state): State<Arc<TableAppState>>,
    headers: HeaderMap,
    Json(req): Json<CreateTableRequest>,
) -> Result<Json<Table>> {
    let auth_header = headers.get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let _auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;
    let table = Table::new(
        req.club_id,
        req.name.clone(),
        req.small_blind,
        req.big_blind,
        req.small_blind * DEFAULT_MIN_BUYIN_BB,
        req.big_blind * DEFAULT_MAX_BUYIN_BB,
        DEFAULT_MAX_SEATS as i32,
    );

    // Insert into database
    sqlx::query(
        "INSERT INTO tables (id, club_id, name, small_blind, big_blind, min_buyin, max_buyin, max_players, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&table.id)
    .bind(&table.club_id)
    .bind(&table.name)
    .bind(table.small_blind)
    .bind(table.big_blind)
    .bind(table.min_buyin)
    .bind(table.max_buyin)
    .bind(table.max_players)
    .bind(&table.created_at)
    .execute(&state.pool)
    .await?;

    // Create in-memory game table
    state.game_server.create_table(
        table.id.clone(),
        req.name,
        req.small_blind,
        req.big_blind,
    ).await;

    // Notify all users viewing this club that a new table was created
    state.game_server.notify_club(&table.club_id).await;

    Ok(Json(table))
}

async fn list_tables(
    State(state): State<Arc<TableAppState>>,
    Path(club_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TableListResponse>> {
    let auth_header = headers.get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let _auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;
    let tables: Vec<Table> = sqlx::query_as(
        "SELECT * FROM tables WHERE club_id = ?",
    )
    .bind(&club_id)
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(TableListResponse { tables }))
}
