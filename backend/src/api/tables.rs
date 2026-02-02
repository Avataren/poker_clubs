use crate::{
    auth::{AuthUser, JwtManager},
    db::models::Table,
    error::Result,
    game::{
        available_variants,
        constants::{DEFAULT_MAX_BUYIN_BB, DEFAULT_MAX_SEATS, DEFAULT_MIN_BUYIN_BB},
        format::{available_formats, format_from_id},
        variant_from_id,
    },
    ws::GameServer,
};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    routing::{get, post},
    Json, Router,
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
    /// Optional variant ID (default: "holdem")
    pub variant_id: Option<String>,
    /// Optional format ID (default: "cash")
    pub format_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TableListResponse {
    pub tables: Vec<Table>,
}

/// Response for available game variants
#[derive(Debug, Serialize)]
pub struct VariantsResponse {
    pub variants: Vec<VariantInfo>,
}

#[derive(Debug, Serialize)]
pub struct VariantInfo {
    pub id: String,
    pub name: String,
}

/// Response for available game formats
#[derive(Debug, Serialize)]
pub struct FormatsResponse {
    pub formats: Vec<FormatInfo>,
}

#[derive(Debug, Serialize)]
pub struct FormatInfo {
    pub id: String,
    pub name: String,
}

pub fn router() -> Router<Arc<TableAppState>> {
    Router::new()
        .route("/", post(create_table))
        .route("/club/:club_id", get(list_tables))
        .route("/variants", get(list_variants))
        .route("/formats", get(list_formats))
}

async fn create_table(
    State(state): State<Arc<TableAppState>>,
    headers: HeaderMap,
    Json(req): Json<CreateTableRequest>,
) -> Result<Json<Table>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let _auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;

    // Get variant (default to Texas Hold'em)
    let variant_id = req.variant_id.as_deref().unwrap_or("holdem").to_string();
    let variant = variant_from_id(&variant_id).ok_or_else(|| {
        crate::error::AppError::BadRequest(format!("Unknown variant: {}", variant_id))
    })?;

    // Get format (default to Cash Game)
    let format_id = req.format_id.as_deref().unwrap_or("cash").to_string();
    let format = format_from_id(
        &format_id,
        req.small_blind,
        req.big_blind,
        DEFAULT_MAX_SEATS,
    )
    .ok_or_else(|| crate::error::AppError::BadRequest(format!("Unknown format: {}", format_id)))?;

    let table = Table::with_variant_and_format(
        req.club_id,
        req.name.clone(),
        req.small_blind,
        req.big_blind,
        req.small_blind * DEFAULT_MIN_BUYIN_BB,
        req.big_blind * DEFAULT_MAX_BUYIN_BB,
        DEFAULT_MAX_SEATS as i32,
        variant_id.clone(),
        format_id.clone(),
    );

    // Insert into database
    sqlx::query(
        "INSERT INTO tables (id, club_id, name, small_blind, big_blind, min_buyin, max_buyin, max_players, variant_id, format_id, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&table.id)
    .bind(&table.club_id)
    .bind(&table.name)
    .bind(table.small_blind)
    .bind(table.big_blind)
    .bind(table.min_buyin)
    .bind(table.max_buyin)
    .bind(table.max_players)
    .bind(&table.variant_id)
    .bind(&table.format_id)
    .bind(&table.created_at)
    .execute(&state.pool)
    .await?;

    // Create in-memory game table with variant and format
    state
        .game_server
        .create_table_with_options(
            table.id.clone(),
            req.name,
            req.small_blind,
            req.big_blind,
            variant,
            format,
        )
        .await;

    // Notify all users viewing this club that a new table was created
    state.game_server.notify_club(&table.club_id).await;

    Ok(Json(table))
}

/// List all available poker variants
async fn list_variants() -> Json<VariantsResponse> {
    let variants: Vec<VariantInfo> = available_variants()
        .into_iter()
        .filter_map(|id| {
            variant_from_id(id).map(|v| VariantInfo {
                id: id.to_string(),
                name: v.name().to_string(),
            })
        })
        .collect();

    Json(VariantsResponse { variants })
}

/// List all available game formats
async fn list_formats() -> Json<FormatsResponse> {
    let formats: Vec<FormatInfo> = available_formats()
        .into_iter()
        .map(|id| {
            // Create a dummy format to get its name
            let name = match id {
                "cash" => "Cash Game",
                "sng" => "Sit & Go",
                "mtt" => "Multi-Table Tournament",
                _ => id,
            };
            FormatInfo {
                id: id.to_string(),
                name: name.to_string(),
            }
        })
        .collect();

    Json(FormatsResponse { formats })
}

async fn list_tables(
    State(state): State<Arc<TableAppState>>,
    Path(club_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<TableListResponse>> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::error::AppError::Unauthorized)?;
    let _auth_user = AuthUser::from_header(&state.jwt_manager, auth_header)?;
    let tables: Vec<Table> = sqlx::query_as("SELECT * FROM tables WHERE club_id = ?")
        .bind(&club_id)
        .fetch_all(&state.pool)
        .await?;

    Ok(Json(TableListResponse { tables }))
}
