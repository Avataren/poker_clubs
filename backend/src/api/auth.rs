use crate::{
    auth::JwtManager,
    db::{models::User, DbPool},
    error::{AppError, Result},
    ws::GameServer,
};
use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub email: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub token: String,
    pub user: UserResponse,
}

#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: String,
    pub username: String,
    pub email: String,
}

impl From<User> for UserResponse {
    fn from(user: User) -> Self {
        Self {
            id: user.id,
            username: user.username,
            email: user.email,
        }
    }
}

pub struct AppState {
    pub pool: DbPool,
    pub jwt_manager: Arc<JwtManager>,
    pub game_server: Arc<GameServer>,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/register", post(register))
        .route("/login", post(login))
}

async fn register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>> {
    // Validate input
    if req.username.is_empty() || req.email.is_empty() || req.password.is_empty() {
        return Err(AppError::Validation("All fields are required".to_string()));
    }

    // Prevent users from registering with bot-like usernames
    if req.username.starts_with("Bot_") || req.username.to_lowercase().starts_with("bot_") {
        return Err(AppError::Validation(
            "Username cannot start with 'Bot_' - this prefix is reserved for system bots".to_string(),
        ));
    }

    if req.password.len() < 6 {
        return Err(AppError::Validation(
            "Password must be at least 6 characters".to_string(),
        ));
    }

    // Check if username or email already exists (case-insensitive)
    let existing: Option<(String,)> = sqlx::query_as(
        "SELECT id FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
    )
    .bind(&req.username)
    .bind(&req.email)
    .fetch_optional(&state.pool)
    .await?;

    if existing.is_some() {
        return Err(AppError::Validation(
            "Username or email already exists".to_string(),
        ));
    }

    // Hash password
    let password_hash = bcrypt::hash(req.password.as_bytes(), bcrypt::DEFAULT_COST)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to hash password: {}", e)))?;

    // Create user
    let user = User::new(req.username, req.email, password_hash);

    sqlx::query(
        "INSERT INTO users (id, username, email, password_hash, created_at, is_bot) VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&user.id)
    .bind(&user.username)
    .bind(&user.email)
    .bind(&user.password_hash)
    .bind(&user.created_at)
    .bind(false)
    .execute(&state.pool)
    .await?;

    // Generate JWT token
    let token = state
        .jwt_manager
        .create_token(user.id.clone(), user.username.clone())?;

    Ok(Json(AuthResponse {
        token,
        user: user.into(),
    }))
}

async fn login(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<AuthResponse>> {
    // Find user by username (case-insensitive)
    let user: User = sqlx::query_as("SELECT * FROM users WHERE LOWER(username) = LOWER(?)")
        .bind(&req.username)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| AppError::Auth("Invalid username or password".to_string()))?;

    // Verify password
    let valid = bcrypt::verify(req.password.as_bytes(), &user.password_hash)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to verify password: {}", e)))?;

    if !valid {
        return Err(AppError::Auth("Invalid username or password".to_string()));
    }

    // Generate JWT token
    let token = state
        .jwt_manager
        .create_token(user.id.clone(), user.username.clone())?;

    Ok(Json(AuthResponse {
        token,
        user: user.into(),
    }))
}
