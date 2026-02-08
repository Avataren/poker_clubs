use crate::{
    audit,
    auth::JwtManager,
    config::OAuthConfig,
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
    pub oauth_config: OAuthConfig,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/register", post(register))
        .route("/login", post(login))
        .route("/oauth/google", post(google_login))
        .route("/oauth/apple", post(apple_login))
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
            "Username cannot start with 'Bot_' - this prefix is reserved for system bots"
                .to_string(),
        ));
    }

    if let Err(msg) = validate_password(&req.password) {
        return Err(AppError::Validation(msg));
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

/// Dummy hash for timing-safe comparison when user is not found.
/// Generated once so that bcrypt::verify takes similar time as a real check.
const DUMMY_HASH: &str = "$2b$12$LJ3m4ys3Lg2VBe.LBsDMzuCdNhJFUJShHTzu/hNRccWFEMOAb.Kze";

async fn login(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<AuthResponse>> {
    // Find user by username (case-insensitive)
    let user: Option<User> = sqlx::query_as("SELECT * FROM users WHERE LOWER(username) = LOWER(?)")
        .bind(&req.username)
        .fetch_optional(&state.pool)
        .await?;

    // Timing-safe: always perform bcrypt::verify even when user not found
    let (user, valid) = match user {
        Some(u) => {
            if u.auth_provider != "local" {
                return Err(AppError::Auth(
                    "Use Google/Apple login for this account".to_string(),
                ));
            }
            let ok = bcrypt::verify(req.password.as_bytes(), &u.password_hash).map_err(|e| {
                AppError::Internal(anyhow::anyhow!("Failed to verify password: {}", e))
            })?;
            (Some(u), ok)
        }
        None => {
            // Perform dummy verify to equalize timing
            let _ = bcrypt::verify(req.password.as_bytes(), DUMMY_HASH);
            (None, false)
        }
    };

    if !valid {
        audit::log_auth_event(&req.username, "login_failed", false);
        return Err(AppError::Auth("Invalid username or password".to_string()));
    }

    let user = user.unwrap(); // Safe: valid=true means user is Some

    audit::log_auth_event(&user.username, "login", true);

    // Generate JWT token
    let token = state
        .jwt_manager
        .create_token(user.id.clone(), user.username.clone())?;

    Ok(Json(AuthResponse {
        token,
        user: user.into(),
    }))
}

#[derive(Debug, Deserialize)]
struct OAuthLoginRequest {
    id_token: String,
}

async fn google_login(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OAuthLoginRequest>,
) -> Result<Json<AuthResponse>> {
    if !state.oauth_config.google_enabled() {
        return Err(AppError::Auth("Google login is not configured".to_string()));
    }

    let profile =
        crate::auth::oauth::verify_google_token(&req.id_token, &state.oauth_config).await?;
    let user = crate::auth::oauth::find_or_create_user(&state.pool, profile).await?;

    audit::log_auth_event(&user.username, "login_google", true);

    let token = state
        .jwt_manager
        .create_token(user.id.clone(), user.username.clone())?;

    Ok(Json(AuthResponse {
        token,
        user: user.into(),
    }))
}

async fn apple_login(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OAuthLoginRequest>,
) -> Result<Json<AuthResponse>> {
    if !state.oauth_config.apple_enabled() {
        return Err(AppError::Auth("Apple login is not configured".to_string()));
    }

    let profile =
        crate::auth::oauth::verify_apple_token(&req.id_token, &state.oauth_config).await?;
    let user = crate::auth::oauth::find_or_create_user(&state.pool, profile).await?;

    audit::log_auth_event(&user.username, "login_apple", true);

    let token = state
        .jwt_manager
        .create_token(user.id.clone(), user.username.clone())?;

    Ok(Json(AuthResponse {
        token,
        user: user.into(),
    }))
}

/// Validate password meets security requirements.
fn validate_password(password: &str) -> std::result::Result<(), String> {
    if password.len() < 8 {
        return Err("Password must be at least 8 characters".to_string());
    }
    if password.len() > 72 {
        return Err("Password must be at most 72 characters".to_string());
    }
    let has_upper = password.chars().any(|c| c.is_ascii_uppercase());
    let has_lower = password.chars().any(|c| c.is_ascii_lowercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());
    if !has_upper || !has_lower || !has_digit {
        return Err(
            "Password must contain at least one uppercase letter, one lowercase letter, and one digit".to_string(),
        );
    }
    Ok(())
}
