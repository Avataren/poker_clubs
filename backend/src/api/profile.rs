use crate::{
    api::auth::AppState,
    auth::AuthUser,
    error::{AppError, Result},
};
use axum::{
    extract::State,
    http::HeaderMap,
    routing::{get, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const MIN_AVATAR_INDEX: i32 = 0;
const MAX_AVATAR_INDEX: i32 = 24;
const DECK_STYLE_CLASSIC: &str = "classic";
const DECK_STYLE_MULTI_COLOR: &str = "multi_color";

#[derive(Debug, Serialize)]
pub struct UserProfileResponse {
    pub user_id: String,
    pub username: String,
    pub avatar_index: i32,
    pub deck_style: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProfileRequest {
    pub avatar_index: Option<i32>,
    pub deck_style: Option<String>,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/me", get(get_my_profile))
        .route("/me", put(update_my_profile))
}

async fn get_my_profile(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<UserProfileResponse>> {
    let auth_user = auth_user_from_headers(&state, &headers)?;
    let profile = load_profile(&state, &auth_user.user_id).await?;
    Ok(Json(profile))
}

async fn update_my_profile(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<UpdateProfileRequest>,
) -> Result<Json<UserProfileResponse>> {
    let auth_user = auth_user_from_headers(&state, &headers)?;

    if let Some(avatar_index) = req.avatar_index {
        if !(MIN_AVATAR_INDEX..=MAX_AVATAR_INDEX).contains(&avatar_index) {
            return Err(AppError::Validation(format!(
                "avatar_index must be between {} and {}",
                MIN_AVATAR_INDEX, MAX_AVATAR_INDEX
            )));
        }

        sqlx::query("UPDATE users SET avatar_index = ? WHERE id = ?")
            .bind(avatar_index)
            .bind(&auth_user.user_id)
            .execute(&state.pool)
            .await?;
    }

    if let Some(deck_style) = req.deck_style.as_deref() {
        let normalized = deck_style.trim().to_lowercase();
        let valid = normalized == DECK_STYLE_CLASSIC || normalized == DECK_STYLE_MULTI_COLOR;
        if !valid {
            return Err(AppError::Validation(format!(
                "deck_style must be '{}' or '{}'",
                DECK_STYLE_CLASSIC, DECK_STYLE_MULTI_COLOR
            )));
        }

        sqlx::query("UPDATE users SET deck_style = ? WHERE id = ?")
            .bind(&normalized)
            .bind(&auth_user.user_id)
            .execute(&state.pool)
            .await?;
    }

    let profile = load_profile(&state, &auth_user.user_id).await?;
    Ok(Json(profile))
}

fn auth_user_from_headers(state: &Arc<AppState>, headers: &HeaderMap) -> Result<AuthUser> {
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AppError::Unauthorized)?;
    AuthUser::from_header(&state.jwt_manager, auth_header)
}

async fn load_profile(state: &Arc<AppState>, user_id: &str) -> Result<UserProfileResponse> {
    let profile: Option<(String, String, i32, String)> =
        sqlx::query_as("SELECT id, username, avatar_index, deck_style FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_optional(&state.pool)
            .await?;

    let Some((id, username, avatar_index, deck_style)) = profile else {
        return Err(AppError::Auth(
            "User account no longer exists. Please log in again.".to_string(),
        ));
    };

    Ok(UserProfileResponse {
        user_id: id,
        username,
        avatar_index,
        deck_style,
    })
}
