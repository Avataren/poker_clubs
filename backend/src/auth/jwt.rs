use crate::error::{AppError, Result};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: String, // user ID
    pub username: String,
    pub exp: usize, // expiration time
}

impl Claims {
    pub fn new(user_id: String, username: String, expiration_hours: i64) -> Self {
        let exp =
            (chrono::Utc::now() + chrono::Duration::hours(expiration_hours)).timestamp() as usize;

        Self {
            sub: user_id,
            username,
            exp,
        }
    }
}

#[derive(Clone)]
pub struct JwtManager {
    secret: String,
}

impl JwtManager {
    pub fn new(secret: String) -> Self {
        Self { secret }
    }

    pub fn create_token(&self, user_id: String, username: String) -> Result<String> {
        let claims = Claims::new(user_id, username, 24 * 7); // 7 days

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.secret.as_bytes()),
        )
        .map_err(|e| AppError::Auth(format!("Failed to create token: {}", e)))
    }

    pub fn verify_token(&self, token: &str) -> Result<Claims> {
        decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.secret.as_bytes()),
            &Validation::default(),
        )
        .map(|data| data.claims)
        .map_err(|e| AppError::Auth(format!("Invalid token: {}", e)))
    }
}

// Simple auth user struct
pub struct AuthUser {
    pub user_id: String,
    #[allow(dead_code)] // Useful for logging and future authorization checks
    pub username: String,
}

impl AuthUser {
    pub fn from_header(jwt_manager: &JwtManager, auth_header: &str) -> Result<Self> {
        // Bearer token format
        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or(AppError::Unauthorized)?;

        // Verify the token
        let claims = jwt_manager.verify_token(token)?;

        Ok(AuthUser {
            user_id: claims.sub,
            username: claims.username,
        })
    }
}
