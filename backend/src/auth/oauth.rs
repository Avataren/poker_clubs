use crate::{
    config::OAuthConfig,
    db::{models::User, DbPool},
    error::{AppError, Result},
};
use chrono::Utc;
use jsonwebtoken::{decode, decode_header, jwk::JwkSet, DecodingKey, Validation};
use serde::Deserialize;
use uuid::Uuid;

const GOOGLE_JWKS_URL: &str = "https://www.googleapis.com/oauth2/v3/certs";
const APPLE_JWKS_URL: &str = "https://appleid.apple.com/auth/keys";

#[derive(Clone, Debug)]
pub enum AuthProvider {
    Google,
    Apple,
}

impl AuthProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuthProvider::Google => "google",
            AuthProvider::Apple => "apple",
        }
    }
}

#[derive(Clone, Debug)]
pub struct OAuthProfile {
    pub provider: AuthProvider,
    pub provider_user_id: String,
    pub email: Option<String>,
    pub email_verified: bool,
    pub display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Audience {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BoolOrString {
    Bool(bool),
    String(String),
}

#[derive(Debug, Deserialize)]
struct GoogleClaims {
    sub: String,
    email: Option<String>,
    email_verified: Option<bool>,
    name: Option<String>,
    aud: Audience,
    iss: String,
}

#[derive(Debug, Deserialize)]
struct AppleClaims {
    sub: String,
    email: Option<String>,
    email_verified: Option<BoolOrString>,
    aud: Audience,
    iss: String,
}

pub async fn verify_google_token(token: &str, config: &OAuthConfig) -> Result<OAuthProfile> {
    let jwks = fetch_jwks(GOOGLE_JWKS_URL).await?;
    let claims: GoogleClaims = decode_with_jwks(token, &jwks)?;

    if !audience_matches(&claims.aud, &config.google_client_id) {
        return Err(AppError::Auth("Invalid Google token audience".to_string()));
    }

    if claims.iss != "accounts.google.com" && claims.iss != "https://accounts.google.com" {
        return Err(AppError::Auth("Invalid Google token issuer".to_string()));
    }

    Ok(OAuthProfile {
        provider: AuthProvider::Google,
        provider_user_id: claims.sub,
        email: claims.email,
        email_verified: claims.email_verified.unwrap_or(false),
        display_name: claims.name,
    })
}

pub async fn verify_apple_token(token: &str, config: &OAuthConfig) -> Result<OAuthProfile> {
    let jwks = fetch_jwks(APPLE_JWKS_URL).await?;
    let claims: AppleClaims = decode_with_jwks(token, &jwks)?;

    if !audience_matches(&claims.aud, &config.apple_client_id) {
        return Err(AppError::Auth("Invalid Apple token audience".to_string()));
    }

    if claims.iss != "https://appleid.apple.com" {
        return Err(AppError::Auth("Invalid Apple token issuer".to_string()));
    }

    Ok(OAuthProfile {
        provider: AuthProvider::Apple,
        provider_user_id: claims.sub,
        email: claims.email,
        email_verified: parse_bool(claims.email_verified),
        display_name: None,
    })
}

pub async fn find_or_create_user(pool: &DbPool, profile: OAuthProfile) -> Result<User> {
    if let Some(user) = find_user_by_identity(pool, &profile).await? {
        return Ok(user);
    }

    let mut linked_user = None;
    let verified_email = profile
        .email
        .as_ref()
        .filter(|_| profile.email_verified)
        .map(|email| email.to_string());

    if let Some(email) = verified_email.as_ref() {
        linked_user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE LOWER(email) = LOWER(?)")
            .bind(email)
            .fetch_optional(pool)
            .await?;
    }

    let user = match linked_user {
        Some(existing) => existing,
        None => create_oauth_user(pool, &profile, verified_email.as_deref()).await?,
    };

    insert_identity(pool, &user.id, &profile, verified_email.as_deref()).await?;

    Ok(user)
}

async fn find_user_by_identity(pool: &DbPool, profile: &OAuthProfile) -> Result<Option<User>> {
    let user_id: Option<(String,)> = sqlx::query_as(
        "SELECT user_id FROM auth_identities WHERE provider = ? AND provider_user_id = ?",
    )
    .bind(profile.provider.as_str())
    .bind(&profile.provider_user_id)
    .fetch_optional(pool)
    .await?;

    let Some((user_id,)) = user_id else {
        return Ok(None);
    };

    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?")
        .bind(user_id)
        .fetch_optional(pool)
        .await?;

    Ok(user)
}

async fn create_oauth_user(
    pool: &DbPool,
    profile: &OAuthProfile,
    verified_email: Option<&str>,
) -> Result<User> {
    let base_username = profile
        .email
        .as_ref()
        .and_then(|email| email.split('@').next())
        .map(sanitize_username)
        .filter(|name| !name.is_empty())
        .or_else(|| profile.display_name.as_deref().map(sanitize_username))
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "player".to_string());

    let username = generate_unique_username(pool, &base_username).await?;
    let email = verified_email
        .map(|value| value.to_string())
        .unwrap_or_else(|| {
            format!(
                "{}_{}@oauth.local",
                profile.provider.as_str(),
                profile.provider_user_id
            )
        });

    let password_hash = bcrypt::hash(Uuid::new_v4().to_string(), bcrypt::DEFAULT_COST)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to hash password: {}", e)))?;

    let user = User {
        id: Uuid::new_v4().to_string(),
        username,
        email,
        password_hash,
        created_at: Utc::now().to_rfc3339(),
        is_bot: false,
        auth_provider: "oauth".to_string(),
    };

    sqlx::query("INSERT INTO users (id, username, email, password_hash, created_at, is_bot, auth_provider) VALUES (?, ?, ?, ?, ?, ?, ?)")
        .bind(&user.id)
        .bind(&user.username)
        .bind(&user.email)
        .bind(&user.password_hash)
        .bind(&user.created_at)
        .bind(user.is_bot)
        .bind(&user.auth_provider)
        .execute(pool)
        .await?;

    Ok(user)
}

async fn insert_identity(
    pool: &DbPool,
    user_id: &str,
    profile: &OAuthProfile,
    verified_email: Option<&str>,
) -> Result<()> {
    sqlx::query("INSERT INTO auth_identities (id, user_id, provider, provider_user_id, email, email_verified) VALUES (?, ?, ?, ?, ?, ?)")
        .bind(Uuid::new_v4().to_string())
        .bind(user_id)
        .bind(profile.provider.as_str())
        .bind(&profile.provider_user_id)
        .bind(verified_email)
        .bind(profile.email_verified)
        .execute(pool)
        .await?;

    Ok(())
}

async fn generate_unique_username(pool: &DbPool, base: &str) -> Result<String> {
    let mut candidate = base.to_string();
    let mut counter = 1;
    loop {
        let exists: Option<(String,)> =
            sqlx::query_as("SELECT id FROM users WHERE LOWER(username) = LOWER(?)")
                .bind(&candidate)
                .fetch_optional(pool)
                .await?;
        if exists.is_none() {
            return Ok(candidate);
        }
        counter += 1;
        candidate = format!("{}{}", base, counter);
    }
}

fn sanitize_username(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_lowercase()
}

async fn fetch_jwks(url: &str) -> Result<JwkSet> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| AppError::Auth(format!("Failed to fetch JWKS: {}", e)))?;
    let jwks = response
        .json::<JwkSet>()
        .await
        .map_err(|e| AppError::Auth(format!("Invalid JWKS response: {}", e)))?;
    Ok(jwks)
}

fn decode_with_jwks<T>(token: &str, jwks: &JwkSet) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let header = decode_header(token)
        .map_err(|_| AppError::Auth("Invalid token header".to_string()))?;
    let kid = header
        .kid
        .ok_or_else(|| AppError::Auth("Missing token key id".to_string()))?;
    let jwk = jwks
        .find(&kid)
        .ok_or_else(|| AppError::Auth("Unknown token key id".to_string()))?;
    let decoding_key = DecodingKey::from_jwk(jwk)
        .map_err(|_| AppError::Auth("Unsupported token key".to_string()))?;

    let mut validation = Validation::new(header.alg);
    validation.validate_aud = false;

    let token_data = decode::<T>(token, &decoding_key, &validation)
        .map_err(|_| AppError::Auth("Invalid token signature".to_string()))?;
    Ok(token_data.claims)
}

fn audience_matches(aud: &Audience, expected: &str) -> bool {
    match aud {
        Audience::Single(value) => value == expected,
        Audience::Multiple(values) => values.iter().any(|value| value == expected),
    }
}

fn parse_bool(value: Option<BoolOrString>) -> bool {
    match value {
        Some(BoolOrString::Bool(flag)) => flag,
        Some(BoolOrString::String(flag)) => flag.eq_ignore_ascii_case("true"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn creates_new_user_for_oauth_profile() {
        let pool = crate::create_test_db().await;
        let profile = OAuthProfile {
            provider: AuthProvider::Google,
            provider_user_id: "google-123".to_string(),
            email: Some("player@example.com".to_string()),
            email_verified: true,
            display_name: Some("Player One".to_string()),
        };

        let user = find_or_create_user(&pool, profile).await.expect("user");

        assert_eq!(user.email, "player@example.com");
        assert_eq!(user.auth_provider, "oauth");

        let identity_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(1) FROM auth_identities WHERE user_id = ? AND provider = ?",
        )
        .bind(&user.id)
        .bind("google")
        .fetch_one(&pool)
        .await
        .expect("identity count");
        assert_eq!(identity_count.0, 1);
    }

    #[tokio::test]
    async fn links_existing_user_by_verified_email() {
        let pool = crate::create_test_db().await;
        let password_hash = bcrypt::hash("Password1", bcrypt::DEFAULT_COST).unwrap();
        let existing = User::new("player".to_string(), "linked@example.com".to_string(), password_hash);
        sqlx::query("INSERT INTO users (id, username, email, password_hash, created_at, is_bot, auth_provider) VALUES (?, ?, ?, ?, ?, ?, ?)")
            .bind(&existing.id)
            .bind(&existing.username)
            .bind(&existing.email)
            .bind(&existing.password_hash)
            .bind(&existing.created_at)
            .bind(existing.is_bot)
            .bind(&existing.auth_provider)
            .execute(&pool)
            .await
            .expect("insert user");

        let profile = OAuthProfile {
            provider: AuthProvider::Apple,
            provider_user_id: "apple-456".to_string(),
            email: Some("linked@example.com".to_string()),
            email_verified: true,
            display_name: None,
        };

        let user = find_or_create_user(&pool, profile).await.expect("user");

        assert_eq!(user.id, existing.id);

        let identity_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(1) FROM auth_identities WHERE user_id = ? AND provider = ?",
        )
        .bind(&existing.id)
        .bind("apple")
        .fetch_one(&pool)
        .await
        .expect("identity count");
        assert_eq!(identity_count.0, 1);
    }
}
