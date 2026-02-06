use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub server_host: String,
    pub server_port: u16,
    pub cors_allowed_origins: Vec<String>,
    pub is_production: bool,
    pub oauth: OAuthConfig,
}

#[derive(Clone, Debug)]
pub struct OAuthConfig {
    pub google_client_id: String,
    pub apple_client_id: String,
    pub apple_team_id: String,
    pub apple_key_id: String,
    pub apple_private_key: String,
}

impl Config {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();

        let is_production = env::var("POKER_ENV")
            .map(|v| v.eq_ignore_ascii_case("production"))
            .unwrap_or(false);

        let jwt_secret = match env::var("JWT_SECRET") {
            Ok(secret) => {
                if is_production && secret.len() < 32 {
                    panic!("JWT_SECRET must be at least 32 characters in production");
                }
                secret
            }
            Err(_) => {
                if is_production {
                    panic!("JWT_SECRET environment variable must be set in production");
                }
                tracing::warn!(
                    "WARNING: Using default JWT secret. Set JWT_SECRET in production!"
                );
                "development_secret_key_change_in_production".to_string()
            }
        };

        let cors_allowed_origins = env::var("CORS_ALLOWED_ORIGINS")
            .map(|origins| {
                origins
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_else(|_| {
                vec![
                    "http://localhost:3000".to_string(),
                    "http://127.0.0.1:3000".to_string(),
                ]
            });

        Self {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:poker.db".to_string()),
            jwt_secret,
            server_host: env::var("SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            server_port: env::var("SERVER_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .expect("SERVER_PORT must be a number"),
            cors_allowed_origins,
            is_production,
            oauth: OAuthConfig::from_env(is_production),
        }
    }

    pub fn server_addr(&self) -> String {
        format!("{}:{}", self.server_host, self.server_port)
    }
}

impl OAuthConfig {
    pub fn from_env(is_production: bool) -> Self {
        let google_client_id = env::var("GOOGLE_CLIENT_ID").unwrap_or_default();
        let apple_client_id = env::var("APPLE_CLIENT_ID").unwrap_or_default();
        let apple_team_id = env::var("APPLE_TEAM_ID").unwrap_or_default();
        let apple_key_id = env::var("APPLE_KEY_ID").unwrap_or_default();
        let apple_private_key = env::var("APPLE_PRIVATE_KEY").unwrap_or_default();

        if is_production
            && (google_client_id.is_empty()
                || apple_client_id.is_empty()
                || apple_team_id.is_empty()
                || apple_key_id.is_empty()
                || apple_private_key.is_empty())
        {
            panic!(
                "OAuth environment variables must be set in production (GOOGLE_CLIENT_ID, APPLE_CLIENT_ID, APPLE_TEAM_ID, APPLE_KEY_ID, APPLE_PRIVATE_KEY)"
            );
        }

        if !is_production {
            if google_client_id.is_empty() {
                tracing::warn!("GOOGLE_CLIENT_ID is not set; Google login will be disabled.");
            }
            if apple_client_id.is_empty()
                || apple_team_id.is_empty()
                || apple_key_id.is_empty()
                || apple_private_key.is_empty()
            {
                tracing::warn!("Apple OAuth variables are missing; Apple login will be disabled.");
            }
        }

        Self {
            google_client_id,
            apple_client_id,
            apple_team_id,
            apple_key_id,
            apple_private_key,
        }
    }

    pub fn google_enabled(&self) -> bool {
        !self.google_client_id.is_empty()
    }

    pub fn apple_enabled(&self) -> bool {
        !self.apple_client_id.is_empty()
            && !self.apple_team_id.is_empty()
            && !self.apple_key_id.is_empty()
            && !self.apple_private_key.is_empty()
    }
}
