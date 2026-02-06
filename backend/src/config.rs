use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub server_host: String,
    pub server_port: u16,
    pub cors_allowed_origins: Vec<String>,
    pub is_production: bool,
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
        }
    }

    pub fn server_addr(&self) -> String {
        format!("{}:{}", self.server_host, self.server_port)
    }
}
