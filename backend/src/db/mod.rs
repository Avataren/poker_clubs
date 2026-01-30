pub mod models;

use sqlx::{sqlite::SqlitePool, Pool, Sqlite};
use std::path::Path;

pub type DbPool = Pool<Sqlite>;

pub async fn create_pool(database_url: &str) -> Result<DbPool, sqlx::Error> {
    // Create the database file if it doesn't exist
    if database_url.starts_with("sqlite:") {
        let db_path = database_url.strip_prefix("sqlite:").unwrap();
        if !Path::new(db_path).exists() {
            if let Some(parent) = Path::new(db_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::File::create(db_path).ok();
        }
    }

    SqlitePool::connect(database_url).await
}

pub async fn run_migrations(pool: &DbPool) -> Result<(), sqlx::Error> {
    // Read and execute migration file
    let migration_sql = include_str!("migrations/001_initial_schema.sql");

    sqlx::query(migration_sql)
        .execute(pool)
        .await?;

    tracing::info!("Database migrations completed successfully");
    Ok(())
}
