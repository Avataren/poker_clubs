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
    // Disable foreign key checks for migrations
    sqlx::query("PRAGMA foreign_keys = OFF")
        .execute(pool)
        .await?;

    // Create migrations table if it doesn't exist
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS _sqlx_migrations (
            version BIGINT PRIMARY KEY,
            description TEXT NOT NULL,
            installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN NOT NULL,
            execution_time BIGINT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // List of migrations: (version, name, sql)
    let migrations: Vec<(i64, &str, &str)> = vec![
        (
            1,
            "initial_schema",
            include_str!("migrations/001_initial_schema.sql"),
        ),
        (
            2,
            "add_variant_and_format",
            include_str!("migrations/002_add_variant_and_format.sql"),
        ),
        (
            3,
            "tournament_tables",
            include_str!("migrations/003_tournament_tables.sql"),
        ),
        // Migration 4 was redundant - migration 3 already has pre_seat_secs field
    ];

    for (version, name, sql) in migrations {
        // Check if this migration has already been run
        let already_run = sqlx::query("SELECT 1 FROM _sqlx_migrations WHERE version = ?")
            .bind(version)
            .fetch_optional(pool)
            .await?
            .is_some();

        if already_run {
            tracing::debug!("Migration {} ({}) already applied", version, name);
            continue;
        }

        tracing::info!("Running migration {} ({})", version, name);

        let start_time = std::time::Instant::now();

        // Execute the migration
        match execute_migration_sql(pool, sql).await {
            Ok(_) => {
                let elapsed = start_time.elapsed().as_millis() as i64;
                sqlx::query(
                    "INSERT INTO _sqlx_migrations (version, description, success, execution_time) 
                     VALUES (?, ?, TRUE, ?)",
                )
                .bind(version)
                .bind(name)
                .bind(elapsed)
                .execute(pool)
                .await?;

                tracing::info!(
                    "Migration {} ({}) completed successfully in {}ms",
                    version,
                    name,
                    elapsed
                );
            }
            Err(e) => {
                let elapsed = start_time.elapsed().as_millis() as i64;

                // Try to record the failure, but don't fail if this fails
                let _ = sqlx::query(
                    "INSERT INTO _sqlx_migrations (version, description, success, execution_time) 
                     VALUES (?, ?, FALSE, ?)",
                )
                .bind(version)
                .bind(name)
                .bind(elapsed)
                .execute(pool)
                .await;

                // Re-enable foreign keys before returning error
                let _ = sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await;

                tracing::error!("Migration {} ({}) failed: {}", version, name, e);
                return Err(e);
            }
        }
    }

    // Re-enable foreign keys after all migrations
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await?;

    tracing::info!("All migrations completed successfully");
    Ok(())
}

async fn execute_migration_sql(pool: &DbPool, sql: &str) -> Result<(), sqlx::Error> {
    // Split by semicolon and execute each statement
    let statements: Vec<&str> = sql
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    for statement in statements.iter() {
        // Filter out pure comment blocks - only keep non-empty, non-comment lines
        let non_comment_lines: Vec<&str> = statement
            .lines()
            .filter(|line| !line.trim().starts_with("--") && !line.trim().is_empty())
            .collect();

        if non_comment_lines.is_empty() {
            continue;
        }

        // Reconstruct statement without leading comments
        let clean_statement = non_comment_lines.join("\n");

        sqlx::query(&clean_statement).execute(pool).await?;
    }

    Ok(())
}
