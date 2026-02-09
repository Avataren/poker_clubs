pub mod models;

use sqlx::{
    pool::PoolConnection,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
    Pool, Sqlite,
};
use std::path::Path;
use std::str::FromStr;

pub type DbPool = Pool<Sqlite>;

pub async fn create_pool(database_url: &str) -> Result<DbPool, sqlx::Error> {
    // Create the database file if it doesn't exist
    if database_url.starts_with("sqlite:") {
        let db_path = database_url.strip_prefix("sqlite:").unwrap();
        if !Path::new(db_path).exists() {
            if let Some(parent) = Path::new(db_path).parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    tracing::warn!("Failed to create database directory {:?}: {}", parent, e);
                }
            }
            if let Err(e) = std::fs::File::create(db_path) {
                tracing::warn!("Failed to create database file {}: {}", db_path, e);
            }
        }
    }

    let options = SqliteConnectOptions::from_str(database_url)?
        .journal_mode(SqliteJournalMode::Wal)
        .busy_timeout(std::time::Duration::from_secs(10))
        .pragma("synchronous", "NORMAL");

    SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await
}

pub async fn run_migrations(pool: &DbPool) -> Result<(), sqlx::Error> {
    let mut conn: PoolConnection<Sqlite> = pool.acquire().await?;

    // Disable foreign key checks for migrations
    sqlx::query("PRAGMA foreign_keys = OFF")
        .execute(&mut *conn)
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
    .execute(&mut *conn)
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
        (
            4,
            "add_is_bot_field",
            include_str!("migrations/004_add_is_bot_field.sql"),
        ),
        (
            5,
            "fix_tournament_tables_fk",
            include_str!("migrations/005_fix_tournament_tables_fk.sql"),
        ),
        (
            6,
            "add_tournament_rebuys_addons_late_registration",
            include_str!("migrations/006_add_tournament_rebuys_addons_late_registration.sql"),
        ),
        (
            7,
            "add_balance_constraints",
            include_str!("migrations/007_add_balance_constraints.sql"),
        ),
        (
            8,
            "add_oauth_identities",
            include_str!("migrations/008_add_oauth_identities.sql"),
        ),
        (
            9,
            "relax_tables_max_players_constraint",
            include_str!("migrations/009_relax_tables_max_players_constraint.sql"),
        ),
        (
            10,
            "add_user_settings",
            include_str!("migrations/010_add_user_settings.sql"),
        ),
    ];

    for (version, name, sql) in migrations {
        // Check migration status. Retry migrations that were previously recorded as failed.
        let migration_status: Option<(bool,)> =
            sqlx::query_as("SELECT success FROM _sqlx_migrations WHERE version = ?")
                .bind(version)
                .fetch_optional(&mut *conn)
                .await?;

        if let Some((true,)) = migration_status {
            tracing::debug!("Migration {} ({}) already applied", version, name);
            continue;
        }

        if let Some((false,)) = migration_status {
            tracing::warn!(
                "Migration {} ({}) was previously marked failed, retrying",
                version,
                name
            );
            sqlx::query("DELETE FROM _sqlx_migrations WHERE version = ?")
                .bind(version)
                .execute(&mut *conn)
                .await?;
        }

        tracing::info!("Running migration {} ({})", version, name);
        let start_time = std::time::Instant::now();

        // Run each migration in a transaction to avoid partial schema state on failure.
        sqlx::query("BEGIN IMMEDIATE").execute(&mut *conn).await?;

        match execute_migration_sql(&mut conn, sql).await {
            Ok(_) => {
                sqlx::query("COMMIT").execute(&mut *conn).await?;
                let elapsed = start_time.elapsed().as_millis() as i64;
                sqlx::query(
                    "INSERT INTO _sqlx_migrations (version, description, success, execution_time)
                     VALUES (?, ?, TRUE, ?)",
                )
                .bind(version)
                .bind(name)
                .bind(elapsed)
                .execute(&mut *conn)
                .await?;

                tracing::info!(
                    "Migration {} ({}) completed successfully in {}ms",
                    version,
                    name,
                    elapsed
                );
            }
            Err(e) => {
                let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
                let elapsed = start_time.elapsed().as_millis() as i64;

                // Try to record the failure, but don't fail if this insert fails.
                let _ = sqlx::query(
                    "INSERT INTO _sqlx_migrations (version, description, success, execution_time)
                     VALUES (?, ?, FALSE, ?)",
                )
                .bind(version)
                .bind(name)
                .bind(elapsed)
                .execute(&mut *conn)
                .await;

                // Re-enable foreign keys before returning error.
                let _ = sqlx::query("PRAGMA foreign_keys = ON")
                    .execute(&mut *conn)
                    .await;

                tracing::error!("Migration {} ({}) failed: {}", version, name, e);
                return Err(e);
            }
        }
    }

    // Re-enable foreign keys after all migrations
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&mut *conn)
        .await?;

    tracing::info!("All migrations completed successfully");
    Ok(())
}

async fn execute_migration_sql(
    conn: &mut PoolConnection<Sqlite>,
    sql: &str,
) -> Result<(), sqlx::Error> {
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

        sqlx::query(&clean_statement).execute(&mut **conn).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn retries_failed_migration_and_recovers_tables_schema() {
        let db_path =
            std::env::temp_dir().join(format!("db_migration_retry_{}.sqlite", Uuid::new_v4()));
        let db_url = format!("sqlite:{}", db_path.display());

        let pool = create_pool(&db_url).await.expect("create pool");
        run_migrations(&pool).await.expect("initial migrations");

        // Simulate a failed v9 run that left `tables_new` behind and no `tables`.
        sqlx::query("UPDATE _sqlx_migrations SET success = FALSE WHERE version = 9")
            .execute(&pool)
            .await
            .expect("mark migration 9 failed");
        sqlx::query("ALTER TABLE tables RENAME TO tables_new")
            .execute(&pool)
            .await
            .expect("rename tables to tables_new");

        // Rerun migrations: v9 should be retried and repair schema.
        run_migrations(&pool)
            .await
            .expect("rerun migrations should recover");

        let (success,): (i64,) =
            sqlx::query_as("SELECT success FROM _sqlx_migrations WHERE version = 9")
                .fetch_one(&pool)
                .await
                .expect("load migration status");
        assert_eq!(success, 1, "migration 9 should be marked successful");

        // Verify relaxed constraint works by inserting a >9 seat table.
        sqlx::query("INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)")
            .bind("mig_u1")
            .bind("mig_user")
            .bind("mig_user@example.com")
            .bind("x")
            .execute(&pool)
            .await
            .expect("insert user");

        sqlx::query("INSERT INTO clubs (id, name, admin_id) VALUES (?, ?, ?)")
            .bind("mig_c1")
            .bind("Migration Club")
            .bind("mig_u1")
            .execute(&pool)
            .await
            .expect("insert club");

        sqlx::query(
            "INSERT INTO tables (id, club_id, name, small_blind, big_blind, min_buyin, max_buyin, max_players, variant_id, format_id, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind("mig_t1")
        .bind("mig_c1")
        .bind("Large SNG Test Table")
        .bind(10_i64)
        .bind(20_i64)
        .bind(1000_i64)
        .bind(2000_i64)
        .bind(20_i64)
        .bind("holdem")
        .bind("sng")
        .execute(&pool)
        .await
        .expect("insert 20-seat table");
    }
}
