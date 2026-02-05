use crate::{
    db::models::Tournament,
    error::Result,
};
use chrono::{DateTime, Utc};
use std::sync::Arc;

use super::context::TournamentContext;

pub(crate) struct BroadcastService {
    ctx: Arc<TournamentContext>,
}

impl BroadcastService {
    pub(crate) fn new(ctx: Arc<TournamentContext>) -> Self {
        Self { ctx }
    }

    /// Broadcast tournament info to all players at tournament tables
    /// Called every second by background task
    pub(crate) async fn broadcast_tournament_info(&self) -> Result<()> {
        use crate::ws::messages::ServerMessage;

        // Get all running tournaments
        let running: Vec<Tournament> =
            sqlx::query_as("SELECT * FROM tournaments WHERE status = 'running'")
                .fetch_all(self.ctx.pool.as_ref())
                .await?;

        if running.is_empty() {
            return Ok(()); // No running tournaments, skip silently
        }

        for tournament in running {
            // Skip if no level start time (shouldn't happen in running state)
            let level_start_time = match &tournament.level_start_time {
                Some(t) => t.clone(),
                None => continue,
            };

            // Get current blind level
            let current_level: Option<crate::db::models::TournamentBlindLevel> = sqlx::query_as(
                "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? AND level_number = ?",
            )
            .bind(&tournament.id)
            .bind(tournament.current_blind_level)
            .fetch_optional(self.ctx.pool.as_ref())
            .await?;

            let current_level = match current_level {
                Some(l) => l,
                None => {
                    tracing::warn!(
                        "No blind level found for tournament {} level {}",
                        tournament.id,
                        tournament.current_blind_level
                    );
                    continue;
                }
            };

            // Get next blind level
            let next_level: Option<crate::db::models::TournamentBlindLevel> = sqlx::query_as(
                "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? AND level_number = ?",
            )
            .bind(&tournament.id)
            .bind(tournament.current_blind_level + 1)
            .fetch_optional(self.ctx.pool.as_ref())
            .await?;

            // Calculate remaining time for this level (server-side)
            let remaining_secs = remaining_level_time_secs(
                &level_start_time,
                tournament.level_duration_secs,
                Utc::now(),
            )
            .expect("Invalid tournament level_start_time");

            // Create tournament info message with current server time
            let message = ServerMessage::TournamentInfo {
                tournament_id: tournament.id.clone(),
                server_time: Utc::now().to_rfc3339(),
                level: tournament.current_blind_level as i64,
                small_blind: current_level.small_blind,
                big_blind: current_level.big_blind,
                ante: current_level.ante,
                level_start_time,
                level_duration_secs: tournament.level_duration_secs,
                level_time_remaining_secs: remaining_secs,
                next_small_blind: next_level.as_ref().map(|l| l.small_blind),
                next_big_blind: next_level.as_ref().map(|l| l.big_blind),
            };

            // Broadcast to all tables in this tournament
            self.ctx
                .game_server
                .broadcast_tournament_event(&tournament.id, message)
                .await;
        }

        Ok(())
    }
}

fn remaining_level_time_secs(
    level_start_time: &str,
    level_duration_secs: i64,
    now: DateTime<Utc>,
) -> Option<i64> {
    let level_start = DateTime::parse_from_rfc3339(level_start_time)
        .ok()?
        .with_timezone(&Utc);
    let elapsed_secs = (now - level_start).num_seconds();
    Some((level_duration_secs - elapsed_secs).max(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn remaining_level_time_secs_computes_remaining() {
        let now = Utc::now();
        let start = (now - Duration::seconds(30)).to_rfc3339();

        assert_eq!(remaining_level_time_secs(&start, 60, now), Some(30));
        assert_eq!(remaining_level_time_secs("bad", 60, now), None);
    }
}
