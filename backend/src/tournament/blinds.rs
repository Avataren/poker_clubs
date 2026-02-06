use crate::{
    db::models::Tournament,
    error::Result,
};
use chrono::{DateTime, Utc};
use std::sync::Arc;

use super::context::TournamentContext;

pub(crate) struct BlindsService {
    ctx: Arc<TournamentContext>,
}

impl BlindsService {
    pub(crate) fn new(ctx: Arc<TournamentContext>) -> Self {
        Self { ctx }
    }

    /// Advance to the next blind level
    pub(crate) async fn advance_blind_level(&self, tournament_id: &str) -> Result<bool> {
        tracing::info!(
            "advance_blind_level called for tournament {}",
            tournament_id
        );
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;

        if tournament.status != "running" {
            tracing::warn!(
                "Tournament {} not running (status: {}), skipping blind advance",
                tournament_id,
                tournament.status
            );
            return Ok(false);
        }

        // Load blind levels
        let blind_levels = self.ctx.load_blind_levels(tournament_id).await?;
        let next_level = match next_blind_level_index(
            tournament.current_blind_level as usize,
            blind_levels.len(),
        ) {
            Some(value) => value,
            None => {
                // No more levels, keep current
                tracing::warn!(
                    "Tournament {} at max blind level {}, no more levels",
                    tournament_id,
                    tournament.current_blind_level
                );
                return Ok(false);
            }
        };

        // Update tournament
        tournament.current_blind_level += 1;
        tournament.level_start_time = Some(Utc::now().to_rfc3339());

        tracing::info!(
            "Tournament {} advancing from level {} to level {}",
            tournament_id,
            tournament.current_blind_level - 1,
            tournament.current_blind_level
        );

        sqlx::query(
            "UPDATE tournaments SET current_blind_level = ?, level_start_time = ? WHERE id = ?",
        )
        .bind(tournament.current_blind_level)
        .bind(&tournament.level_start_time)
        .bind(tournament_id)
        .execute(&*self.ctx.pool)
        .await?;

        tracing::info!(
            "Tournament {} database updated with new level",
            tournament_id
        );

        // Update in-memory cache if it exists
        if let Some(state) = self.ctx.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
            tracing::info!("Tournament {} in-memory cache updated", tournament_id);
        }

        let new_level = &blind_levels[next_level];
        tracing::info!(
            "Tournament {} advanced to level {}: {}/{}",
            tournament_id,
            next_level + 1,
            new_level.small_blind,
            new_level.big_blind
        );

        // Update blinds on all active tournament tables
        let tournament_tables: Vec<(String,)> = sqlx::query_as(
            "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1",
        )
        .bind(tournament_id)
        .fetch_all(&*self.ctx.pool)
        .await?;

        for (table_id,) in tournament_tables {
            self.ctx
                .game_server
                .update_table_blinds(&table_id, new_level.small_blind, new_level.big_blind, new_level.ante)
                .await;
        }

        // Broadcast blind level increase event
        use crate::ws::messages::ServerMessage;
        self.ctx
            .game_server
            .broadcast_tournament_event(
                tournament_id,
                ServerMessage::TournamentBlindLevelIncreased {
                    tournament_id: tournament_id.to_string(),
                    level: tournament.current_blind_level as i64,
                    small_blind: new_level.small_blind,
                    big_blind: new_level.big_blind,
                    ante: new_level.ante,
                },
            )
            .await;

        Ok(true)
    }

    /// Check all running tournaments for blind level advancement
    /// Called periodically by background task
    pub(crate) async fn check_all_blind_levels(&self) -> Result<()> {
        // Get all running tournaments
        let tournaments: Vec<Tournament> = sqlx::query_as(
            "SELECT * FROM tournaments WHERE status = 'running' AND level_start_time IS NOT NULL",
        )
        .fetch_all(&*self.ctx.pool)
        .await?;

        for tournament in tournaments {
            if let Some(level_start_str) = &tournament.level_start_time {
                // Parse the level start time and convert to UTC
                if let Ok(level_start) = DateTime::parse_from_rfc3339(level_start_str) {
                    let now = Utc::now();
                    let level_start_utc = level_start.with_timezone(&Utc);
                    let elapsed_secs = (now - level_start_utc).num_seconds();

                    tracing::info!(
                        "CHECK BLINDS - Tournament {} level {} - elapsed: {}s / duration: {}s",
                        tournament.id,
                        tournament.current_blind_level,
                        elapsed_secs,
                        tournament.level_duration_secs
                    );

                    // Check if it's time to advance
                    if elapsed_secs >= tournament.level_duration_secs {
                        tracing::warn!(
                            "ADVANCING NOW - Tournament {} level {} expired after {}s (duration: {}s)",
                            tournament.id,
                            tournament.current_blind_level,
                            elapsed_secs,
                            tournament.level_duration_secs
                        );

                        if let Err(e) = self.advance_blind_level(&tournament.id).await {
                            tracing::error!(
                                "Failed to advance blind level for tournament {}: {:?}",
                                tournament.id,
                                e
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

fn next_blind_level_index(current_level: usize, total_levels: usize) -> Option<usize> {
    if total_levels == 0 {
        return None;
    }

    let next_level = current_level + 1;
    if next_level >= total_levels {
        None
    } else {
        Some(next_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_blind_level_index_handles_bounds() {
        assert_eq!(next_blind_level_index(0, 3), Some(1));
        assert_eq!(next_blind_level_index(2, 3), None);
        assert_eq!(next_blind_level_index(0, 0), None);
    }
}
