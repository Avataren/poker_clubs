use crate::{
    db::models::{Tournament, TournamentRegistration},
    error::{AppError, Result},
    game::{constants::DEFAULT_MAX_SEATS, format::BlindSchedule},
    tournament::prizes::{PrizeStructure, PrizeWinner},
};
use chrono::Utc;
use std::collections::HashSet;
use std::sync::Arc;

use super::{
    context::{TournamentContext, TournamentState},
    manager::{MttConfig, SngConfig},
};

pub(crate) struct LifecycleService {
    ctx: Arc<TournamentContext>,
}

impl LifecycleService {
    pub(crate) fn new(ctx: Arc<TournamentContext>) -> Self {
        Self { ctx }
    }

    /// Create a new Sit & Go tournament
    pub(crate) async fn create_sng(&self, club_id: &str, config: SngConfig) -> Result<Tournament> {
        let min_players = normalize_min_players(config.min_players, config.max_players)?;

        // Create tournament record
        let tournament = Tournament::new(
            club_id.to_string(),
            config.name,
            "sng".to_string(),
            config.variant_id,
            config.buy_in,
            config.starting_stack,
            config.max_players,
            min_players,
            config.level_duration_secs,
            0,
            config.allow_rebuys,
            config.max_rebuys,
            config.rebuy_amount,
            config.rebuy_stack,
            config.allow_addons,
            config.max_addons,
            config.addon_amount,
            config.addon_stack,
            config.late_registration_secs,
        );

        // Generate blind schedule
        let starting_bb = config.starting_stack / 100; // Starting BB = 1% of stack
        let blind_schedule =
            BlindSchedule::standard_tournament(starting_bb, config.level_duration_secs as u64);

        // Save tournament to database
        self.ctx.save_tournament(&tournament).await?;

        // Save blind levels
        self.ctx
            .save_blind_levels(&tournament.id, &blind_schedule.levels)
            .await?;

        // Add to in-memory state
        let prize_structure = PrizeStructure::for_player_count(config.max_players);
        self.ctx.tournaments.write().await.insert(
            tournament.id.clone(),
            TournamentState {
                tournament: tournament.clone(),
                blind_schedule,
                prize_structure,
            },
        );

        tracing::info!(
            "Created SNG tournament: {} ({})",
            tournament.name,
            tournament.id
        );
        Ok(tournament)
    }

    /// Create a new Multi-Table Tournament
    pub(crate) async fn create_mtt(&self, club_id: &str, config: MttConfig) -> Result<Tournament> {
        let min_players = normalize_min_players(config.min_players, config.max_players)?;

        let mut tournament = Tournament::new(
            club_id.to_string(),
            config.name,
            "mtt".to_string(),
            config.variant_id,
            config.buy_in,
            config.starting_stack,
            config.max_players,
            min_players,
            config.level_duration_secs,
            config.pre_seat_secs,
            config.allow_rebuys,
            config.max_rebuys,
            config.rebuy_amount,
            config.rebuy_stack,
            config.allow_addons,
            config.max_addons,
            config.addon_amount,
            config.addon_stack,
            config.late_registration_secs,
        );

        tournament.scheduled_start = config.scheduled_start;

        // Generate blind schedule
        let starting_bb = config.starting_stack / 100;
        let blind_schedule =
            BlindSchedule::standard_tournament(starting_bb, config.level_duration_secs as u64);

        // Save to database
        self.ctx.save_tournament(&tournament).await?;
        self.ctx
            .save_blind_levels(&tournament.id, &blind_schedule.levels)
            .await?;

        // Add to in-memory state
        let prize_structure = PrizeStructure::for_player_count(config.max_players);
        self.ctx.tournaments.write().await.insert(
            tournament.id.clone(),
            TournamentState {
                tournament: tournament.clone(),
                blind_schedule,
                prize_structure,
            },
        );

        tracing::info!(
            "Created MTT tournament: {} ({})",
            tournament.name,
            tournament.id
        );
        Ok(tournament)
    }

    /// Start a tournament
    pub(crate) async fn start_tournament(&self, tournament_id: &str) -> Result<()> {
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;
        if tournament.status == "running" {
            return Ok(());
        }
        if tournament.status == "cancelled" {
            let reason = tournament
                .cancel_reason
                .clone()
                .unwrap_or_else(|| "Tournament was cancelled".to_string());
            return Err(AppError::BadRequest(reason));
        }
        self.ctx
            .start_tournament_with_state(tournament_id, &mut tournament)
            .await?;

        // Create table(s) and seat players
        if tournament.format_id == "sng" {
            self.ctx.start_sng_table(&tournament).await?;
        } else {
            self.ctx.start_mtt_tables(&tournament).await?;
        }

        tracing::info!(
            "Started tournament: {} ({})",
            tournament.name,
            tournament_id
        );
        Ok(())
    }

    /// Cancel a tournament manually with a reason.
    pub(crate) async fn cancel_tournament(
        &self,
        tournament_id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;
        let reason = reason.unwrap_or("Tournament cancelled by admin");
        self.ctx
            .cancel_tournament_with_state(&mut tournament, reason)
            .await
    }

    /// Handle player elimination
    pub(crate) async fn eliminate_player(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<Option<Vec<PrizeWinner>>> {
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;

        if tournament.status != "running" {
            return Ok(None);
        }

        let mut tx = self.ctx.pool.begin().await?;

        // Update remaining players (prevent underflow) and fetch the new value atomically.
        let remaining_players: i32 = sqlx::query_scalar(
            "UPDATE tournaments
             SET remaining_players = CASE
                 WHEN remaining_players > 0 THEN remaining_players - 1
                 ELSE 0
             END
             WHERE id = ?
             RETURNING remaining_players",
        )
        .bind(tournament_id)
        .fetch_one(&mut *tx)
        .await?;

        let finish_position = remaining_players + 1;

        // Record elimination within the same transaction.
        sqlx::query(
            "UPDATE tournament_registrations
             SET eliminated_at = ?, finish_position = ?
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(finish_position)
        .bind(tournament_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        tournament.remaining_players = remaining_players;

        // Get username and potential prize
        let username = self.ctx.get_username(user_id).await?;
        let prize = if let Some(state) = self.ctx.tournaments.read().await.get(tournament_id) {
            state
                .prize_structure
                .prize_for_position(finish_position, tournament.prize_pool)
        } else {
            0
        };

        tracing::info!(
            "Player {} eliminated from tournament {} at position {}",
            user_id,
            tournament_id,
            finish_position
        );

        // Broadcast player elimination event
        use crate::ws::messages::ServerMessage;
        self.ctx
            .game_server
            .broadcast_tournament_event(
                tournament_id,
                ServerMessage::TournamentPlayerEliminated {
                    tournament_id: tournament_id.to_string(),
                    username,
                    position: finish_position as i64,
                    prize,
                },
            )
            .await;

        // Check if tournament is finished
        if remaining_players <= 1 {
            return Ok(Some(self.finish_tournament(tournament_id).await?));
        }

        Ok(None)
    }

    /// Finish tournament and distribute prizes
    async fn finish_tournament(&self, tournament_id: &str) -> Result<Vec<PrizeWinner>> {
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;

        tournament.status = "finished".to_string();
        tournament.finished_at = Some(Utc::now().to_rfc3339());

        sqlx::query("UPDATE tournaments SET status = ?, finished_at = ? WHERE id = ?")
            .bind(&tournament.status)
            .bind(&tournament.finished_at)
            .bind(tournament_id)
            .execute(&*self.ctx.pool)
            .await?;

        // Mark the winner (last remaining player) as position 1
        sqlx::query(
            "UPDATE tournament_registrations 
             SET finish_position = 1 
             WHERE tournament_id = ? AND finish_position IS NULL",
        )
        .bind(tournament_id)
        .execute(&*self.ctx.pool)
        .await?;

        // Distribute prizes
        let winners = self.distribute_prizes(tournament_id).await?;

        tracing::info!(
            "Tournament {} finished with {} winners",
            tournament_id,
            winners.len()
        );

        // Broadcast tournament finished event
        use crate::ws::messages::{ServerMessage, TournamentWinner};
        let winner_info: Vec<TournamentWinner> = winners
            .iter()
            .map(|w| TournamentWinner {
                username: w.username.clone(),
                position: w.position as i64,
                prize: w.prize_amount,
            })
            .collect();

        self.ctx
            .game_server
            .broadcast_tournament_event(
                tournament_id,
                ServerMessage::TournamentFinished {
                    tournament_id: tournament_id.to_string(),
                    tournament_name: tournament.name.clone(),
                    winners: winner_info,
                },
            )
            .await;

        Ok(winners)
    }

    /// Distribute prizes to winners (atomic transaction)
    async fn distribute_prizes(&self, tournament_id: &str) -> Result<Vec<PrizeWinner>> {
        let tournament = self.ctx.load_tournament(tournament_id).await?;

        // Get prize structure
        let prize_structure = if let Some(state) = self.ctx.tournaments.read().await.get(tournament_id)
        {
            state.prize_structure.clone()
        } else {
            PrizeStructure::for_player_count(tournament.max_players)
        };

        // Get all registrations ordered by finish position
        let registrations = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations
             WHERE tournament_id = ?
             ORDER BY COALESCE(finish_position, 999999)",
        )
        .bind(tournament_id)
        .fetch_all(&*self.ctx.pool)
        .await?;

        // Wrap entire prize distribution in a single transaction
        let mut tx = self.ctx.pool.begin().await?;

        let mut winners = Vec::new();

        for registration in registrations {
            if let Some(position) = registration.finish_position {
                let prize = prize_structure.prize_for_position(position, tournament.prize_pool);

                if prize > 0 {
                    // Update registration with prize
                    sqlx::query(
                        "UPDATE tournament_registrations SET prize_amount = ?
                         WHERE tournament_id = ? AND user_id = ?",
                    )
                    .bind(prize)
                    .bind(tournament_id)
                    .bind(&registration.user_id)
                    .execute(&mut *tx)
                    .await?;

                    // Credit balance
                    self.ctx
                        .credit_prize_tx(&mut tx, &tournament.club_id, &registration.user_id, prize)
                        .await?;

                    // Load username
                    let username = self
                        .ctx
                        .get_username_tx(&mut tx, &registration.user_id)
                        .await?;

                    winners.push(PrizeWinner {
                        user_id: registration.user_id,
                        username,
                        position,
                        prize_amount: prize,
                    });
                }
            }
        }

        tx.commit().await?;

        Ok(winners)
    }

    /// Check all tournaments with a scheduled start time and trigger
    /// `refresh_tournament_timing()` so they auto-start or auto-cancel.
    /// Called periodically by background task.
    pub(crate) async fn check_scheduled_starts(&self) -> Result<()> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT id FROM tournaments
             WHERE status IN ('registering', 'seating')
               AND scheduled_start IS NOT NULL",
        )
        .fetch_all(&*self.ctx.pool)
        .await?;

        for (tournament_id,) in rows {
            if let Err(e) = self.ctx.load_tournament(&tournament_id).await {
                tracing::error!(
                    "Error refreshing scheduled tournament {}: {:?}",
                    tournament_id,
                    e
                );
            }
        }

        Ok(())
    }

    /// Check all tournament tables for player eliminations
    /// Called periodically by background task
    pub(crate) async fn check_tournament_eliminations(&self) -> Result<()> {
        // Get all active tournament tables
        let tables: Vec<(String, String)> = sqlx::query_as(
            "SELECT tournament_id, table_id FROM tournament_tables WHERE is_active = 1",
        )
        .fetch_all(&*self.ctx.pool)
        .await?;

        let mut tournaments_with_eliminations = HashSet::new();

        for (tournament_id, table_id) in tables {
            // Check if this table has any eliminations
            if let Some((_, eliminated_users)) =
                self.ctx.game_server.check_table_eliminations(&table_id).await
            {
                // Process each elimination
                for user_id in eliminated_users {
                    tracing::info!(
                        "Player {} eliminated from tournament {}",
                        user_id,
                        tournament_id
                    );

                    // Record the elimination and check if tournament is finished
                    match self.eliminate_player(&tournament_id, &user_id).await {
                        Ok(Some(prizes)) => {
                            tracing::info!(
                                "Tournament {} finished! {} prize winners",
                                tournament_id,
                                prizes.len()
                            );
                        }
                        Ok(None) => {
                            // Elimination recorded, tournament continues
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to process elimination for player {} in tournament {}: {:?}",
                                user_id,
                                tournament_id,
                                e
                            );
                        }
                    }
                }

                tournaments_with_eliminations.insert(tournament_id);
            }
        }

        // Balance tables for tournaments that had eliminations
        for tournament_id in tournaments_with_eliminations {
            if let Err(e) = self.balance_tournament_tables(&tournament_id).await {
                tracing::error!(
                    "Failed to balance tables for tournament {}: {:?}",
                    tournament_id,
                    e
                );
            }
        }

        Ok(())
    }

    /// Balance player counts across tournament tables.
    /// Closes empty tables, consolidates 1-player tables, and rebalances uneven tables.
    async fn balance_tournament_tables(&self, tournament_id: &str) -> Result<()> {
        // Get active tournament tables
        let table_rows: Vec<(String,)> = sqlx::query_as(
            "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1",
        )
        .bind(tournament_id)
        .fetch_all(&*self.ctx.pool)
        .await?;

        let table_ids: Vec<String> = table_rows.into_iter().map(|(id,)| id).collect();
        if table_ids.len() <= 1 {
            return Ok(());
        }

        let mut counts = self
            .ctx
            .game_server
            .get_table_player_counts(&table_ids)
            .await;

        // Close empty tables
        let empty_tables: Vec<String> = counts
            .iter()
            .filter(|(_, c)| *c == 0)
            .map(|(id, _)| id.clone())
            .collect();
        for table_id in &empty_tables {
            sqlx::query(
                "UPDATE tournament_tables SET is_active = 0 WHERE tournament_id = ? AND table_id = ?",
            )
            .bind(tournament_id)
            .bind(table_id)
            .execute(&*self.ctx.pool)
            .await?;
            tracing::info!(
                "Closed empty tournament table {} in tournament {}",
                table_id,
                tournament_id
            );
        }
        counts.retain(|(_, c)| *c > 0);

        // Final table consolidation: if total remaining players fit on one table, merge all into one
        let total_players: usize = counts.iter().map(|(_, c)| *c).sum();
        if counts.len() > 1 && total_players <= DEFAULT_MAX_SEATS {
            // Pick the table with the most players as the final table
            let final_table_id = counts.iter().max_by_key(|(_, c)| *c).unwrap().0.clone();

            // Collect all other tables
            let other_tables: Vec<String> = counts
                .iter()
                .filter(|(id, _)| *id != final_table_id)
                .map(|(id, _)| id.clone())
                .collect();

            // Move all players from other tables to the final table
            for source_table_id in &other_tables {
                let player_ids = self
                    .ctx
                    .game_server
                    .get_all_player_ids_at_table(source_table_id)
                    .await;

                for player_id in &player_ids {
                    if let Err(e) = self
                        .ctx
                        .game_server
                        .move_tournament_player(source_table_id, &final_table_id, player_id)
                        .await
                    {
                        tracing::warn!(
                            "Failed to move player {} to final table: {}",
                            player_id,
                            e
                        );
                        continue;
                    }

                    // Update DB registration
                    sqlx::query("UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?")
                        .bind(&final_table_id)
                        .bind(tournament_id)
                        .bind(player_id)
                        .execute(&*self.ctx.pool)
                        .await?;
                }

                // Close the now-empty source table
                sqlx::query("UPDATE tournament_tables SET is_active = 0 WHERE tournament_id = ? AND table_id = ?")
                    .bind(tournament_id)
                    .bind(source_table_id)
                    .execute(&*self.ctx.pool)
                    .await?;
            }

            tracing::info!(
                "Final table formed for tournament {} — {} players consolidated to table {}",
                tournament_id,
                total_players,
                final_table_id
            );

            return Ok(());
        }

        if counts.len() <= 1 {
            return Ok(());
        }

        // Consolidate 1-player tables: move the lone player elsewhere and close the table
        loop {
            // Find a table with exactly 1 player
            let lone_table = counts.iter().find(|(_, c)| *c == 1).map(|(id, _)| id.clone());
            let lone_table_id = match lone_table {
                Some(id) => id,
                None => break,
            };

            // Find another table with room (not the lone table itself)
            let dest = counts
                .iter()
                .filter(|(id, _)| *id != lone_table_id)
                .min_by_key(|(_, c)| *c)
                .map(|(id, _)| id.clone());
            let dest_table_id = match dest {
                Some(id) => id,
                None => break,
            };

            // Get the lone player
            let player_id = match self
                .ctx
                .game_server
                .get_last_player_at_table(&lone_table_id)
                .await
            {
                Some(id) => id,
                None => break,
            };

            // Move them
            if let Err(e) = self
                .ctx
                .game_server
                .move_tournament_player(&lone_table_id, &dest_table_id, &player_id)
                .await
            {
                tracing::warn!(
                    "Failed to consolidate player {} from table {}: {}",
                    player_id,
                    lone_table_id,
                    e
                );
                break;
            }

            // Update registration
            sqlx::query(
                "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?",
            )
            .bind(&dest_table_id)
            .bind(tournament_id)
            .bind(&player_id)
            .execute(&*self.ctx.pool)
            .await?;

            // Close the now-empty table
            sqlx::query(
                "UPDATE tournament_tables SET is_active = 0 WHERE tournament_id = ? AND table_id = ?",
            )
            .bind(tournament_id)
            .bind(&lone_table_id)
            .execute(&*self.ctx.pool)
            .await?;
            tracing::info!(
                "Consolidated lone player from table {} to {} in tournament {}",
                lone_table_id,
                dest_table_id,
                tournament_id
            );

            // Update local counts
            if let Some(pos) = counts.iter().position(|(id, _)| *id == dest_table_id) {
                counts[pos].1 += 1;
            }
            counts.retain(|(id, _)| *id != lone_table_id);

            if counts.len() <= 1 {
                return Ok(());
            }
        }

        // Rebalance loop: while max - min >= 2, move one player from max to min
        loop {
            let max_entry = counts.iter().max_by_key(|(_, c)| *c).cloned();
            let min_entry = counts.iter().min_by_key(|(_, c)| *c).cloned();

            let (max_id, max_count) = match max_entry {
                Some(e) => e,
                None => break,
            };
            let (min_id, min_count) = match min_entry {
                Some(e) => e,
                None => break,
            };

            if (max_count as i64) - (min_count as i64) < 2 {
                break;
            }

            // Pick the last player from the most populated table
            let player_id = match self
                .ctx
                .game_server
                .get_last_player_at_table(&max_id)
                .await
            {
                Some(id) => id,
                None => break,
            };

            if let Err(e) = self
                .ctx
                .game_server
                .move_tournament_player(&max_id, &min_id, &player_id)
                .await
            {
                tracing::warn!(
                    "Failed to balance player {} from table {} to {}: {}",
                    player_id,
                    max_id,
                    min_id,
                    e
                );
                break;
            }

            // Update registration
            sqlx::query(
                "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?",
            )
            .bind(&min_id)
            .bind(tournament_id)
            .bind(&player_id)
            .execute(&*self.ctx.pool)
            .await?;

            tracing::info!(
                "Rebalanced player {} from table {} ({}) to table {} ({}) in tournament {}",
                player_id,
                max_id,
                max_count,
                min_id,
                min_count,
                tournament_id
            );

            // Update local counts
            if let Some(pos) = counts.iter().position(|(id, _)| *id == max_id) {
                counts[pos].1 -= 1;
            }
            if let Some(pos) = counts.iter().position(|(id, _)| *id == min_id) {
                counts[pos].1 += 1;
            }
        }

        Ok(())
    }
}

fn normalize_min_players(min_players: i32, max_players: i32) -> Result<i32> {
    let min_players = if min_players <= 0 { 2 } else { min_players };

    if min_players < 2 || min_players > max_players {
        return Err(AppError::BadRequest(
            "Minimum players must be between 2 and max players".to_string(),
        ));
    }

    Ok(min_players)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{db, ws::GameServer};
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn normalize_min_players_defaults_to_two() {
        assert_eq!(normalize_min_players(0, 8).unwrap(), 2);
    }

    #[test]
    fn normalize_min_players_rejects_out_of_bounds() {
        assert!(normalize_min_players(1, 8).is_err());
        assert!(normalize_min_players(10, 8).is_err());
    }

    #[tokio::test]
    async fn concurrent_eliminations_assign_unique_positions() {
        let db_path = std::env::temp_dir().join(format!(
            "tournament_concurrency_{}.sqlite",
            Uuid::new_v4()
        ));
        let db_options = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(db_options)
            .await
            .expect("Failed to create database");

        db::run_migrations(&pool)
            .await
            .expect("Failed to run migrations");

        let jwt_manager = Arc::new(crate::auth::JwtManager::new("test_secret".to_string()));
        let game_server = Arc::new(GameServer::new(jwt_manager, Arc::new(pool.clone())));
        let ctx = Arc::new(TournamentContext::new(Arc::new(pool.clone()), game_server));
        let service = LifecycleService::new(ctx);

        let club_id = Uuid::new_v4().to_string();
        let admin_id = Uuid::new_v4().to_string();
        let player_two = Uuid::new_v4().to_string();
        let player_three = Uuid::new_v4().to_string();
        let player_four = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        for (user_id, username) in [
            (&admin_id, "admin_user"),
            (&player_two, "player_two"),
            (&player_three, "player_three"),
            (&player_four, "player_four"),
        ] {
            sqlx::query(
                "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            )
            .bind(user_id)
            .bind(username)
            .bind(format!("{}@example.com", username))
            .bind("password")
            .execute(&pool)
            .await
            .unwrap();
        }

        sqlx::query("INSERT INTO clubs (id, name, admin_id) VALUES (?, ?, ?)")
            .bind(&club_id)
            .bind("Concurrency Club")
            .bind(&admin_id)
            .execute(&pool)
            .await
            .unwrap();

        let tournament_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO tournaments (
                id, club_id, name, format_id, variant_id, buy_in, starting_stack, prize_pool,
                max_players, min_players, registered_players, remaining_players, current_blind_level,
                level_duration_secs, level_start_time, status, scheduled_start, pre_seat_secs,
                actual_start, finished_at, cancel_reason, allow_rebuys, max_rebuys, rebuy_amount,
                rebuy_stack, allow_addons, max_addons, addon_amount, addon_stack, late_registration_secs,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&tournament_id)
        .bind(&club_id)
        .bind("Concurrency Tournament")
        .bind("sng")
        .bind("holdem")
        .bind(100_i64)
        .bind(1500_i64)
        .bind(400_i64)
        .bind(4_i32)
        .bind(2_i32)
        .bind(4_i32)
        .bind(4_i32)
        .bind(0_i32)
        .bind(60_i64)
        .bind(&now)
        .bind("running")
        .bind::<Option<String>>(None)
        .bind(0_i64)
        .bind(&now)
        .bind::<Option<String>>(None)
        .bind::<Option<String>>(None)
        .bind(0_i32)
        .bind(0_i32)
        .bind(0_i64)
        .bind(0_i32)
        .bind(0_i32)
        .bind(0_i64)
        .bind(0_i64)
        .bind(0_i32)
        .bind(0_i32)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

        for user_id in [&admin_id, &player_two, &player_three, &player_four] {
            sqlx::query(
                "INSERT INTO tournament_registrations (tournament_id, user_id) VALUES (?, ?)",
            )
            .bind(&tournament_id)
            .bind(user_id)
            .execute(&pool)
            .await
            .unwrap();
        }

        let elimination_one = service.eliminate_player(&tournament_id, &player_two);
        let elimination_two = service.eliminate_player(&tournament_id, &player_three);

        let (result_one, result_two) = tokio::join!(elimination_one, elimination_two);
        result_one.unwrap();
        result_two.unwrap();

        let position_one: i32 = sqlx::query_scalar(
            "SELECT finish_position FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(&tournament_id)
        .bind(&player_two)
        .fetch_one(&pool)
        .await
        .unwrap();

        let position_two: i32 = sqlx::query_scalar(
            "SELECT finish_position FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(&tournament_id)
        .bind(&player_three)
        .fetch_one(&pool)
        .await
        .unwrap();

        let mut positions = vec![position_one, position_two];
        positions.sort();
        assert_eq!(positions, vec![3, 4]);

        let remaining_players: i32 = sqlx::query_scalar(
            "SELECT remaining_players FROM tournaments WHERE id = ?",
        )
        .bind(&tournament_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(remaining_players, 2);
    }
}
