use crate::{
    db::models::{Tournament, TournamentRegistration},
    error::{AppError, Result},
    game::{constants::DEFAULT_MAX_SEATS, format::BlindSchedule},
    tournament::prizes::{PrizeStructure, PrizeWinner},
};
use chrono::Utc;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    context::{TournamentContext, TournamentState},
    manager::{MttConfig, SngConfig},
};

pub(crate) struct LifecycleService {
    ctx: Arc<TournamentContext>,
    pending_moves: Arc<RwLock<HashMap<String, Vec<PendingTableMove>>>>,
}

#[derive(Debug, Clone)]
struct PendingTableMove {
    source_table_id: String,
    user_id: String,
}

impl LifecycleService {
    pub(crate) fn new(ctx: Arc<TournamentContext>) -> Self {
        Self {
            ctx,
            pending_moves: Arc::new(RwLock::new(HashMap::new())),
        }
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
        if tournament.status == "finished" {
            return Err(AppError::BadRequest(
                "Tournament has already finished".to_string(),
            ));
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
        let prize_structure =
            if let Some(state) = self.ctx.tournaments.read().await.get(tournament_id) {
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

    async fn enqueue_pending_move(
        &self,
        tournament_id: &str,
        source_table_id: &str,
        user_id: &str,
    ) -> bool {
        let mut pending = self.pending_moves.write().await;
        let queue = pending.entry(tournament_id.to_string()).or_default();
        if queue.iter().any(|m| m.user_id == user_id) {
            return false;
        }

        queue.push(PendingTableMove {
            source_table_id: source_table_id.to_string(),
            user_id: user_id.to_string(),
        });
        true
    }

    async fn update_registration_table(
        &self,
        tournament_id: &str,
        user_id: &str,
        table_id: &str,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(table_id)
        .bind(tournament_id)
        .bind(user_id)
        .execute(&*self.ctx.pool)
        .await?;
        Ok(())
    }

    async fn pick_destination_table(
        &self,
        source_table_id: &str,
        candidate_table_ids: &[String],
    ) -> Option<String> {
        let mut destinations = Vec::new();
        for table_id in candidate_table_ids {
            if table_id == source_table_id {
                continue;
            }
            if self
                .ctx
                .game_server
                .can_seat_tournament_player(table_id)
                .await
                .is_ok()
            {
                destinations.push(table_id.clone());
            }
        }

        if destinations.is_empty() {
            return None;
        }

        let dest_counts = self
            .ctx
            .game_server
            .get_table_player_counts(&destinations)
            .await;
        let min_dest_count = dest_counts.iter().map(|(_, c)| *c).min()?;
        let dest_candidates: Vec<String> = dest_counts
            .iter()
            .filter(|(_, c)| *c == min_dest_count)
            .map(|(id, _)| id.clone())
            .collect();
        let mut rng = rand::thread_rng();
        dest_candidates.choose(&mut rng).cloned()
    }

    async fn flush_pending_moves_for_tournament(&self, tournament_id: &str) -> Result<()> {
        let queue_snapshot = {
            let pending = self.pending_moves.read().await;
            pending.get(tournament_id).cloned().unwrap_or_default()
        };

        if queue_snapshot.is_empty() {
            return Ok(());
        }

        let table_rows: Vec<(String,)> = sqlx::query_as(
            "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1",
        )
        .bind(tournament_id)
        .fetch_all(&*self.ctx.pool)
        .await?;
        let active_table_ids: Vec<String> = table_rows.into_iter().map(|(id,)| id).collect();

        if active_table_ids.len() <= 1 {
            return Ok(());
        }

        let active_tables: HashSet<&str> = active_table_ids.iter().map(|id| id.as_str()).collect();
        let mut still_pending = Vec::new();

        for pending_move in queue_snapshot {
            if !active_tables.contains(pending_move.source_table_id.as_str()) {
                continue;
            }

            let source_players = self
                .ctx
                .game_server
                .get_all_player_ids_at_table(&pending_move.source_table_id)
                .await;
            if !source_players.iter().any(|id| id == &pending_move.user_id) {
                continue;
            }

            if self
                .ctx
                .game_server
                .is_table_mid_hand(&pending_move.source_table_id)
                .await
            {
                still_pending.push(pending_move);
                continue;
            }

            let dest_table_id = match self
                .pick_destination_table(&pending_move.source_table_id, &active_table_ids)
                .await
            {
                Some(id) => id,
                None => {
                    still_pending.push(pending_move);
                    continue;
                }
            };

            match self
                .ctx
                .game_server
                .move_tournament_player(
                    &pending_move.source_table_id,
                    &dest_table_id,
                    &pending_move.user_id,
                )
                .await
            {
                Ok(()) => {
                    self.update_registration_table(
                        tournament_id,
                        &pending_move.user_id,
                        &dest_table_id,
                    )
                    .await?;
                    tracing::info!(
                        "Applied deferred move for player {} from table {} to table {} in tournament {}",
                        pending_move.user_id,
                        pending_move.source_table_id,
                        dest_table_id,
                        tournament_id
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed deferred move for player {} from table {}: {}",
                        pending_move.user_id,
                        pending_move.source_table_id,
                        e
                    );
                    still_pending.push(pending_move);
                }
            }
        }

        let mut pending = self.pending_moves.write().await;
        if still_pending.is_empty() {
            pending.remove(tournament_id);
        } else {
            pending.insert(tournament_id.to_string(), still_pending);
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

        let mut active_tournaments = HashSet::new();
        let mut tournaments_with_eliminations = HashSet::new();

        for (tournament_id, table_id) in tables {
            active_tournaments.insert(tournament_id.clone());
            // Check if this table has any eliminations
            if let Some((_, eliminated_users)) = self
                .ctx
                .game_server
                .check_table_eliminations(&table_id)
                .await
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

        // First pass: try to execute previously deferred moves for all active tournaments.
        for tournament_id in &active_tournaments {
            if let Err(e) = self.flush_pending_moves_for_tournament(tournament_id).await {
                tracing::error!(
                    "Failed to flush deferred moves for tournament {}: {:?}",
                    tournament_id,
                    e
                );
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
    /// Closes empty tables, consolidates into the fewest tables possible, and
    /// rebalances so all tables have a similar player count.
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

        if counts.len() <= 1 {
            return Ok(());
        }

        // Consolidate tables: reduce to the minimum number of tables needed
        let total_players: usize = counts.iter().map(|(_, c)| *c).sum();
        let min_tables_needed = (total_players + DEFAULT_MAX_SEATS - 1) / DEFAULT_MAX_SEATS;
        let min_tables_needed = min_tables_needed.max(1);

        if min_tables_needed < counts.len() {
            let tables_to_close = counts.len() - min_tables_needed;
            let is_final_table = min_tables_needed == 1;

            // Sort ascending by player count so we close the smallest tables first
            counts.sort_by_key(|(_, c)| *c);

            // The tables to close are the first `tables_to_close` entries (smallest)
            let mut closing: Vec<String> = counts[..tables_to_close]
                .iter()
                .map(|(id, _)| id.clone())
                .collect();
            {
                let mut rng = rand::thread_rng();
                closing.shuffle(&mut rng);
            }

            // The tables that stay open are the rest
            let remaining: Vec<String> = counts[tables_to_close..]
                .iter()
                .map(|(id, _)| id.clone())
                .collect();

            for source_table_id in &closing {
                let mut player_ids = self
                    .ctx
                    .game_server
                    .get_all_player_ids_at_table(source_table_id)
                    .await;
                {
                    let mut rng = rand::thread_rng();
                    player_ids.shuffle(&mut rng);
                }

                // Defer all moves from tables currently in-hand.
                if self
                    .ctx
                    .game_server
                    .is_table_mid_hand(source_table_id)
                    .await
                {
                    let mut deferred_count = 0usize;
                    for player_id in &player_ids {
                        if self
                            .enqueue_pending_move(tournament_id, source_table_id, player_id)
                            .await
                        {
                            deferred_count += 1;
                        }
                    }

                    if deferred_count > 0 {
                        tracing::info!(
                            "Deferred {} consolidation move(s) from table {} in tournament {} until Waiting",
                            deferred_count,
                            source_table_id,
                            tournament_id
                        );
                    }
                    continue;
                }

                for player_id in &player_ids {
                    let dest_table_id = match self
                        .pick_destination_table(source_table_id, &remaining)
                        .await
                    {
                        Some(id) => id,
                        None => break,
                    };

                    match self
                        .ctx
                        .game_server
                        .move_tournament_player(source_table_id, &dest_table_id, player_id)
                        .await
                    {
                        Ok(()) => {
                            self.update_registration_table(
                                tournament_id,
                                player_id,
                                &dest_table_id,
                            )
                            .await?;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to move player {} from table {} to {}: {}",
                                player_id,
                                source_table_id,
                                dest_table_id,
                                e
                            );
                            if self
                                .enqueue_pending_move(tournament_id, source_table_id, player_id)
                                .await
                            {
                                tracing::info!(
                                    "Deferred consolidation move for player {} from table {}",
                                    player_id,
                                    source_table_id
                                );
                            }
                        }
                    }
                }

                // Close the source table if it is now empty.
                let source_count = self
                    .ctx
                    .game_server
                    .get_all_player_ids_at_table(source_table_id)
                    .await
                    .len();
                if source_count == 0 {
                    sqlx::query("UPDATE tournament_tables SET is_active = 0 WHERE tournament_id = ? AND table_id = ?")
                        .bind(tournament_id)
                        .bind(source_table_id)
                        .execute(&*self.ctx.pool)
                        .await?;
                }
            }

            if is_final_table {
                tracing::info!(
                    "Final table formed for tournament {} — {} players consolidated to {} table",
                    tournament_id,
                    total_players,
                    min_tables_needed
                );
            } else {
                tracing::info!(
                    "Consolidated tournament {} from {} tables to {} ({} players)",
                    tournament_id,
                    counts.len(),
                    min_tables_needed,
                    total_players
                );
            }

            // Refresh counts after consolidation
            let remaining_rows: Vec<(String,)> = sqlx::query_as(
                "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1",
            )
            .bind(tournament_id)
            .fetch_all(&*self.ctx.pool)
            .await?;
            let remaining_ids: Vec<String> = remaining_rows.into_iter().map(|(id,)| id).collect();
            counts = self
                .ctx
                .game_server
                .get_table_player_counts(&remaining_ids)
                .await;
        }

        if counts.len() <= 1 {
            return Ok(());
        }

        // Rebalance loop: while max - min >= 2, move one player from max to min
        loop {
            let max_count = match counts.iter().map(|(_, c)| *c).max() {
                Some(c) => c,
                None => break,
            };
            let min_count = match counts.iter().map(|(_, c)| *c).min() {
                Some(c) => c,
                None => break,
            };

            if (max_count as i64) - (min_count as i64) < 2 {
                break;
            }

            // Pick a random donor from all most-populated tables.
            let donor_candidates: Vec<String> = counts
                .iter()
                .filter(|(_, count)| *count == max_count)
                .map(|(table_id, _)| table_id.clone())
                .collect();
            let max_id = {
                let mut rng = rand::thread_rng();
                match donor_candidates.choose(&mut rng) {
                    Some(id) => id.clone(),
                    None => break,
                }
            };

            // Pick a random recipient from all least-populated tables.
            let recipient_candidates: Vec<String> = counts
                .iter()
                .filter(|(id, c)| *c == min_count && id.as_str() != max_id.as_str())
                .map(|(id, _)| id.clone())
                .collect();
            let min_id = match self
                .pick_destination_table(&max_id, &recipient_candidates)
                .await
            {
                Some(id) => id,
                None => break,
            };

            // Pick a random player from the donor table.
            let player_ids = self
                .ctx
                .game_server
                .get_all_player_ids_at_table(&max_id)
                .await;
            let player_id = {
                let mut rng = rand::thread_rng();
                match player_ids.choose(&mut rng) {
                    Some(id) => id.clone(),
                    None => break,
                }
            };

            let move_applied = if self.ctx.game_server.is_table_mid_hand(&max_id).await {
                if self
                    .enqueue_pending_move(tournament_id, &max_id, &player_id)
                    .await
                {
                    tracing::info!(
                        "Deferred rebalance move for player {} from table {} to table {} in tournament {}",
                        player_id,
                        max_id,
                        min_id,
                        tournament_id
                    );
                    true
                } else {
                    false
                }
            } else {
                match self
                    .ctx
                    .game_server
                    .move_tournament_player(&max_id, &min_id, &player_id)
                    .await
                {
                    Ok(()) => {
                        self.update_registration_table(tournament_id, &player_id, &min_id)
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
                        true
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to balance player {} from table {} to {}: {}",
                            player_id,
                            max_id,
                            min_id,
                            e
                        );
                        self.enqueue_pending_move(tournament_id, &max_id, &player_id)
                            .await
                    }
                }
            };

            if !move_applied {
                break;
            }

            // Update local counts
            if let Some(pos) = counts.iter().position(|(id, _)| *id == max_id) {
                counts[pos].1 -= 1;
            }
            if let Some(pos) = counts.iter().position(|(id, _)| *id == min_id) {
                counts[pos].1 += 1;
            }
        }

        // Apply any deferred moves that may now be eligible.
        self.flush_pending_moves_for_tournament(tournament_id)
            .await?;

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
        let db_path =
            std::env::temp_dir().join(format!("tournament_concurrency_{}.sqlite", Uuid::new_v4()));
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

        let remaining_players: i32 =
            sqlx::query_scalar("SELECT remaining_players FROM tournaments WHERE id = ?")
                .bind(&tournament_id)
                .fetch_one(&pool)
                .await
                .unwrap();

        assert_eq!(remaining_players, 2);
    }

    #[tokio::test]
    async fn pending_moves_flush_when_source_table_is_waiting() {
        use crate::game::format::MultiTableTournament;
        use crate::game::variant::TexasHoldem;

        let pool = Arc::new(crate::create_test_db().await);
        let jwt_manager = Arc::new(crate::auth::JwtManager::new("test_secret".to_string()));
        let game_server = Arc::new(GameServer::new(jwt_manager, pool.clone()));
        let ctx = Arc::new(TournamentContext::new(pool.clone(), game_server.clone()));
        let service = LifecycleService::new(ctx);

        let club_id = Uuid::new_v4().to_string();
        let admin_id = Uuid::new_v4().to_string();
        let p1 = Uuid::new_v4().to_string();
        let p2 = Uuid::new_v4().to_string();
        let p3 = Uuid::new_v4().to_string();
        let tournament_id = Uuid::new_v4().to_string();
        let table_a = format!("table-{}", Uuid::new_v4());
        let table_b = format!("table-{}", Uuid::new_v4());
        let now = Utc::now().to_rfc3339();

        for (user_id, username) in [
            (&admin_id, "admin"),
            (&p1, "player_one"),
            (&p2, "player_two"),
            (&p3, "player_three"),
        ] {
            sqlx::query(
                "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            )
            .bind(user_id)
            .bind(username)
            .bind(format!("{}@example.com", username))
            .bind("password")
            .execute(&*pool)
            .await
            .unwrap();
        }

        sqlx::query("INSERT INTO clubs (id, name, admin_id) VALUES (?, ?, ?)")
            .bind(&club_id)
            .bind("Deferred Move Club")
            .bind(&admin_id)
            .execute(&*pool)
            .await
            .unwrap();

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
        .bind("Deferred Move Tournament")
        .bind("mtt")
        .bind("holdem")
        .bind(100_i64)
        .bind(1500_i64)
        .bind(0_i64)
        .bind(9_i32)
        .bind(2_i32)
        .bind(3_i32)
        .bind(3_i32)
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
        .bind(0_i64)
        .bind(0_i32)
        .bind(0_i32)
        .bind(0_i64)
        .bind(0_i64)
        .bind(0_i64)
        .bind(&now)
        .execute(&*pool)
        .await
        .unwrap();

        for (user_id, start_table) in [(&p1, &table_a), (&p2, &table_a), (&p3, &table_b)] {
            sqlx::query(
                "INSERT INTO tournament_registrations (tournament_id, user_id, starting_table_id) VALUES (?, ?, ?)",
            )
            .bind(&tournament_id)
            .bind(user_id)
            .bind(start_table)
            .execute(&*pool)
            .await
            .unwrap();
        }

        for (table_id, table_number) in [(&table_a, 1_i64), (&table_b, 2_i64)] {
            sqlx::query(
                "INSERT INTO tournament_tables (tournament_id, table_id, table_number, is_active) VALUES (?, ?, ?, 1)",
            )
            .bind(&tournament_id)
            .bind(table_id)
            .bind(table_number)
            .execute(&*pool)
            .await
            .unwrap();
        }

        let table_format = || {
            Box::new(MultiTableTournament::new(
                "Deferred Move".to_string(),
                100,
                1500,
                300,
            ))
        };
        game_server
            .create_table_with_options(
                table_a.clone(),
                "Table A".to_string(),
                50,
                100,
                Box::new(TexasHoldem),
                table_format(),
            )
            .await;
        game_server
            .create_table_with_options(
                table_b.clone(),
                "Table B".to_string(),
                50,
                100,
                Box::new(TexasHoldem),
                table_format(),
            )
            .await;
        game_server
            .set_table_tournament(&table_a, tournament_id.clone())
            .await;
        game_server
            .set_table_tournament(&table_b, tournament_id.clone())
            .await;

        game_server
            .add_player_to_table(&table_a, p1.clone(), "player_one".to_string(), 0, 1000)
            .await
            .unwrap();
        game_server
            .add_player_to_table(&table_a, p2.clone(), "player_two".to_string(), 1, 1000)
            .await
            .unwrap();
        game_server
            .add_player_to_table(&table_b, p3.clone(), "player_three".to_string(), 0, 1000)
            .await
            .unwrap();

        assert!(
            service
                .enqueue_pending_move(&tournament_id, &table_a, &p1)
                .await
        );
        service
            .flush_pending_moves_for_tournament(&tournament_id)
            .await
            .unwrap();

        let from_players = game_server.get_all_player_ids_at_table(&table_a).await;
        let to_players = game_server.get_all_player_ids_at_table(&table_b).await;
        assert!(!from_players.contains(&p1));
        assert!(to_players.contains(&p1));

        let new_start_table: Option<String> = sqlx::query_scalar(
            "SELECT starting_table_id FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(&tournament_id)
        .bind(&p1)
        .fetch_one(&*pool)
        .await
        .unwrap();
        assert_eq!(new_start_table.as_deref(), Some(table_b.as_str()));
    }
}
