use crate::{
    db::{
        models::{Tournament, TournamentBlindLevel, TournamentRegistration},
        DbPool,
    },
    error::{AppError, Result},
    game::format::{BlindLevel, BlindSchedule},
    tournament::prizes::PrizeStructure,
    ws::GameServer,
};
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

/// In-memory state for a running tournament
pub(crate) struct TournamentState {
    pub(crate) tournament: Tournament,
    pub(crate) blind_schedule: BlindSchedule,
    pub(crate) prize_structure: PrizeStructure,
}

pub(crate) struct TournamentContext {
    pub(crate) pool: Arc<DbPool>,
    pub(crate) game_server: Arc<GameServer>,
    pub(crate) tournaments: Arc<RwLock<HashMap<String, TournamentState>>>,
}

impl TournamentContext {
    pub(crate) fn new(pool: Arc<DbPool>, game_server: Arc<GameServer>) -> Self {
        Self {
            pool,
            game_server,
            tournaments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub(crate) async fn save_tournament(&self, tournament: &Tournament) -> Result<()> {
        sqlx::query(
            "INSERT INTO tournaments (
                id, club_id, name, format_id, variant_id, buy_in, starting_stack,
                prize_pool, max_players, min_players, registered_players, remaining_players,
                current_blind_level, level_duration_secs, level_start_time,
                status, scheduled_start, pre_seat_secs, actual_start, finished_at, cancel_reason,
                allow_rebuys, max_rebuys, rebuy_amount, rebuy_stack,
                allow_addons, max_addons, addon_amount, addon_stack, late_registration_secs,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&tournament.id)
        .bind(&tournament.club_id)
        .bind(&tournament.name)
        .bind(&tournament.format_id)
        .bind(&tournament.variant_id)
        .bind(tournament.buy_in)
        .bind(tournament.starting_stack)
        .bind(tournament.prize_pool)
        .bind(tournament.max_players)
        .bind(tournament.min_players)
        .bind(tournament.registered_players)
        .bind(tournament.remaining_players)
        .bind(tournament.current_blind_level)
        .bind(tournament.level_duration_secs)
        .bind(&tournament.level_start_time)
        .bind(&tournament.status)
        .bind(&tournament.scheduled_start)
        .bind(tournament.pre_seat_secs)
        .bind(&tournament.actual_start)
        .bind(&tournament.finished_at)
        .bind(&tournament.cancel_reason)
        .bind(tournament.allow_rebuys)
        .bind(tournament.max_rebuys)
        .bind(tournament.rebuy_amount)
        .bind(tournament.rebuy_stack)
        .bind(tournament.allow_addons)
        .bind(tournament.max_addons)
        .bind(tournament.addon_amount)
        .bind(tournament.addon_stack)
        .bind(tournament.late_registration_secs)
        .bind(&tournament.created_at)
        .execute(&*self.pool)
        .await?;

        Ok(())
    }

    pub(crate) async fn load_registration(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<TournamentRegistration> {
        sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(tournament_id)
        .bind(user_id)
        .fetch_optional(&*self.pool)
        .await?
        .ok_or_else(|| AppError::BadRequest("Player is not registered".to_string()))
    }

    pub(crate) async fn save_blind_levels(
        &self,
        tournament_id: &str,
        levels: &[BlindLevel],
    ) -> Result<()> {
        for (i, level) in levels.iter().enumerate() {
            let blind_level = TournamentBlindLevel::new(
                tournament_id.to_string(),
                i as i32,
                level.small_blind,
                level.big_blind,
                level.ante,
            );

            sqlx::query(
                "INSERT INTO tournament_blind_levels (tournament_id, level_number, small_blind, big_blind, ante) 
                 VALUES (?, ?, ?, ?, ?)",
            )
            .bind(&blind_level.tournament_id)
            .bind(blind_level.level_number)
            .bind(blind_level.small_blind)
            .bind(blind_level.big_blind)
            .bind(blind_level.ante)
            .execute(&*self.pool)
            .await?;
        }

        Ok(())
    }

    pub(crate) async fn load_tournament(&self, tournament_id: &str) -> Result<Tournament> {
        let mut tournament =
            sqlx::query_as::<_, Tournament>("SELECT * FROM tournaments WHERE id = ?")
                .bind(tournament_id)
                .fetch_one(&*self.pool)
                .await
                .map_err(|e| match e {
                    sqlx::Error::RowNotFound => {
                        AppError::NotFound("Tournament not found".to_string())
                    }
                    _ => AppError::Database(e),
                })?;

        self.refresh_tournament_timing(&mut tournament).await?;

        Ok(tournament)
    }

    pub(crate) async fn refresh_tournament_timing(
        &self,
        tournament: &mut Tournament,
    ) -> Result<()> {
        let scheduled_start = match tournament.scheduled_start.as_ref() {
            Some(value) => value,
            None => return Ok(()),
        };

        let scheduled_start = match DateTime::parse_from_rfc3339(scheduled_start) {
            Ok(value) => value.with_timezone(&Utc),
            Err(_) => return Ok(()),
        };

        let pre_seat_at = scheduled_start - Duration::seconds(tournament.pre_seat_secs.max(0));
        let now = Utc::now();

        if tournament.status == "registering" && now >= pre_seat_at && now < scheduled_start {
            tournament.status = "seating".to_string();
            sqlx::query("UPDATE tournaments SET status = ? WHERE id = ?")
                .bind(&tournament.status)
                .bind(&tournament.id)
                .execute(&*self.pool)
                .await?;
        }

        if (tournament.status == "registering" || tournament.status == "seating")
            && now >= scheduled_start
        {
            if tournament.registered_players >= tournament.min_players {
                let tournament_id = tournament.id.clone();
                self.start_tournament_with_state(&tournament_id, tournament)
                    .await?;
                if tournament.format_id == "sng" {
                    self.start_sng_table(tournament).await?;
                } else {
                    self.start_mtt_tables(tournament).await?;
                }
            } else {
                self.cancel_tournament_with_state(
                    tournament,
                    "Tournament cancelled due to insufficient players",
                )
                .await?;
            }
        }

        Ok(())
    }

    pub(crate) async fn start_tournament_with_state(
        &self,
        tournament_id: &str,
        tournament: &mut Tournament,
    ) -> Result<()> {
        if tournament.status != "registering" && tournament.status != "seating" {
            return Err(AppError::BadRequest(
                "Tournament already started or finished".to_string(),
            ));
        }

        if tournament.registered_players < tournament.min_players {
            return Err(AppError::BadRequest(format!(
                "Need at least {} players to start",
                tournament.min_players
            )));
        }

        // Update tournament status
        tournament.status = "running".to_string();
        tournament.actual_start = Some(Utc::now().to_rfc3339());
        tournament.level_start_time = Some(Utc::now().to_rfc3339());
        tournament.remaining_players = tournament.registered_players;

        sqlx::query(
            "UPDATE tournaments 
             SET status = ?, actual_start = ?, level_start_time = ?, remaining_players = ? 
             WHERE id = ?",
        )
        .bind(&tournament.status)
        .bind(&tournament.actual_start)
        .bind(&tournament.level_start_time)
        .bind(tournament.remaining_players)
        .bind(tournament_id)
        .execute(&*self.pool)
        .await?;

        Ok(())
    }

    pub(crate) async fn cancel_tournament_with_state(
        &self,
        tournament: &mut Tournament,
        reason: &str,
    ) -> Result<()> {
        if tournament.status == "cancelled" {
            return Ok(());
        }

        if tournament.status == "finished" {
            return Err(AppError::BadRequest(
                "Tournament already finished".to_string(),
            ));
        }

        let is_running = tournament.status == "running" || tournament.status == "paused";

        // Get all registrations
        let registrations = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ?",
        )
        .bind(&tournament.id)
        .fetch_all(&*self.pool)
        .await?;

        // If tournament is running, deactivate all tables
        if is_running {
            tracing::info!(
                "Deactivating all tables for running tournament {}",
                tournament.id
            );

            // Deactivate all tournament tables
            sqlx::query("UPDATE tournament_tables SET is_active = 0 WHERE tournament_id = ?")
                .bind(&tournament.id)
                .execute(&*self.pool)
                .await?;
        }

        // Wrap refunds + status update in a transaction
        sqlx::query("BEGIN IMMEDIATE")
            .execute(&*self.pool)
            .await?;

        // Refund all registered players first
        for registration in &registrations {
            if let Err(e) = self
                .refund_buy_in(
                    &tournament.club_id,
                    &registration.user_id,
                    tournament.buy_in,
                )
                .await
            {
                let _ = sqlx::query("ROLLBACK").execute(&*self.pool).await;
                return Err(e);
            }
        }

        tournament.status = "cancelled".to_string();
        tournament.cancel_reason = Some(reason.to_string());
        tournament.finished_at = Some(Utc::now().to_rfc3339());
        tournament.prize_pool = 0;
        tournament.remaining_players = 0;

        if let Err(e) = sqlx::query(
            "UPDATE tournaments SET status = ?, cancel_reason = ?, finished_at = ?, prize_pool = ?, remaining_players = ? WHERE id = ?",
        )
        .bind(&tournament.status)
        .bind(&tournament.cancel_reason)
        .bind(&tournament.finished_at)
        .bind(tournament.prize_pool)
        .bind(tournament.remaining_players)
        .bind(&tournament.id)
        .execute(&*self.pool)
        .await
        {
            let _ = sqlx::query("ROLLBACK").execute(&*self.pool).await;
            return Err(e.into());
        }

        sqlx::query("COMMIT")
            .execute(&*self.pool)
            .await?;

        if let Some(state) = self.tournaments.write().await.get_mut(&tournament.id) {
            state.tournament = tournament.clone();
        }

        tracing::info!("Cancelled tournament {}: {}", tournament.id, reason);

        // Broadcast tournament cancelled event
        use crate::ws::messages::ServerMessage;
        self.game_server
            .broadcast_tournament_event(
                &tournament.id,
                ServerMessage::TournamentCancelled {
                    tournament_id: tournament.id.clone(),
                    tournament_name: tournament.name.clone(),
                    reason: reason.to_string(),
                },
            )
            .await;

        Ok(())
    }

    pub(crate) async fn load_blind_levels(
        &self,
        tournament_id: &str,
    ) -> Result<Vec<TournamentBlindLevel>> {
        Ok(sqlx::query_as::<_, TournamentBlindLevel>(
            "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? ORDER BY level_number",
        )
        .bind(tournament_id)
        .fetch_all(&*self.pool)
        .await?)
    }

    pub(crate) async fn seat_tournament_player(
        &self,
        tournament: &Tournament,
        user_id: &str,
        username: &str,
        stack: i64,
    ) -> Result<()> {
        let table_ids: Vec<(String,)> = sqlx::query_as(
            "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1 ORDER BY table_number",
        )
        .bind(&tournament.id)
        .fetch_all(&*self.pool)
        .await?;

        for (table_id,) in table_ids {
            match self
                .game_server
                .add_player_to_table_next_seat(&table_id, user_id, username, stack)
                .await
            {
                Ok(seat) => {
                    sqlx::query(
                        "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?",
                    )
                    .bind(&table_id)
                    .bind(&tournament.id)
                    .bind(user_id)
                    .execute(&*self.pool)
                    .await?;

                    tracing::info!(
                        "Seated late entrant {} at table {} seat {} for tournament {}",
                        user_id,
                        table_id,
                        seat,
                        tournament.id
                    );

                    return Ok(());
                }
                Err(crate::game::error::GameError::TableFull) => continue,
                Err(err) => {
                    return Err(AppError::BadRequest(format!(
                        "Failed to seat player: {:?}",
                        err
                    )));
                }
            }
        }

        Err(AppError::BadRequest(
            "No available tournament seats for late entry".to_string(),
        ))
    }

    pub(crate) async fn apply_addon_chips(
        &self,
        registration: &TournamentRegistration,
        addon_stack: i64,
    ) -> Result<()> {
        let table_id = registration.starting_table_id.as_ref().ok_or_else(|| {
            AppError::BadRequest("Player is not seated at a tournament table".to_string())
        })?;

        self.game_server
            .tournament_top_up(table_id, &registration.user_id, addon_stack)
            .await
            .map_err(|e| AppError::BadRequest(format!("Failed to apply add-on: {:?}", e)))?;

        Ok(())
    }

    pub(crate) async fn ensure_late_registration_seat_available(
        &self,
        tournament: &Tournament,
    ) -> Result<()> {
        let table_ids: Vec<(String,)> = sqlx::query_as(
            "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1 ORDER BY table_number",
        )
        .bind(&tournament.id)
        .fetch_all(&*self.pool)
        .await?;

        for (table_id,) in table_ids {
            if self
                .game_server
                .can_seat_tournament_player(&table_id)
                .await
                .is_ok()
            {
                return Ok(());
            }
        }

        Err(AppError::BadRequest(
            "No available tournament seats for late registration".to_string(),
        ))
    }

    pub(crate) async fn ensure_rebuy_seat_available(&self, tournament: &Tournament) -> Result<()> {
        let table_ids: Vec<(String,)> = sqlx::query_as(
            "SELECT table_id FROM tournament_tables WHERE tournament_id = ? AND is_active = 1 ORDER BY table_number",
        )
        .bind(&tournament.id)
        .fetch_all(&*self.pool)
        .await?;

        for (table_id,) in table_ids {
            if self
                .game_server
                .can_seat_tournament_player(&table_id)
                .await
                .is_ok()
            {
                return Ok(());
            }
        }

        Err(AppError::BadRequest(
            "No available tournament seats for rebuy".to_string(),
        ))
    }

    pub(crate) async fn ensure_addon_seated(
        &self,
        registration: &TournamentRegistration,
    ) -> Result<()> {
        let table_id = registration.starting_table_id.as_ref().ok_or_else(|| {
            AppError::BadRequest("Player is not seated at a tournament table".to_string())
        })?;

        self.game_server
            .can_tournament_top_up(table_id, &registration.user_id)
            .await
            .map_err(|e| AppError::BadRequest(format!("Add-on unavailable: {:?}", e)))?;

        Ok(())
    }

    pub(crate) async fn rollback_rebuy(
        &self,
        tournament: &Tournament,
        user_id: &str,
        rebuy_amount: i64,
        previous_rebuys: i32,
        previous_eliminated_at: Option<String>,
        previous_finish_position: Option<i32>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE tournaments SET prize_pool = MAX(prize_pool - ?, 0), remaining_players = MAX(remaining_players - 1, 0) WHERE id = ?",
        )
        .bind(rebuy_amount)
        .bind(&tournament.id)
        .execute(&*self.pool)
        .await?;

        sqlx::query(
            "UPDATE tournament_registrations 
             SET rebuys = ?, eliminated_at = ?, finish_position = ?
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(previous_rebuys)
        .bind(previous_eliminated_at)
        .bind(previous_finish_position)
        .bind(&tournament.id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        if let Some(state) = self.tournaments.write().await.get_mut(&tournament.id) {
            let mut updated = tournament.clone();
            updated.prize_pool = (updated.prize_pool - rebuy_amount).max(0);
            updated.remaining_players = (updated.remaining_players - 1).max(0);
            state.tournament = updated;
        }

        self.refund_buy_in(&tournament.club_id, user_id, rebuy_amount)
            .await?;

        Ok(())
    }

    pub(crate) async fn rollback_late_registration(
        &self,
        tournament: &Tournament,
        user_id: &str,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE tournaments
             SET registered_players = MAX(registered_players - 1, 0),
                 prize_pool = MAX(prize_pool - ?, 0),
                 remaining_players = MAX(remaining_players - 1, 0)
             WHERE id = ?",
        )
        .bind(tournament.buy_in)
        .bind(&tournament.id)
        .execute(&*self.pool)
        .await?;

        sqlx::query("DELETE FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?")
            .bind(&tournament.id)
            .bind(user_id)
            .execute(&*self.pool)
            .await?;

        if let Some(state) = self.tournaments.write().await.get_mut(&tournament.id) {
            let mut updated = tournament.clone();
            updated.registered_players = (updated.registered_players - 1).max(0);
            updated.prize_pool = (updated.prize_pool - updated.buy_in).max(0);
            updated.remaining_players = (updated.remaining_players - 1).max(0);
            state.tournament = updated;
        }

        self.refund_buy_in(&tournament.club_id, user_id, tournament.buy_in)
            .await?;

        Ok(())
    }

    pub(crate) async fn rollback_addon(
        &self,
        tournament: &Tournament,
        user_id: &str,
        addon_amount: i64,
    ) -> Result<()> {
        sqlx::query("UPDATE tournaments SET prize_pool = prize_pool - ? WHERE id = ?")
            .bind(addon_amount)
            .bind(&tournament.id)
            .execute(&*self.pool)
            .await?;

        sqlx::query(
            "UPDATE tournament_registrations 
             SET addons = CASE WHEN addons > 0 THEN addons - 1 ELSE 0 END 
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(&tournament.id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        if let Some(state) = self.tournaments.write().await.get_mut(&tournament.id) {
            let mut updated = tournament.clone();
            updated.prize_pool = (updated.prize_pool - addon_amount).max(0);
            state.tournament = updated;
        }

        self.refund_buy_in(&tournament.club_id, user_id, addon_amount)
            .await?;

        Ok(())
    }

    pub(crate) async fn deduct_buy_in(
        &self,
        club_id: &str,
        user_id: &str,
        amount: i64,
    ) -> Result<()> {
        // Check if this is a bot user - bots don't need balance
        let (is_bot,): (bool,) = sqlx::query_as("SELECT is_bot FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_one(&*self.pool)
            .await?;

        if is_bot {
            // Bots bypass balance requirements
            return Ok(());
        }

        let result = sqlx::query(
            "UPDATE club_members SET balance = balance - ? WHERE club_id = ? AND user_id = ? AND balance >= ?",
        )
        .bind(amount)
        .bind(club_id)
        .bind(user_id)
        .bind(amount)
        .execute(&*self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(AppError::BadRequest("Insufficient balance".to_string()));
        }

        Ok(())
    }

    pub(crate) async fn refund_buy_in(
        &self,
        club_id: &str,
        user_id: &str,
        amount: i64,
    ) -> Result<()> {
        // Check if this is a bot user - bots don't have club membership
        let (is_bot,): (bool,) = sqlx::query_as("SELECT is_bot FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_one(&*self.pool)
            .await?;

        if is_bot {
            // Bots bypass refund logic
            return Ok(());
        }

        sqlx::query(
            "UPDATE club_members SET balance = balance + ? WHERE club_id = ? AND user_id = ?",
        )
        .bind(amount)
        .bind(club_id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        Ok(())
    }

    pub(crate) async fn credit_prize(
        &self,
        club_id: &str,
        user_id: &str,
        amount: i64,
    ) -> Result<()> {
        // Check if this is a bot user - bots don't have club membership
        let (is_bot,): (bool,) = sqlx::query_as("SELECT is_bot FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_one(&*self.pool)
            .await?;

        if is_bot {
            // Bots bypass prize credit logic
            return Ok(());
        }

        sqlx::query(
            "UPDATE club_members SET balance = balance + ? WHERE club_id = ? AND user_id = ?",
        )
        .bind(amount)
        .bind(club_id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        Ok(())
    }

    pub(crate) async fn get_username(&self, user_id: &str) -> Result<String> {
        let result: (String,) = sqlx::query_as("SELECT username FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_one(&*self.pool)
            .await?;

        Ok(result.0)
    }

    pub(crate) async fn start_sng_table(&self, tournament: &Tournament) -> Result<()> {
        use crate::game::{format::SitAndGo, variant::variant_from_id};
        use uuid::Uuid;

        // Get all registered players
        let registrations: Vec<TournamentRegistration> = sqlx::query_as(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? ORDER BY registered_at",
        )
        .bind(&tournament.id)
        .fetch_all(&*self.pool)
        .await?;

        if registrations.is_empty() {
            return Err(AppError::BadRequest("No players registered".to_string()));
        }

        // Create table ID and name
        let table_id = Uuid::new_v4().to_string();
        let table_name = format!("{} - Table 1", tournament.name);

        // Get current blind level
        let blind_levels = self.load_blind_levels(&tournament.id).await?;
        let current_level = &blind_levels[tournament.current_blind_level as usize];

        // Create variant
        let variant = variant_from_id(&tournament.variant_id).ok_or_else(|| {
            AppError::BadRequest(format!("Invalid variant: {}", tournament.variant_id))
        })?;

        // Create SNG format
        let format = SitAndGo::new(
            tournament.buy_in,
            tournament.starting_stack,
            tournament.max_players as usize,
            tournament.level_duration_secs as u64,
        );

        // Insert table into database first
        sqlx::query(
            "INSERT INTO tables (id, club_id, name, small_blind, big_blind, min_buyin, max_buyin, max_players, variant_id, format_id, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&table_id)
        .bind(&tournament.club_id)
        .bind(&table_name)
        .bind(current_level.small_blind)
        .bind(current_level.big_blind)
        .bind(tournament.starting_stack)
        .bind(tournament.starting_stack)
        .bind(tournament.max_players)
        .bind(&tournament.variant_id)
        .bind(&tournament.format_id)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&*self.pool)
        .await?;

        // Create the table in-memory
        self.game_server
            .create_table_with_options(
                table_id.clone(),
                table_name,
                current_level.small_blind,
                current_level.big_blind,
                variant,
                Box::new(format),
            )
            .await;

        // Add all registered players to the table
        for (seat, registration) in registrations.iter().enumerate() {
            let username = self.get_username(&registration.user_id).await?;

            // Check if this user is a bot
            let is_bot: bool = sqlx::query_scalar("SELECT is_bot FROM users WHERE id = ?")
                .bind(&registration.user_id)
                .fetch_one(&*self.pool)
                .await
                .unwrap_or(false);

            // Add player to table via game server
            if let Err(e) = self
                .game_server
                .add_player_to_table(
                    &table_id,
                    registration.user_id.clone(),
                    username.clone(),
                    seat,
                    tournament.starting_stack,
                )
                .await
            {
                tracing::error!("Failed to seat player {}: {:?}", registration.user_id, e);
            }

            // Register bots with the bot manager
            if is_bot {
                tracing::info!(
                    "Registering bot {} for tournament table {}",
                    username,
                    table_id
                );
                self.game_server
                    .register_bot(&table_id, registration.user_id.clone(), username, None)
                    .await;
            }

            // Update registration with table assignment
            sqlx::query(
                "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?",
            )
            .bind(&table_id)
            .bind(&tournament.id)
            .bind(&registration.user_id)
            .execute(&*self.pool)
            .await?;
        }

        // Link table to tournament
        sqlx::query(
            "INSERT INTO tournament_tables (tournament_id, table_id, table_number, is_active) VALUES (?, ?, ?, 1)",
        )
        .bind(&tournament.id)
        .bind(&table_id)
        .bind(1)
        .execute(&*self.pool)
        .await?;

        // Set tournament ID on the table
        self.game_server
            .set_table_tournament(&table_id, tournament.id.clone())
            .await;

        // Force start the hand now that all players are seated
        self.game_server.force_start_table_hand(&table_id).await;

        // Broadcast tournament started event
        use crate::ws::messages::ServerMessage;
        self.game_server
            .broadcast_tournament_event(
                &tournament.id,
                ServerMessage::TournamentStarted {
                    tournament_id: tournament.id.clone(),
                    tournament_name: tournament.name.clone(),
                    table_id: Some(table_id.clone()),
                },
            )
            .await;

        tracing::info!(
            "Created SNG table {} for tournament {} with {} players and started first hand",
            table_id,
            tournament.id,
            registrations.len()
        );

        Ok(())
    }

    pub(crate) async fn start_mtt_tables(&self, tournament: &Tournament) -> Result<()> {
        use crate::game::{
            constants::DEFAULT_MAX_SEATS, format::MultiTableTournament, variant::variant_from_id,
        };
        use uuid::Uuid;

        // Get all registered players
        let registrations: Vec<TournamentRegistration> = sqlx::query_as(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? ORDER BY registered_at",
        )
        .bind(&tournament.id)
        .fetch_all(&*self.pool)
        .await?;

        if registrations.is_empty() {
            return Err(AppError::BadRequest("No players registered".to_string()));
        }

        let player_count = registrations.len();

        // Calculate number of tables needed (max 9 players per table)
        let players_per_table = DEFAULT_MAX_SEATS;
        let table_count = (player_count + players_per_table - 1) / players_per_table;

        // Get current blind level
        let blind_levels = self.load_blind_levels(&tournament.id).await?;
        let current_level = &blind_levels[tournament.current_blind_level as usize];

        // Create variant
        let variant_id = tournament.variant_id.clone();

        // Distribute players across tables
        let mut player_index = 0;
        for table_num in 0..table_count {
            let table_id = Uuid::new_v4().to_string();
            let table_name = format!("{} - Table {}", tournament.name, table_num + 1);

            // Create MTT format
            let format = MultiTableTournament::new(
                table_name.clone(),
                tournament.buy_in,
                tournament.starting_stack,
                tournament.level_duration_secs as u64,
            );

            // Insert table into database first
            sqlx::query(
                "INSERT INTO tables (id, club_id, name, small_blind, big_blind, min_buyin, max_buyin, max_players, variant_id, format_id, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&table_id)
            .bind(&tournament.club_id)
            .bind(&table_name)
            .bind(current_level.small_blind)
            .bind(current_level.big_blind)
            .bind(tournament.starting_stack)
            .bind(tournament.starting_stack)
            .bind(DEFAULT_MAX_SEATS as i32)
            .bind(&variant_id)
            .bind(&tournament.format_id)
            .bind(chrono::Utc::now().to_rfc3339())
            .execute(&*self.pool)
            .await?;

            // Create the table in-memory
            let variant = variant_from_id(&variant_id)
                .ok_or_else(|| AppError::BadRequest(format!("Invalid variant: {}", variant_id)))?;
            self.game_server
                .create_table_with_options(
                    table_id.clone(),
                    table_name,
                    current_level.small_blind,
                    current_level.big_blind,
                    variant,
                    Box::new(format),
                )
                .await;

            // Seat players at this table
            let mut seat = 0;
            while player_index < player_count && seat < players_per_table {
                let registration = &registrations[player_index];
                let username = self.get_username(&registration.user_id).await?;

                // Check if this user is a bot
                let is_bot: bool = sqlx::query_scalar("SELECT is_bot FROM users WHERE id = ?")
                    .bind(&registration.user_id)
                    .fetch_one(&*self.pool)
                    .await
                    .unwrap_or(false);

                // Add player to table
                if let Err(e) = self
                    .game_server
                    .add_player_to_table(
                        &table_id,
                        registration.user_id.clone(),
                        username.clone(),
                        seat,
                        tournament.starting_stack,
                    )
                    .await
                {
                    tracing::error!("Failed to seat player {}: {:?}", registration.user_id, e);
                }

                // Register bots with the bot manager
                if is_bot {
                    tracing::info!("Registering bot {} for MTT table {}", username, table_id);
                    self.game_server
                        .register_bot(&table_id, registration.user_id.clone(), username, None)
                        .await;
                }

                // Update registration with table assignment
                sqlx::query(
                    "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?",
                )
                .bind(&table_id)
                .bind(&tournament.id)
                .bind(&registration.user_id)
                .execute(&*self.pool)
                .await?;

                player_index += 1;
                seat += 1;
            }

            // Link table to tournament
            sqlx::query(
                "INSERT INTO tournament_tables (tournament_id, table_id, table_number, is_active) VALUES (?, ?, ?, 1)",
            )
            .bind(&tournament.id)
            .bind(&table_id)
            .bind((table_num + 1) as i32)
            .execute(&*self.pool)
            .await?;

            // Set tournament ID on the table
            self.game_server
                .set_table_tournament(&table_id, tournament.id.clone())
                .await;

            // Force start the hand now that all players are seated at this table
            self.game_server.force_start_table_hand(&table_id).await;

            tracing::info!(
                "Created MTT table {} (#{}) for tournament {} with {} players and started first hand",
                table_id,
                table_num + 1,
                tournament.id,
                seat
            );
        }

        tracing::info!(
            "Created {} tables for MTT {} with {} total players",
            table_count,
            tournament.id,
            player_count
        );

        // Broadcast tournament started event
        use crate::ws::messages::ServerMessage;
        self.game_server
            .broadcast_tournament_event(
                &tournament.id,
                ServerMessage::TournamentStarted {
                    tournament_id: tournament.id.clone(),
                    tournament_name: tournament.name.clone(),
                    table_id: None, // MTT has multiple tables
                },
            )
            .await;

        Ok(())
    }

    /// Remove finished/cancelled tournaments from in-memory cache after 1 hour.
    pub(crate) async fn cleanup_finished_tournaments(&self) {
        let now = Utc::now();
        let one_hour = Duration::hours(1);

        let mut tournaments = self.tournaments.write().await;
        let before = tournaments.len();

        tournaments.retain(|_id, state| {
            let status = &state.tournament.status;
            if status == "finished" || status == "cancelled" {
                // Check finished_at timestamp
                if let Some(ref finished_str) = state.tournament.finished_at {
                    if let Ok(finished_at) = NaiveDateTime::parse_from_str(finished_str, "%Y-%m-%d %H:%M:%S") {
                        let finished_utc = finished_at.and_utc();
                        if now - finished_utc > one_hour {
                            tracing::info!(
                                "Cleaning up finished tournament {} (finished at {})",
                                state.tournament.id,
                                finished_str
                            );
                            return false; // Remove from cache
                        }
                    }
                }
            }
            true // Keep in cache
        });

        let removed = before - tournaments.len();
        if removed > 0 {
            tracing::info!("Cleaned up {} finished tournaments from memory", removed);
        }
    }
}
