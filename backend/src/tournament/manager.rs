//! Tournament Manager
//!
//! Centralized management for tournament lifecycle:
//! - Registration and unregistration
//! - Starting tournaments
//! - Blind level advancement
//! - Player elimination tracking
//! - Prize distribution

use crate::{
    db::{
        models::{Tournament, TournamentBlindLevel, TournamentRegistration},
        DbPool,
    },
    error::{AppError, Result},
    game::format::{BlindLevel, BlindSchedule},
    tournament::prizes::{PrizeStructure, PrizeWinner},
    ws::GameServer,
};
use chrono::{DateTime, Duration, Utc};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

/// Configuration for creating an SNG
#[derive(Debug, Clone)]
pub struct SngConfig {
    pub name: String,
    pub variant_id: String,
    pub buy_in: i64,
    pub starting_stack: i64,
    pub max_players: i32,
    pub min_players: i32,
    pub level_duration_secs: i64,
    pub allow_rebuys: bool,
    pub max_rebuys: i32,
    pub rebuy_amount: i64,
    pub rebuy_stack: i64,
    pub allow_addons: bool,
    pub max_addons: i32,
    pub addon_amount: i64,
    pub addon_stack: i64,
    pub late_registration_secs: i64,
}

/// Configuration for creating an MTT
#[derive(Debug, Clone)]
pub struct MttConfig {
    pub name: String,
    pub variant_id: String,
    pub buy_in: i64,
    pub starting_stack: i64,
    pub max_players: i32,
    pub min_players: i32,
    pub level_duration_secs: i64,
    pub scheduled_start: Option<String>,
    pub pre_seat_secs: i64,
    pub allow_rebuys: bool,
    pub max_rebuys: i32,
    pub rebuy_amount: i64,
    pub rebuy_stack: i64,
    pub allow_addons: bool,
    pub max_addons: i32,
    pub addon_amount: i64,
    pub addon_stack: i64,
    pub late_registration_secs: i64,
}

/// Manages all tournaments
pub struct TournamentManager {
    pool: Arc<DbPool>,
    game_server: Arc<GameServer>,
    /// Active tournament controllers (in-memory state)
    tournaments: Arc<RwLock<HashMap<String, TournamentState>>>,
}

/// In-memory state for a running tournament
struct TournamentState {
    tournament: Tournament,
    blind_schedule: BlindSchedule,
    prize_structure: PrizeStructure,
}

impl TournamentManager {
    pub fn new(pool: Arc<DbPool>, game_server: Arc<GameServer>) -> Self {
        Self {
            pool,
            game_server,
            tournaments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new Sit & Go tournament
    pub async fn create_sng(&self, club_id: &str, config: SngConfig) -> Result<Tournament> {
        let min_players = if config.min_players <= 0 {
            2
        } else {
            config.min_players
        };

        if min_players < 2 || min_players > config.max_players {
            return Err(AppError::BadRequest(
                "Minimum players must be between 2 and max players".to_string(),
            ));
        }

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
        self.save_tournament(&tournament).await?;

        // Save blind levels
        self.save_blind_levels(&tournament.id, &blind_schedule.levels)
            .await?;

        // Add to in-memory state
        let prize_structure = PrizeStructure::for_player_count(config.max_players);
        self.tournaments.write().await.insert(
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
    pub async fn create_mtt(&self, club_id: &str, config: MttConfig) -> Result<Tournament> {
        let min_players = if config.min_players <= 0 {
            2
        } else {
            config.min_players
        };

        if min_players < 2 || min_players > config.max_players {
            return Err(AppError::BadRequest(
                "Minimum players must be between 2 and max players".to_string(),
            ));
        }

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
        self.save_tournament(&tournament).await?;
        self.save_blind_levels(&tournament.id, &blind_schedule.levels)
            .await?;

        // Add to in-memory state
        let prize_structure = PrizeStructure::for_player_count(config.max_players);
        self.tournaments.write().await.insert(
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

    /// Register a player for a tournament
    pub async fn register_player(
        &self,
        tournament_id: &str,
        user_id: &str,
        username: &str,
    ) -> Result<()> {
        // Load tournament
        let mut tournament = self.load_tournament(tournament_id).await?;

        // Check status
        if tournament.status == "cancelled" {
            let reason = tournament
                .cancel_reason
                .clone()
                .unwrap_or_else(|| "Tournament was cancelled".to_string());
            return Err(AppError::BadRequest(reason));
        }

        if tournament.status == "running" && self.late_registration_open(&tournament) {
            // Allow late registration
        } else if tournament.status != "registering" && tournament.status != "seating" {
            return Err(AppError::BadRequest(
                "Tournament is not accepting registrations".to_string(),
            ));
        }

        // Check if full
        if tournament.registered_players >= tournament.max_players {
            return Err(AppError::BadRequest("Tournament is full".to_string()));
        }

        // Check if already registered
        let existing = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(tournament_id)
        .bind(user_id)
        .fetch_optional(&*self.pool)
        .await?;

        if existing.is_some() {
            return Err(AppError::BadRequest("Already registered".to_string()));
        }

        // Deduct buy-in from club balance
        self.deduct_buy_in(&tournament.club_id, user_id, tournament.buy_in)
            .await?;

        // Create registration
        let registration =
            TournamentRegistration::new(tournament_id.to_string(), user_id.to_string());

        sqlx::query(
            "INSERT INTO tournament_registrations (tournament_id, user_id, registered_at, prize_amount) 
             VALUES (?, ?, ?, ?)"
        )
        .bind(&registration.tournament_id)
        .bind(&registration.user_id)
        .bind(&registration.registered_at)
        .bind(registration.prize_amount)
        .execute(&*self.pool)
        .await?;

        // Update tournament counts and prize pool
        tournament.registered_players += 1;
        tournament.prize_pool += tournament.buy_in;
        if tournament.status == "running" {
            tournament.remaining_players += 1;
        }

        sqlx::query(
            "UPDATE tournaments 
             SET registered_players = ?, prize_pool = ?, remaining_players = ? 
             WHERE id = ?",
        )
        .bind(tournament.registered_players)
        .bind(tournament.prize_pool)
        .bind(tournament.remaining_players)
        .bind(tournament_id)
        .execute(&*self.pool)
        .await?;

        // Update in-memory state
        if let Some(state) = self.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        tracing::info!(
            "Player {} ({}) registered for tournament {}",
            username,
            user_id,
            tournament_id
        );

        if tournament.status == "running" {
            self.seat_tournament_player(&tournament, user_id, username, tournament.starting_stack)
                .await?;
        }

        // Check if SNG should auto-start
        if tournament.format_id == "sng" && tournament.registered_players >= tournament.max_players
        {
            self.start_tournament(tournament_id).await?;
        }

        Ok(())
    }

    /// Unregister a player from a tournament (before it starts)
    pub async fn unregister_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        let tournament = self.load_tournament(tournament_id).await?;

        if tournament.status == "cancelled" {
            return Err(AppError::BadRequest(
                "Tournament has been cancelled".to_string(),
            ));
        }

        if tournament.status != "registering" && tournament.status != "seating" {
            return Err(AppError::BadRequest(
                "Cannot unregister after tournament has started".to_string(),
            ));
        }

        // Check if registered
        let registration = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(tournament_id)
        .bind(user_id)
        .fetch_optional(&*self.pool)
        .await?;

        if registration.is_none() {
            return Err(AppError::BadRequest("Not registered".to_string()));
        }

        // Refund buy-in
        self.refund_buy_in(&tournament.club_id, user_id, tournament.buy_in)
            .await?;

        // Delete registration
        sqlx::query("DELETE FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?")
            .bind(tournament_id)
            .bind(user_id)
            .execute(&*self.pool)
            .await?;

        // Update tournament counts
        sqlx::query(
            "UPDATE tournaments 
             SET registered_players = registered_players - 1, prize_pool = prize_pool - ? 
             WHERE id = ?",
        )
        .bind(tournament.buy_in)
        .bind(tournament_id)
        .execute(&*self.pool)
        .await?;

        tracing::info!(
            "Player {} unregistered from tournament {}",
            user_id,
            tournament_id
        );
        Ok(())
    }

    /// Start a tournament
    pub async fn start_tournament(&self, tournament_id: &str) -> Result<()> {
        let mut tournament = self.load_tournament(tournament_id).await?;
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
        self.start_tournament_with_state(tournament_id, &mut tournament)
            .await?;

        // Create table(s) and seat players
        if tournament.format_id == "sng" {
            self.start_sng_table(&tournament).await?;
        } else {
            self.start_mtt_tables(&tournament).await?;
        }

        tracing::info!(
            "Started tournament: {} ({})",
            tournament.name,
            tournament_id
        );
        Ok(())
    }

    /// Cancel a tournament manually with a reason.
    pub async fn cancel_tournament(&self, tournament_id: &str, reason: Option<&str>) -> Result<()> {
        let mut tournament = self.load_tournament(tournament_id).await?;
        let reason = reason.unwrap_or("Tournament cancelled by admin");
        self.cancel_tournament_with_state(&mut tournament, reason)
            .await
    }

    /// Rebuy into a running tournament (if enabled)
    pub async fn rebuy_player(
        &self,
        tournament_id: &str,
        user_id: &str,
        username: &str,
    ) -> Result<()> {
        let mut tournament = self.load_tournament(tournament_id).await?;

        if !tournament.allow_rebuys {
            return Err(AppError::BadRequest(
                "Rebuys are not enabled for this tournament".to_string(),
            ));
        }

        if tournament.status != "running" {
            return Err(AppError::BadRequest(
                "Rebuys are only allowed while the tournament is running".to_string(),
            ));
        }

        if !self.late_registration_open(&tournament) {
            return Err(AppError::BadRequest("Rebuy period has ended".to_string()));
        }

        let mut registration = self.load_registration(tournament_id, user_id).await?;

        if registration.eliminated_at.is_none() {
            return Err(AppError::BadRequest(
                "Player must be eliminated before rebuying".to_string(),
            ));
        }

        if tournament.max_rebuys > 0 && registration.rebuys >= tournament.max_rebuys {
            return Err(AppError::BadRequest(
                "Rebuy limit reached for this tournament".to_string(),
            ));
        }

        let rebuy_amount = if tournament.rebuy_amount > 0 {
            tournament.rebuy_amount
        } else {
            tournament.buy_in
        };
        let rebuy_stack = if tournament.rebuy_stack > 0 {
            tournament.rebuy_stack
        } else {
            tournament.starting_stack
        };

        self.deduct_buy_in(&tournament.club_id, user_id, rebuy_amount)
            .await?;

        tournament.prize_pool += rebuy_amount;
        tournament.remaining_players += 1;

        sqlx::query("UPDATE tournaments SET prize_pool = ?, remaining_players = ? WHERE id = ?")
            .bind(tournament.prize_pool)
            .bind(tournament.remaining_players)
            .bind(tournament_id)
            .execute(&*self.pool)
            .await?;

        if let Some(state) = self.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        registration.rebuys += 1;
        registration.eliminated_at = None;
        registration.finish_position = None;

        sqlx::query(
            "UPDATE tournament_registrations 
             SET rebuys = ?, eliminated_at = NULL, finish_position = NULL 
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(registration.rebuys)
        .bind(tournament_id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        self.seat_tournament_player(&tournament, user_id, username, rebuy_stack)
            .await?;

        tracing::info!(
            "Player {} rebought into tournament {} (rebuys: {})",
            user_id,
            tournament_id,
            registration.rebuys
        );

        Ok(())
    }

    /// Add-on chips to a running tournament (if enabled)
    pub async fn addon_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        let mut tournament = self.load_tournament(tournament_id).await?;

        if !tournament.allow_addons {
            return Err(AppError::BadRequest(
                "Add-ons are not enabled for this tournament".to_string(),
            ));
        }

        if tournament.status != "running" {
            return Err(AppError::BadRequest(
                "Add-ons are only allowed while the tournament is running".to_string(),
            ));
        }

        let registration = self.load_registration(tournament_id, user_id).await?;

        if registration.eliminated_at.is_some() {
            return Err(AppError::BadRequest(
                "Eliminated players cannot take add-ons".to_string(),
            ));
        }

        if tournament.max_addons > 0 && registration.addons >= tournament.max_addons {
            return Err(AppError::BadRequest(
                "Add-on limit reached for this tournament".to_string(),
            ));
        }

        let addon_amount = if tournament.addon_amount > 0 {
            tournament.addon_amount
        } else {
            tournament.buy_in
        };
        let addon_stack = if tournament.addon_stack > 0 {
            tournament.addon_stack
        } else {
            tournament.starting_stack
        };

        self.deduct_buy_in(&tournament.club_id, user_id, addon_amount)
            .await?;

        tournament.prize_pool += addon_amount;

        sqlx::query("UPDATE tournaments SET prize_pool = ? WHERE id = ?")
            .bind(tournament.prize_pool)
            .bind(tournament_id)
            .execute(&*self.pool)
            .await?;

        if let Some(state) = self.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        sqlx::query(
            "UPDATE tournament_registrations 
             SET addons = addons + 1 
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(tournament_id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        self.apply_addon_chips(&registration, addon_stack).await?;

        tracing::info!(
            "Player {} took an add-on in tournament {}",
            user_id,
            tournament_id
        );

        Ok(())
    }

    /// Advance to the next blind level
    pub async fn advance_blind_level(&self, tournament_id: &str) -> Result<bool> {
        tracing::info!(
            "advance_blind_level called for tournament {}",
            tournament_id
        );
        let mut tournament = self.load_tournament(tournament_id).await?;

        if tournament.status != "running" {
            tracing::warn!(
                "Tournament {} not running (status: {}), skipping blind advance",
                tournament_id,
                tournament.status
            );
            return Ok(false);
        }

        // Load blind levels
        let blind_levels = self.load_blind_levels(tournament_id).await?;
        let next_level = (tournament.current_blind_level + 1) as usize;

        if next_level >= blind_levels.len() {
            // No more levels, keep current
            tracing::warn!(
                "Tournament {} at max blind level {}, no more levels",
                tournament_id,
                tournament.current_blind_level
            );
            return Ok(false);
        }

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
        .execute(&*self.pool)
        .await?;

        tracing::info!(
            "Tournament {} database updated with new level",
            tournament_id
        );

        // Update in-memory cache if it exists
        if let Some(state) = self.tournaments.write().await.get_mut(tournament_id) {
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
        .fetch_all(&*self.pool)
        .await?;

        for (table_id,) in tournament_tables {
            self.game_server
                .update_table_blinds(&table_id, new_level.small_blind, new_level.big_blind)
                .await;
        }

        // Broadcast blind level increase event
        use crate::ws::messages::ServerMessage;
        self.game_server
            .broadcast_tournament_event(
                tournament_id,
                ServerMessage::TournamentBlindLevelIncreased {
                    tournament_id: tournament_id.to_string(),
                    level: (tournament.current_blind_level + 1) as i64,
                    small_blind: new_level.small_blind,
                    big_blind: new_level.big_blind,
                    ante: new_level.ante,
                },
            )
            .await;

        Ok(true)
    }

    /// Handle player elimination
    pub async fn eliminate_player(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<Option<Vec<PrizeWinner>>> {
        let mut tournament = self.load_tournament(tournament_id).await?;

        if tournament.status != "running" {
            return Ok(None);
        }

        // Update remaining players
        tournament.remaining_players -= 1;
        let finish_position = tournament.remaining_players + 1;

        sqlx::query("UPDATE tournaments SET remaining_players = ? WHERE id = ?")
            .bind(tournament.remaining_players)
            .bind(tournament_id)
            .execute(&*self.pool)
            .await?;

        // Record elimination
        sqlx::query(
            "UPDATE tournament_registrations 
             SET eliminated_at = ?, finish_position = ? 
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(finish_position)
        .bind(tournament_id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        // Get username and potential prize
        let username = self.get_username(user_id).await?;
        let prize = if let Some(state) = self.tournaments.read().await.get(tournament_id) {
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
        self.game_server
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
        if tournament.remaining_players <= 1 {
            return Ok(Some(self.finish_tournament(tournament_id).await?));
        }

        Ok(None)
    }

    /// Finish tournament and distribute prizes
    async fn finish_tournament(&self, tournament_id: &str) -> Result<Vec<PrizeWinner>> {
        let mut tournament = self.load_tournament(tournament_id).await?;

        tournament.status = "finished".to_string();
        tournament.finished_at = Some(Utc::now().to_rfc3339());

        sqlx::query("UPDATE tournaments SET status = ?, finished_at = ? WHERE id = ?")
            .bind(&tournament.status)
            .bind(&tournament.finished_at)
            .bind(tournament_id)
            .execute(&*self.pool)
            .await?;

        // Mark the winner (last remaining player) as position 1
        sqlx::query(
            "UPDATE tournament_registrations 
             SET finish_position = 1 
             WHERE tournament_id = ? AND finish_position IS NULL",
        )
        .bind(tournament_id)
        .execute(&*self.pool)
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

        self.game_server
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

    /// Distribute prizes to winners
    async fn distribute_prizes(&self, tournament_id: &str) -> Result<Vec<PrizeWinner>> {
        let tournament = self.load_tournament(tournament_id).await?;

        // Get prize structure
        let prize_structure = if let Some(state) = self.tournaments.read().await.get(tournament_id)
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
        .fetch_all(&*self.pool)
        .await?;

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
                    .execute(&*self.pool)
                    .await?;

                    // Credit balance
                    self.credit_prize(&tournament.club_id, &registration.user_id, prize)
                        .await?;

                    // Load username
                    let username = self.get_username(&registration.user_id).await?;

                    winners.push(PrizeWinner {
                        user_id: registration.user_id,
                        username,
                        position,
                        prize_amount: prize,
                    });
                }
            }
        }

        Ok(winners)
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    async fn save_tournament(&self, tournament: &Tournament) -> Result<()> {
        sqlx::query(
            "INSERT INTO tournaments (
                id, club_id, name, format_id, variant_id, buy_in, starting_stack,
                prize_pool, max_players, min_players, registered_players, remaining_players,
                current_blind_level, level_duration_secs, level_start_time,
                status, scheduled_start, pre_seat_secs, actual_start, finished_at, cancel_reason,
                allow_rebuys, max_rebuys, rebuy_amount, rebuy_stack,
                allow_addons, max_addons, addon_amount, addon_stack, late_registration_secs,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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

    async fn load_registration(
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

    fn late_registration_open(&self, tournament: &Tournament) -> bool {
        if tournament.late_registration_secs <= 0 {
            return false;
        }

        let actual_start = match tournament.actual_start.as_ref() {
            Some(start) => start,
            None => return false,
        };

        let start_time = match DateTime::parse_from_rfc3339(actual_start) {
            Ok(value) => value.with_timezone(&Utc),
            Err(_) => return false,
        };

        let deadline = start_time + Duration::seconds(tournament.late_registration_secs);
        Utc::now() <= deadline
    }

    async fn seat_tournament_player(
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

    async fn apply_addon_chips(
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

    async fn save_blind_levels(&self, tournament_id: &str, levels: &[BlindLevel]) -> Result<()> {
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
                 VALUES (?, ?, ?, ?, ?)"
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

    async fn load_tournament(&self, tournament_id: &str) -> Result<Tournament> {
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

    async fn refresh_tournament_timing(&self, tournament: &mut Tournament) -> Result<()> {
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

    async fn start_tournament_with_state(
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

    async fn cancel_tournament_with_state(
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

        // Refund all registered players
        for registration in registrations {
            self.refund_buy_in(
                &tournament.club_id,
                &registration.user_id,
                tournament.buy_in,
            )
            .await?;
        }

        tournament.status = "cancelled".to_string();
        tournament.cancel_reason = Some(reason.to_string());
        tournament.finished_at = Some(Utc::now().to_rfc3339());
        tournament.prize_pool = 0;
        tournament.remaining_players = 0;

        sqlx::query(
            "UPDATE tournaments SET status = ?, cancel_reason = ?, finished_at = ?, prize_pool = ?, remaining_players = ? WHERE id = ?"
        )
        .bind(&tournament.status)
        .bind(&tournament.cancel_reason)
        .bind(&tournament.finished_at)
        .bind(tournament.prize_pool)
        .bind(tournament.remaining_players)
        .bind(&tournament.id)
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

    async fn load_blind_levels(&self, tournament_id: &str) -> Result<Vec<TournamentBlindLevel>> {
        Ok(sqlx::query_as::<_, TournamentBlindLevel>(
            "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? ORDER BY level_number",
        )
        .bind(tournament_id)
        .fetch_all(&*self.pool)
        .await?)
    }

    async fn deduct_buy_in(&self, club_id: &str, user_id: &str, amount: i64) -> Result<()> {
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
            "UPDATE club_members SET balance = balance - ? WHERE club_id = ? AND user_id = ? AND balance >= ?"
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

    async fn refund_buy_in(&self, club_id: &str, user_id: &str, amount: i64) -> Result<()> {
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

    async fn credit_prize(&self, club_id: &str, user_id: &str, amount: i64) -> Result<()> {
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

    async fn get_username(&self, user_id: &str) -> Result<String> {
        let result: (String,) = sqlx::query_as("SELECT username FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_one(&*self.pool)
            .await?;

        Ok(result.0)
    }

    async fn start_sng_table(&self, tournament: &Tournament) -> Result<()> {
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
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
                "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?"
            )
            .bind(&table_id)
            .bind(&tournament.id)
            .bind(&registration.user_id)
            .execute(&*self.pool)
            .await?;
        }

        // Link table to tournament
        sqlx::query(
            "INSERT INTO tournament_tables (tournament_id, table_id, table_number, is_active) VALUES (?, ?, ?, 1)"
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

    async fn start_mtt_tables(&self, tournament: &Tournament) -> Result<()> {
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
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
                    "UPDATE tournament_registrations SET starting_table_id = ? WHERE tournament_id = ? AND user_id = ?"
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
                "INSERT INTO tournament_tables (tournament_id, table_id, table_number, is_active) VALUES (?, ?, ?, 1)"
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

    /// Check all running tournaments for blind level advancement
    /// Called periodically by background task
    pub async fn check_all_blind_levels(&self) -> Result<()> {
        // Get all running tournaments
        let tournaments: Vec<Tournament> = sqlx::query_as(
            "SELECT * FROM tournaments WHERE status = 'running' AND level_start_time IS NOT NULL",
        )
        .fetch_all(&*self.pool)
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

    /// Check all tournament tables for player eliminations
    /// Called periodically by background task
    pub async fn check_tournament_eliminations(&self) -> Result<()> {
        // Get all active tournament tables
        let tables: Vec<(String, String)> = sqlx::query_as(
            "SELECT tournament_id, table_id FROM tournament_tables WHERE is_active = 1",
        )
        .fetch_all(&*self.pool)
        .await?;

        for (tournament_id, table_id) in tables {
            // Check if this table has any eliminations
            if let Some((_, eliminated_users)) =
                self.game_server.check_table_eliminations(&table_id).await
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
                            // TODO: Broadcast tournament finished message via WebSocket
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
            }
        }

        Ok(())
    }

    /// Broadcast tournament info to all players at tournament tables
    /// Called every second by background task
    pub async fn broadcast_tournament_info(&self) -> Result<()> {
        use crate::ws::messages::ServerMessage;

        // Get all running tournaments
        let running: Vec<Tournament> =
            sqlx::query_as("SELECT * FROM tournaments WHERE status = 'running'")
                .fetch_all(self.pool.as_ref())
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
            .fetch_optional(self.pool.as_ref())
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
            .fetch_optional(self.pool.as_ref())
            .await?;

            // Calculate remaining time for this level (server-side)
            let now = Utc::now();
            let level_start = DateTime::parse_from_rfc3339(&level_start_time)
                .unwrap()
                .with_timezone(&Utc);
            let elapsed_secs = (now - level_start).num_seconds();
            let remaining_secs = (tournament.level_duration_secs - elapsed_secs).max(0);

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
            self.game_server
                .broadcast_tournament_event(&tournament.id, message)
                .await;
        }

        Ok(())
    }
}
