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
    pub level_duration_secs: i64,
}

/// Configuration for creating an MTT
#[derive(Debug, Clone)]
pub struct MttConfig {
    pub name: String,
    pub variant_id: String,
    pub buy_in: i64,
    pub starting_stack: i64,
    pub max_players: i32,
    pub level_duration_secs: i64,
    pub scheduled_start: Option<String>,
    pub pre_seat_secs: i64,
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
        // Create tournament record
        let tournament = Tournament::new(
            club_id.to_string(),
            config.name,
            "sng".to_string(),
            config.variant_id,
            config.buy_in,
            config.starting_stack,
            config.max_players,
            config.level_duration_secs,
            0,
        );

        // Generate blind schedule
        let starting_bb = config.starting_stack / 100; // Starting BB = 1% of stack
        let blind_schedule = BlindSchedule::standard_tournament(starting_bb, config.level_duration_secs as u64);

        // Save tournament to database
        self.save_tournament(&tournament).await?;
        
        // Save blind levels
        self.save_blind_levels(&tournament.id, &blind_schedule.levels).await?;

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

        tracing::info!("Created SNG tournament: {} ({})", tournament.name, tournament.id);
        Ok(tournament)
    }

    /// Create a new Multi-Table Tournament
    pub async fn create_mtt(&self, club_id: &str, config: MttConfig) -> Result<Tournament> {
        let mut tournament = Tournament::new(
            club_id.to_string(),
            config.name,
            "mtt".to_string(),
            config.variant_id,
            config.buy_in,
            config.starting_stack,
            config.max_players,
            config.level_duration_secs,
            config.pre_seat_secs,
        );

        tournament.scheduled_start = config.scheduled_start;

        // Generate blind schedule
        let starting_bb = config.starting_stack / 100;
        let blind_schedule = BlindSchedule::standard_tournament(starting_bb, config.level_duration_secs as u64);

        // Save to database
        self.save_tournament(&tournament).await?;
        self.save_blind_levels(&tournament.id, &blind_schedule.levels).await?;

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

        tracing::info!("Created MTT tournament: {} ({})", tournament.name, tournament.id);
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
        if tournament.status != "registering" {
            return Err(AppError::BadRequest("Tournament is not accepting registrations".to_string()));
        }

        // Check if full
        if tournament.registered_players >= tournament.max_players {
            return Err(AppError::BadRequest("Tournament is full".to_string()));
        }

        // Check if already registered
        let existing = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?"
        )
        .bind(tournament_id)
        .bind(user_id)
        .fetch_optional(&*self.pool)
        .await?;

        if existing.is_some() {
            return Err(AppError::BadRequest("Already registered".to_string()));
        }

        // Deduct buy-in from club balance
        self.deduct_buy_in(&tournament.club_id, user_id, tournament.buy_in).await?;

        // Create registration
        let registration = TournamentRegistration::new(tournament_id.to_string(), user_id.to_string());
        
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

        sqlx::query(
            "UPDATE tournaments 
             SET registered_players = ?, prize_pool = ? 
             WHERE id = ?"
        )
        .bind(tournament.registered_players)
        .bind(tournament.prize_pool)
        .bind(tournament_id)
        .execute(&*self.pool)
        .await?;

        // Update in-memory state
        if let Some(state) = self.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        tracing::info!("Player {} ({}) registered for tournament {}", username, user_id, tournament_id);

        // Check if SNG should auto-start
        if tournament.format_id == "sng" && tournament.registered_players >= tournament.max_players {
            self.start_tournament(tournament_id).await?;
        }

        Ok(())
    }

    /// Unregister a player from a tournament (before it starts)
    pub async fn unregister_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        let tournament = self.load_tournament(tournament_id).await?;

        if tournament.status != "registering" {
            return Err(AppError::BadRequest("Cannot unregister after tournament has started".to_string()));
        }

        // Check if registered
        let registration = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?"
        )
        .bind(tournament_id)
        .bind(user_id)
        .fetch_optional(&*self.pool)
        .await?;

        if registration.is_none() {
            return Err(AppError::BadRequest("Not registered".to_string()));
        }

        // Refund buy-in
        self.refund_buy_in(&tournament.club_id, user_id, tournament.buy_in).await?;

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
             WHERE id = ?"
        )
        .bind(tournament.buy_in)
        .bind(tournament_id)
        .execute(&*self.pool)
        .await?;

        tracing::info!("Player {} unregistered from tournament {}", user_id, tournament_id);
        Ok(())
    }

    /// Start a tournament
    pub async fn start_tournament(&self, tournament_id: &str) -> Result<()> {
        let mut tournament = self.load_tournament(tournament_id).await?;
        self.start_tournament_with_state(tournament_id, &mut tournament)
            .await?;

        // Create table(s) and seat players
        if tournament.format_id == "sng" {
            self.start_sng_table(&tournament).await?;
        } else {
            self.start_mtt_tables(&tournament).await?;
        }

        tracing::info!("Started tournament: {} ({})", tournament.name, tournament_id);
        Ok(())
    }

    /// Advance to the next blind level
    pub async fn advance_blind_level(&self, tournament_id: &str) -> Result<bool> {
        let mut tournament = self.load_tournament(tournament_id).await?;

        if tournament.status != "running" {
            return Ok(false);
        }

        // Load blind levels
        let blind_levels = self.load_blind_levels(tournament_id).await?;
        let next_level = (tournament.current_blind_level + 1) as usize;

        if next_level >= blind_levels.len() {
            // No more levels, keep current
            return Ok(false);
        }

        // Update tournament
        tournament.current_blind_level += 1;
        tournament.level_start_time = Some(Utc::now().to_rfc3339());

        sqlx::query(
            "UPDATE tournaments SET current_blind_level = ?, level_start_time = ? WHERE id = ?"
        )
        .bind(tournament.current_blind_level)
        .bind(&tournament.level_start_time)
        .bind(tournament_id)
        .execute(&*self.pool)
        .await?;

        let new_level = &blind_levels[next_level];
        tracing::info!(
            "Tournament {} advanced to level {}: {}/{}",
            tournament_id,
            next_level + 1,
            new_level.small_blind,
            new_level.big_blind
        );

        // TODO: Update blinds on all active tournament tables
        // This will be implemented in Phase 3

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
             WHERE tournament_id = ? AND user_id = ?"
        )
        .bind(Utc::now().to_rfc3339())
        .bind(finish_position)
        .bind(tournament_id)
        .bind(user_id)
        .execute(&*self.pool)
        .await?;

        tracing::info!(
            "Player {} eliminated from tournament {} at position {}",
            user_id,
            tournament_id,
            finish_position
        );

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

        // Distribute prizes
        let winners = self.distribute_prizes(tournament_id).await?;

        tracing::info!("Tournament {} finished with {} winners", tournament_id, winners.len());
        Ok(winners)
    }

    /// Distribute prizes to winners
    async fn distribute_prizes(&self, tournament_id: &str) -> Result<Vec<PrizeWinner>> {
        let tournament = self.load_tournament(tournament_id).await?;
        
        // Get prize structure
        let prize_structure = if let Some(state) = self.tournaments.read().await.get(tournament_id) {
            state.prize_structure.clone()
        } else {
            PrizeStructure::for_player_count(tournament.max_players)
        };

        // Get all registrations ordered by finish position
        let registrations = sqlx::query_as::<_, TournamentRegistration>(
            "SELECT * FROM tournament_registrations 
             WHERE tournament_id = ? 
             ORDER BY COALESCE(finish_position, 999999)"
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
                         WHERE tournament_id = ? AND user_id = ?"
                    )
                    .bind(prize)
                    .bind(tournament_id)
                    .bind(&registration.user_id)
                    .execute(&*self.pool)
                    .await?;

                    // Credit balance
                    self.credit_prize(&tournament.club_id, &registration.user_id, prize).await?;

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
                prize_pool, max_players, registered_players, remaining_players,
                current_blind_level, level_duration_secs, level_start_time,
                status, scheduled_start, pre_seat_secs, actual_start, finished_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
        .bind(&tournament.created_at)
        .execute(&*self.pool)
        .await?;

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
        let mut tournament = sqlx::query_as::<_, Tournament>("SELECT * FROM tournaments WHERE id = ?")
            .bind(tournament_id)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| match e {
                sqlx::Error::RowNotFound => AppError::NotFound("Tournament not found".to_string()),
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

        if (tournament.status == "registering" || tournament.status == "seating") && now >= scheduled_start {
            if tournament.registered_players >= 2 {
                self.start_tournament_with_state(&tournament.id, tournament)
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
            return Err(AppError::BadRequest("Tournament already started or finished".to_string()));
        }

        if tournament.registered_players < 2 {
            return Err(AppError::BadRequest("Need at least 2 players to start".to_string()));
        }

        // Update tournament status
        tournament.status = "running".to_string();
        tournament.actual_start = Some(Utc::now().to_rfc3339());
        tournament.level_start_time = Some(Utc::now().to_rfc3339());
        tournament.remaining_players = tournament.registered_players;

        sqlx::query(
            "UPDATE tournaments 
             SET status = ?, actual_start = ?, level_start_time = ?, remaining_players = ? 
             WHERE id = ?"
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

    async fn load_blind_levels(&self, tournament_id: &str) -> Result<Vec<TournamentBlindLevel>> {
        Ok(sqlx::query_as::<_, TournamentBlindLevel>(
            "SELECT * FROM tournament_blind_levels WHERE tournament_id = ? ORDER BY level_number"
        )
        .bind(tournament_id)
        .fetch_all(&*self.pool)
        .await?)
    }

    async fn deduct_buy_in(&self, club_id: &str, user_id: &str, amount: i64) -> Result<()> {
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
        sqlx::query("UPDATE club_members SET balance = balance + ? WHERE club_id = ? AND user_id = ?")
            .bind(amount)
            .bind(club_id)
            .bind(user_id)
            .execute(&*self.pool)
            .await?;

        Ok(())
    }

    async fn credit_prize(&self, club_id: &str, user_id: &str, amount: i64) -> Result<()> {
        sqlx::query("UPDATE club_members SET balance = balance + ? WHERE club_id = ? AND user_id = ?")
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

    async fn start_sng_table(&self, _tournament: &Tournament) -> Result<()> {
        // TODO: Implement in Phase 3
        // - Create a single table
        // - Seat all registered players
        // - Link table to tournament
        Ok(())
    }

    async fn start_mtt_tables(&self, _tournament: &Tournament) -> Result<()> {
        // TODO: Implement in Phase 3  
        // - Calculate number of tables needed
        // - Create tables
        // - Distribute players evenly
        // - Link tables to tournament
        Ok(())
    }
}
