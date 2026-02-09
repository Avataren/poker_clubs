//! Tournament Manager
//!
//! Centralized management for tournament lifecycle:
//! - Registration and unregistration
//! - Starting tournaments
//! - Blind level advancement
//! - Player elimination tracking
//! - Prize distribution

use crate::{
    db::{models::Tournament, DbPool},
    error::Result,
    ws::GameServer,
};
use std::sync::Arc;

use super::{
    blinds::BlindsService, broadcasts::BroadcastService, context::TournamentContext,
    lifecycle::LifecycleService, registration::RegistrationService,
};

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
    ctx: Arc<TournamentContext>,
    registration: RegistrationService,
    lifecycle: LifecycleService,
    blinds: BlindsService,
    broadcasts: BroadcastService,
}

impl TournamentManager {
    pub fn new(pool: Arc<DbPool>, game_server: Arc<GameServer>) -> Self {
        let ctx = Arc::new(TournamentContext::new(pool, game_server));
        Self {
            ctx: ctx.clone(),
            registration: RegistrationService::new(ctx.clone()),
            lifecycle: LifecycleService::new(ctx.clone()),
            blinds: BlindsService::new(ctx.clone()),
            broadcasts: BroadcastService::new(ctx.clone()),
        }
    }

    /// Create a new Sit & Go tournament
    pub async fn create_sng(&self, club_id: &str, config: SngConfig) -> Result<Tournament> {
        self.lifecycle.create_sng(club_id, config).await
    }

    /// Create a new Multi-Table Tournament
    pub async fn create_mtt(&self, club_id: &str, config: MttConfig) -> Result<Tournament> {
        self.lifecycle.create_mtt(club_id, config).await
    }

    /// Register a player for a tournament
    pub async fn register_player(
        &self,
        tournament_id: &str,
        user_id: &str,
        username: &str,
    ) -> Result<()> {
        self.registration
            .register_player(tournament_id, user_id, username)
            .await
    }

    /// Unregister a player from a tournament (before it starts)
    pub async fn unregister_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        self.registration
            .unregister_player(tournament_id, user_id)
            .await
    }

    /// Start a tournament
    pub async fn start_tournament(&self, tournament_id: &str) -> Result<()> {
        self.lifecycle.start_tournament(tournament_id).await
    }

    /// Cancel a tournament manually with a reason.
    pub async fn cancel_tournament(&self, tournament_id: &str, reason: Option<&str>) -> Result<()> {
        self.lifecycle
            .cancel_tournament(tournament_id, reason)
            .await
    }

    /// Rebuy into a running tournament (if enabled)
    pub async fn rebuy_player(
        &self,
        tournament_id: &str,
        user_id: &str,
        username: &str,
    ) -> Result<()> {
        self.registration
            .rebuy_player(tournament_id, user_id, username)
            .await
    }

    /// Add-on chips to a running tournament (if enabled)
    pub async fn addon_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        self.registration.addon_player(tournament_id, user_id).await
    }

    /// Advance to the next blind level
    pub async fn advance_blind_level(&self, tournament_id: &str) -> Result<bool> {
        self.blinds.advance_blind_level(tournament_id).await
    }

    /// Handle player elimination
    pub async fn eliminate_player(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<Option<Vec<crate::tournament::prizes::PrizeWinner>>> {
        self.lifecycle
            .eliminate_player(tournament_id, user_id)
            .await
    }

    /// Check all running tournaments for blind level advancement
    /// Called periodically by background task
    pub async fn check_all_blind_levels(&self) -> Result<()> {
        self.blinds.check_all_blind_levels().await
    }

    /// Check all tournament tables for player eliminations
    /// Called periodically by background task
    pub async fn check_tournament_eliminations(&self) -> Result<()> {
        self.lifecycle.check_tournament_eliminations().await
    }

    /// Broadcast tournament info to all players at tournament tables
    /// Called every second by background task
    pub async fn broadcast_tournament_info(&self) -> Result<()> {
        self.broadcasts.broadcast_tournament_info().await
    }

    /// Check all tournaments with a scheduled start and trigger auto-start/cancel.
    /// Called periodically by background task.
    pub async fn check_scheduled_starts(&self) -> Result<()> {
        self.lifecycle.check_scheduled_starts().await
    }

    /// Remove a tournament from the in-memory cache.
    pub async fn remove_from_cache(&self, tournament_id: &str) {
        self.ctx.tournaments.write().await.remove(tournament_id);
    }

    /// Remove finished/cancelled tournaments from in-memory cache after 1 hour.
    /// Called periodically by background task.
    pub async fn cleanup_finished_tournaments(&self) {
        self.ctx.cleanup_finished_tournaments().await;
    }

    /// Startup recovery: cancel running tournaments whose in-memory hand state
    /// was lost by a process restart.
    pub async fn cancel_orphaned_running_tournaments_on_startup(&self) -> Result<usize> {
        self.lifecycle
            .cancel_orphaned_running_tournaments_on_startup()
            .await
    }
}
