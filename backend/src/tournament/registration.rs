use crate::{
    db::models::{Tournament, TournamentRegistration},
    error::{AppError, Result},
};
use chrono::{DateTime, Duration, Utc};
use std::sync::Arc;

use super::context::TournamentContext;

pub(crate) struct RegistrationService {
    ctx: Arc<TournamentContext>,
}

impl RegistrationService {
    pub(crate) fn new(ctx: Arc<TournamentContext>) -> Self {
        Self { ctx }
    }

    /// Register a player for a tournament
    pub(crate) async fn register_player(
        &self,
        tournament_id: &str,
        user_id: &str,
        username: &str,
    ) -> Result<()> {
        // Load tournament
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;

        // Check status
        if tournament.status == "cancelled" {
            let reason = tournament
                .cancel_reason
                .clone()
                .unwrap_or_else(|| "Tournament was cancelled".to_string());
            return Err(AppError::BadRequest(reason));
        }

        let is_late_registration =
            tournament.status == "running" && late_registration_open(&tournament, Utc::now());

        if is_late_registration {
            self.ctx
                .ensure_late_registration_seat_available(&tournament)
                .await?;
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
        .fetch_optional(&*self.ctx.pool)
        .await?;

        if existing.is_some() {
            return Err(AppError::BadRequest("Already registered".to_string()));
        }

        // Deduct buy-in from club balance
        self.ctx
            .deduct_buy_in(&tournament.club_id, user_id, tournament.buy_in)
            .await?;

        // Create registration
        let registration =
            TournamentRegistration::new(tournament_id.to_string(), user_id.to_string());

        sqlx::query(
            "INSERT INTO tournament_registrations (tournament_id, user_id, registered_at, prize_amount) 
             VALUES (?, ?, ?, ?)",
        )
        .bind(&registration.tournament_id)
        .bind(&registration.user_id)
        .bind(&registration.registered_at)
        .bind(registration.prize_amount)
        .execute(&*self.ctx.pool)
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
        .execute(&*self.ctx.pool)
        .await?;

        // Update in-memory state
        if let Some(state) = self.ctx.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        tracing::info!(
            "Player {} ({}) registered for tournament {}",
            username,
            user_id,
            tournament_id
        );

        if tournament.status == "running" {
            if let Err(err) = self
                .ctx
                .seat_tournament_player(&tournament, user_id, username, tournament.starting_stack)
                .await
            {
                if is_late_registration {
                    self.ctx
                        .rollback_late_registration(&tournament, user_id)
                        .await?;
                }
                return Err(err);
            }
        }

        // Check if SNG should auto-start
        if tournament.format_id == "sng" && tournament.registered_players >= tournament.max_players
        {
            self.ctx.start_tournament_with_state(tournament_id, &mut tournament)
                .await?;
            self.ctx.start_sng_table(&tournament).await?;
        }

        Ok(())
    }

    /// Unregister a player from a tournament (before it starts)
    pub(crate) async fn unregister_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        let tournament = self.ctx.load_tournament(tournament_id).await?;

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
        .fetch_optional(&*self.ctx.pool)
        .await?;

        if registration.is_none() {
            return Err(AppError::BadRequest("Not registered".to_string()));
        }

        // Refund buy-in
        self.ctx
            .refund_buy_in(&tournament.club_id, user_id, tournament.buy_in)
            .await?;

        // Delete registration
        sqlx::query("DELETE FROM tournament_registrations WHERE tournament_id = ? AND user_id = ?")
            .bind(tournament_id)
            .bind(user_id)
            .execute(&*self.ctx.pool)
            .await?;

        // Update tournament counts
        sqlx::query(
            "UPDATE tournaments 
             SET registered_players = registered_players - 1, prize_pool = prize_pool - ? 
             WHERE id = ?",
        )
        .bind(tournament.buy_in)
        .bind(tournament_id)
        .execute(&*self.ctx.pool)
        .await?;

        tracing::info!(
            "Player {} unregistered from tournament {}",
            user_id,
            tournament_id
        );
        Ok(())
    }

    /// Rebuy into a running tournament (if enabled)
    pub(crate) async fn rebuy_player(
        &self,
        tournament_id: &str,
        user_id: &str,
        username: &str,
    ) -> Result<()> {
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;

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

        if !late_registration_open(&tournament, Utc::now()) {
            return Err(AppError::BadRequest("Rebuy period has ended".to_string()));
        }

        let mut registration = self.ctx.load_registration(tournament_id, user_id).await?;

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

        self.ctx.ensure_rebuy_seat_available(&tournament).await?;

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

        self.ctx
            .deduct_buy_in(&tournament.club_id, user_id, rebuy_amount)
            .await?;

        tournament.prize_pool += rebuy_amount;
        tournament.remaining_players += 1;

        sqlx::query("UPDATE tournaments SET prize_pool = ?, remaining_players = ? WHERE id = ?")
            .bind(tournament.prize_pool)
            .bind(tournament.remaining_players)
            .bind(tournament_id)
            .execute(&*self.ctx.pool)
            .await?;

        if let Some(state) = self.ctx.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        let previous_eliminated_at = registration.eliminated_at.clone();
        let previous_finish_position = registration.finish_position;
        let previous_rebuys = registration.rebuys;

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
        .execute(&*self.ctx.pool)
        .await?;

        if let Err(err) = self
            .ctx
            .seat_tournament_player(&tournament, user_id, username, rebuy_stack)
            .await
        {
            self.ctx
                .rollback_rebuy(
                    &tournament,
                    user_id,
                    rebuy_amount,
                    previous_rebuys,
                    previous_eliminated_at,
                    previous_finish_position,
                )
                .await?;
            return Err(err);
        }

        tracing::info!(
            "Player {} rebought into tournament {} (rebuys: {})",
            user_id,
            tournament_id,
            registration.rebuys
        );

        Ok(())
    }

    /// Add-on chips to a running tournament (if enabled)
    pub(crate) async fn addon_player(&self, tournament_id: &str, user_id: &str) -> Result<()> {
        let mut tournament = self.ctx.load_tournament(tournament_id).await?;

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

        let registration = self.ctx.load_registration(tournament_id, user_id).await?;

        if registration.eliminated_at.is_some() {
            return Err(AppError::BadRequest(
                "Eliminated players cannot take add-ons".to_string(),
            ));
        }

        self.ctx.ensure_addon_seated(&registration).await?;

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

        self.ctx
            .deduct_buy_in(&tournament.club_id, user_id, addon_amount)
            .await?;

        tournament.prize_pool += addon_amount;

        sqlx::query("UPDATE tournaments SET prize_pool = ? WHERE id = ?")
            .bind(tournament.prize_pool)
            .bind(tournament_id)
            .execute(&*self.ctx.pool)
            .await?;

        if let Some(state) = self.ctx.tournaments.write().await.get_mut(tournament_id) {
            state.tournament = tournament.clone();
        }

        sqlx::query(
            "UPDATE tournament_registrations 
             SET addons = addons + 1 
             WHERE tournament_id = ? AND user_id = ?",
        )
        .bind(tournament_id)
        .bind(user_id)
        .execute(&*self.ctx.pool)
        .await?;

        if let Err(err) = self.ctx.apply_addon_chips(&registration, addon_stack).await {
            self.ctx
                .rollback_addon(&tournament, user_id, addon_amount)
                .await?;
            return Err(err);
        }

        tracing::info!(
            "Player {} took an add-on in tournament {}",
            user_id,
            tournament_id
        );

        Ok(())
    }
}

fn late_registration_open(tournament: &Tournament, now: DateTime<Utc>) -> bool {
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
    now <= deadline
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tournament(late_registration_secs: i64) -> Tournament {
        Tournament::new(
            "club".to_string(),
            "Test".to_string(),
            "sng".to_string(),
            "holdem".to_string(),
            100,
            1000,
            9,
            2,
            300,
            0,
            false,
            0,
            0,
            0,
            false,
            0,
            0,
            0,
            late_registration_secs,
        )
    }

    #[test]
    fn late_registration_open_when_within_window() {
        let mut tournament = build_tournament(120);
        let now = Utc::now();
        tournament.actual_start = Some((now - Duration::seconds(30)).to_rfc3339());

        assert!(late_registration_open(&tournament, now));
    }

    #[test]
    fn late_registration_closed_when_expired_or_missing() {
        let mut tournament = build_tournament(60);
        let now = Utc::now();
        tournament.actual_start = Some((now - Duration::seconds(120)).to_rfc3339());

        assert!(!late_registration_open(&tournament, now));

        tournament.actual_start = None;
        assert!(!late_registration_open(&tournament, now));
    }
}
