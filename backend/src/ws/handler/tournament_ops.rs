use crate::ws::messages::ServerMessage;

use super::game_server::GameServer;
use super::TableBroadcast;

impl GameServer {
    /// Add a player to a table at a specific seat (for tournaments)
    pub async fn add_player_to_table(
        &self,
        table_id: &str,
        user_id: String,
        username: String,
        seat: usize,
        stack: i64,
    ) -> Result<(), crate::game::error::GameError> {
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        table.take_seat(user_id, username, seat, stack)?;

        Ok(())
    }

    /// Add a player to the next available seat (for late registration or rebuys)
    pub async fn add_player_to_table_next_seat(
        &self,
        table_id: &str,
        user_id: &str,
        username: &str,
        stack: i64,
    ) -> Result<usize, crate::game::error::GameError> {
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        if table.players.len() >= table.max_seats {
            return Err(crate::game::error::GameError::TableFull);
        }

        let mut occupied = vec![false; table.max_seats];
        for player in &table.players {
            if player.seat < table.max_seats {
                occupied[player.seat] = true;
            }
        }

        let seat = occupied
            .iter()
            .position(|taken| !*taken)
            .ok_or(crate::game::error::GameError::TableFull)?;

        table.take_seat(user_id.to_string(), username.to_string(), seat, stack)?;

        drop(tables);
        self.notify_table_update(table_id).await;

        Ok(seat)
    }

    /// Apply a tournament add-on or rebuy stack to a seated player
    pub async fn tournament_top_up(
        &self,
        table_id: &str,
        user_id: &str,
        amount: i64,
    ) -> Result<(), crate::game::error::GameError> {
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        table.top_up(user_id, amount)?;

        drop(tables);
        self.notify_table_update(table_id).await;

        Ok(())
    }

    /// Validate that a tournament player can be topped up right now.
    pub async fn can_tournament_top_up(
        &self,
        table_id: &str,
        user_id: &str,
    ) -> Result<(), crate::game::error::GameError> {
        let tables = self.tables.read().await;
        let table = tables
            .get(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        let player = table
            .players
            .iter()
            .find(|p| p.user_id == user_id)
            .ok_or(crate::game::error::GameError::PlayerNotAtTable)?;

        if table.phase != crate::game::GamePhase::Waiting
            && player.state != crate::game::player::PlayerState::SittingOut
        {
            return Err(crate::game::error::GameError::GameInProgress);
        }

        Ok(())
    }

    /// Validate that a tournament table has an open seat for a new player.
    pub async fn can_seat_tournament_player(
        &self,
        table_id: &str,
    ) -> Result<(), crate::game::error::GameError> {
        let tables = self.tables.read().await;
        let table = tables
            .get(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        if table.players.len() >= table.max_seats {
            return Err(crate::game::error::GameError::TableFull);
        }

        Ok(())
    }

    /// Update blinds on a table (for tournaments)
    pub async fn update_table_blinds(&self, table_id: &str, small_blind: i64, big_blind: i64, ante: i64) {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            table.update_blinds_and_ante(small_blind, big_blind, ante);
            tracing::info!(
                "Updated blinds on table {} to {}/{} ante {}",
                table_id,
                small_blind,
                big_blind,
                ante
            );
        }
    }

    /// Set tournament ID for a table
    pub async fn set_table_tournament(&self, table_id: &str, tournament_id: String) {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            table.set_tournament_id(Some(tournament_id));
        }
    }

    /// Force start a hand on a table (for tournaments after all players are seated)
    pub async fn force_start_table_hand(&self, table_id: &str) {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            table.force_start_hand();
            tracing::info!("Force-started first hand on table {}", table_id);
        }
        drop(tables);
        self.notify_table_update(table_id).await;
    }

    /// Check a table for eliminations and return (tournament_id, eliminated_users)
    pub async fn check_table_eliminations(&self, table_id: &str) -> Option<(String, Vec<String>)> {
        let mut tables = self.tables.write().await;
        if let Some(table) = tables.get_mut(table_id) {
            let eliminated = table.check_eliminations();
            if !eliminated.is_empty() {
                if let Some(tournament_id) = &table.tournament_id {
                    return Some((tournament_id.clone(), eliminated));
                }
            }
        }
        None
    }

    /// Broadcast a tournament event to all tables in a tournament
    pub async fn broadcast_tournament_event(&self, tournament_id: &str, message: ServerMessage) {
        // Get tournament to find its club_id
        let tournament_result: Result<(String,), sqlx::Error> =
            sqlx::query_as("SELECT club_id FROM tournaments WHERE id = ?")
                .bind(tournament_id)
                .fetch_one(self.pool.as_ref())
                .await;

        if let Ok((club_id,)) = tournament_result {
            // Send via tournament broadcast channel (scoped by club)
            let broadcasts = self.tournament_broadcasts.read().await;
            if let Some(tx) = broadcasts.get(&club_id) {
                let _ = tx.send(message.clone());
            }
            drop(broadcasts);
        }

        // Also find and trigger table updates for certain tournament events
        let tables = self.tables.read().await;
        let mut table_ids = Vec::new();

        for (table_id, table) in tables.iter() {
            if let Some(tid) = &table.tournament_id {
                if tid == tournament_id {
                    table_ids.push(table_id.clone());
                }
            }
        }

        drop(tables);

        // Trigger table state updates
        let table_broadcasts = self.table_broadcasts.read().await;
        for table_id in table_ids {
            if let Some(tx) = table_broadcasts.get(&table_id) {
                let _ = tx.send(TableBroadcast {
                    table_id: table_id.clone(),
                });
            }
        }

        let club_id: Option<(String,)> =
            sqlx::query_as("SELECT club_id FROM tournaments WHERE id = ?")
                .bind(tournament_id)
                .fetch_optional(self.pool.as_ref())
                .await
                .ok()
                .flatten();
        if let Some((club_id,)) = club_id {
            let broadcasts = self.tournament_broadcasts.read().await;
            if let Some(tx) = broadcasts.get(&club_id) {
                let _ = tx.send(message);
                tracing::debug!(
                    "Broadcasted tournament event for {} to club {}",
                    tournament_id,
                    club_id
                );
            }
        }
    }
}
