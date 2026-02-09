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
        let avatar_index = self.get_user_avatar_index(&user_id).await;
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(table_id)
            .ok_or(crate::game::error::GameError::InvalidTableId)?;

        table.take_seat(user_id, username, seat, stack)?;
        if let Some(player) = table.players.iter_mut().find(|p| p.seat == seat) {
            player.avatar_index = avatar_index;
        }

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
        let avatar_index = self.get_user_avatar_index(user_id).await;
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
        if let Some(player) = table.players.iter_mut().find(|p| p.user_id == user_id) {
            player.avatar_index = avatar_index;
        }

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
    pub async fn update_table_blinds(
        &self,
        table_id: &str,
        small_blind: i64,
        big_blind: i64,
        ante: i64,
    ) {
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

    /// Drain buffered eliminations from a table.
    /// Eliminations are buffered by `check_eliminations()` which runs at the
    /// start of each new hand, so this always has fresh data regardless of
    /// which phase the table is in.
    pub async fn check_table_eliminations(&self, table_id: &str) -> Option<(String, Vec<String>)> {
        let (tournament_id, eliminated) = {
            let mut tables = self.tables.write().await;
            let table = tables.get_mut(table_id)?;
            let eliminated = table.drain_pending_eliminations();
            if eliminated.is_empty() {
                return None;
            }
            let tid = table.tournament_id.clone()?;
            (tid, eliminated)
        };
        // Tables lock dropped — notify clients so eliminated players disappear
        self.notify_table_update(table_id).await;
        // Deregister eliminated bots
        {
            let mut bot_mgr = self.bot_manager.write().await;
            for user_id in &eliminated {
                bot_mgr.remove_bot(table_id, user_id);
            }
        }
        Some((tournament_id, eliminated))
    }

    /// Move a tournament player from one table to another.
    /// Handles bot re-registration and notifies both tables.
    /// Refuses to move a player if the source table is mid-hand.
    pub async fn move_tournament_player(
        &self,
        from_table_id: &str,
        to_table_id: &str,
        user_id: &str,
        tournament_id: Option<&str>,
    ) -> Result<(), String> {
        // Extract player info and move within a single tables lock
        let (username, stack, avatar_index) = {
            let mut tables = self.tables.write().await;
            let from_table = tables
                .get(from_table_id)
                .ok_or_else(|| "Source table not found".to_string())?;

            // Only move players between hands (Waiting phase),
            // so they see the full showdown before being moved
            if from_table.phase != crate::game::GamePhase::Waiting {
                return Err("Cannot move player while hand is in progress".to_string());
            }
            let player = from_table
                .players
                .iter()
                .find(|p| p.user_id == user_id)
                .ok_or_else(|| "Player not found on source table".to_string())?;
            let username = player.username.clone();
            let stack = player.stack;
            let avatar_index = player.avatar_index;

            // Remove from source
            let from_table = tables.get_mut(from_table_id).unwrap();
            from_table.remove_player(user_id);

            // Find next available seat on dest table
            let to_table = tables
                .get_mut(to_table_id)
                .ok_or_else(|| "Destination table not found".to_string())?;
            if to_table.players.len() >= to_table.max_seats {
                return Err("Destination table is full".to_string());
            }
            let mut occupied = vec![false; to_table.max_seats];
            for p in &to_table.players {
                if p.seat < to_table.max_seats {
                    occupied[p.seat] = true;
                }
            }
            let seat = occupied
                .iter()
                .position(|taken| !*taken)
                .ok_or_else(|| "No available seat on destination table".to_string())?;

            to_table
                .take_seat(user_id.to_string(), username.clone(), seat, stack)
                .map_err(|e| format!("Failed to seat player: {:?}", e))?;
            if let Some(player) = to_table.players.iter_mut().find(|p| p.user_id == user_id) {
                player.avatar_index = avatar_index;
            }

            (username, stack, avatar_index)
        };

        // Move bot registration if applicable
        {
            let mut bot_mgr = self.bot_manager.write().await;
            bot_mgr.move_bot(from_table_id, to_table_id, user_id);
        }

        // Notify both tables
        self.notify_table_update(from_table_id).await;
        self.notify_table_update(to_table_id).await;

        // Broadcast table change so the client can auto-switch
        if let Some(tid) = tournament_id {
            self.broadcast_tournament_event(
                tid,
                ServerMessage::TournamentTableChanged {
                    tournament_id: tid.to_string(),
                    table_id: to_table_id.to_string(),
                    user_id: user_id.to_string(),
                },
            )
            .await;
        }

        tracing::info!(
            "Moved player {} ({}) with stack {} and avatar_index {} from table {} to table {}",
            username,
            user_id,
            stack,
            avatar_index,
            from_table_id,
            to_table_id
        );

        Ok(())
    }

    /// Check if a table is currently in a hand (any phase other than Waiting).
    /// Players should only be moved during Waiting so they see the full showdown.
    pub async fn is_table_mid_hand(&self, table_id: &str) -> bool {
        let tables = self.tables.read().await;
        tables
            .get(table_id)
            .map(|t| t.phase != crate::game::GamePhase::Waiting)
            .unwrap_or(false)
    }

    /// Get player counts for a set of tournament tables.
    pub async fn get_table_player_counts(&self, table_ids: &[String]) -> Vec<(String, usize)> {
        let tables = self.tables.read().await;
        table_ids
            .iter()
            .filter_map(|id| {
                tables
                    .get(id.as_str())
                    .map(|t| (id.clone(), t.players.len()))
            })
            .collect()
    }

    /// Get all player user_ids at a table (for final table consolidation).
    pub async fn get_all_player_ids_at_table(&self, table_id: &str) -> Vec<String> {
        let tables = self.tables.read().await;
        tables
            .get(table_id)
            .map(|t| t.players.iter().map(|p| p.user_id.clone()).collect())
            .unwrap_or_default()
    }

    /// Get the user_id of the last player in a table's player list (for balancing moves).
    pub async fn get_last_player_at_table(&self, table_id: &str) -> Option<String> {
        let tables = self.tables.read().await;
        tables
            .get(table_id)
            .and_then(|t| t.players.last().map(|p| p.user_id.clone()))
    }

    async fn get_tournament_club_id_cached(&self, tournament_id: &str) -> Option<String> {
        {
            let cache = self.tournament_club_cache.read().await;
            if let Some(club_id) = cache.get(tournament_id) {
                return Some(club_id.clone());
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
            let mut cache = self.tournament_club_cache.write().await;
            cache.insert(tournament_id.to_string(), club_id.clone());
            Some(club_id)
        } else {
            None
        }
    }

    /// Broadcast a tournament event to all tables in a tournament
    pub async fn broadcast_tournament_event(&self, tournament_id: &str, message: ServerMessage) {
        // TournamentInfo is pushed every second; sending an additional full
        // TableState update for it causes unnecessary DB load and WS traffic.
        let should_notify_tables = !matches!(&message, ServerMessage::TournamentInfo { .. });

        if let Some(club_id) = self.get_tournament_club_id_cached(tournament_id).await {
            // Send via tournament broadcast channel (scoped by club)
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

        if should_notify_tables {
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
        }
    }
}
