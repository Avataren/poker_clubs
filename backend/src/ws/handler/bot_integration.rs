use super::game_server::GameServer;
use rand::Rng;

impl GameServer {
    /// Check all tables for bots whose turn it is and execute their actions.
    pub async fn check_bot_actions(&self) {
        // Phase 1: Check for broke bots and auto top-up
        let topups = {
            let tables = self.tables.read().await;
            let bot_mgr = self.bot_manager.read().await;
            let mut topups = Vec::new();

            for (table_id, table) in tables.iter() {
                // Only in cash games (can_top_up)
                if !table.can_top_up() {
                    continue;
                }

                for player in &table.players {
                    // Check if bot is broke or very low on chips
                    if bot_mgr.is_bot(&player.user_id) && player.stack < table.big_blind * 10 {
                        let topup_amount = table.big_blind * 100; // Top up to 100BB
                        topups.push((table_id.clone(), player.user_id.clone(), topup_amount));
                    }
                }
            }
            topups
        };

        // Execute top-ups
        for (table_id, user_id, amount) in topups {
            let mut tables = self.tables.write().await;
            if let Some(table) = tables.get_mut(&table_id) {
                if let Ok(()) = table.top_up(&user_id, amount) {
                    tracing::info!(
                        "Bot {} topped up ${} at table {}",
                        user_id,
                        amount,
                        table_id
                    );
                    // Notify clients about the top-up
                    drop(tables); // Release lock before notify
                    self.notify_table_update(&table_id).await;
                }
            }
        }

        // Phase 2: read tables + bot manager to collect actions
        let actions = {
            let tables = self.tables.read().await;
            let bot_mgr = self.bot_manager.read().await;
            bot_mgr.collect_bot_actions(&tables)
        };

        if actions.is_empty() {
            return;
        }

        // Phase 3: execute each action (needs write lock)
        for (table_id, user_id, action) in actions {
            let result = {
                let mut tables = self.tables.write().await;
                if let Some(table) = tables.get_mut(&table_id) {
                    match table.handle_action(&user_id, action) {
                        Ok(()) => true,
                        Err(e) => {
                            tracing::warn!(
                                "Bot {} action failed on table {}: {}",
                                user_id,
                                table_id,
                                e
                            );
                            false
                        }
                    }
                } else {
                    false
                }
            };

            if result {
                self.notify_table_update(&table_id).await;
            }
        }
    }

    /// Add a bot to a table. Returns (bot_user_id, bot_username) or error.
    pub async fn add_bot_to_table(
        &self,
        table_id: &str,
        buyin: i64,
        name: Option<String>,
        strategy: Option<&str>,
    ) -> Result<(String, String), String> {
        // Register bot in manager
        let (bot_user_id, bot_username) = {
            let mut bot_mgr = self.bot_manager.write().await;
            bot_mgr.add_bot(table_id, name, strategy)
        };
        let avatar_index: i32 = rand::thread_rng().gen_range(0..25);

        // Seat the bot at the table
        {
            let mut tables = self.tables.write().await;
            let table = tables
                .get_mut(table_id)
                .ok_or_else(|| "Table not found".to_string())?;
            table
                .add_player(bot_user_id.clone(), bot_username.clone(), buyin)
                .map_err(|e| {
                    // Clean up bot registration on failure
                    // (can't await here, so we'll do a blocking attempt)
                    e.to_string()
                })?;
            if let Some(player) = table.players.iter_mut().find(|p| p.user_id == bot_user_id) {
                player.avatar_index = avatar_index;
            }
        }

        self.notify_table_update(table_id).await;

        tracing::info!(
            "Bot {} ({}) added to table {} with {} chips (avatar_index={})",
            bot_username,
            bot_user_id,
            table_id,
            buyin,
            avatar_index
        );

        Ok((bot_user_id, bot_username))
    }

    /// Register an existing user as a bot with the bot manager (for tournament bots)
    pub async fn register_bot(
        &self,
        table_id: &str,
        user_id: String,
        username: String,
        strategy: Option<&str>,
    ) {
        let mut bot_mgr = self.bot_manager.write().await;

        // Register this user as a bot for this table
        bot_mgr.register_existing_bot(table_id, user_id.clone(), username.clone(), strategy);

        tracing::info!(
            "Registered existing user {} ({}) as bot on table {}",
            username,
            user_id,
            table_id
        );
    }

    /// Remove a bot from a table.
    pub async fn remove_bot_from_table(
        &self,
        table_id: &str,
        bot_user_id: &str,
    ) -> Result<(), String> {
        // Remove from bot manager
        {
            let mut bot_mgr = self.bot_manager.write().await;
            if !bot_mgr.remove_bot(table_id, bot_user_id) {
                return Err("Bot not found".to_string());
            }
        }

        // Remove from table
        {
            let mut tables = self.tables.write().await;
            if let Some(table) = tables.get_mut(table_id) {
                table.remove_player(bot_user_id);
            }
        }

        self.notify_table_update(table_id).await;

        tracing::info!("Bot {} removed from table {}", bot_user_id, table_id);
        Ok(())
    }
}
