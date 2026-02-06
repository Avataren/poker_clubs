//! Structured audit logging for security-relevant events.
//!
//! All game actions, balance changes, and tournament events are logged
//! using tracing spans for structured output.

/// Log a game action (bet, fold, raise, etc.)
pub fn log_game_action(table_id: &str, user_id: &str, action: &str, amount: Option<i64>) {
    tracing::info!(
        target: "audit",
        event = "game_action",
        table_id = table_id,
        user_id = user_id,
        action = action,
        amount = amount.unwrap_or(0),
        "Game action: {} by {} at table {}",
        action,
        user_id,
        table_id
    );
}

/// Log a balance change
pub fn log_balance_change(
    club_id: &str,
    user_id: &str,
    changed_by: &str,
    amount: i64,
    new_balance: i64,
) {
    tracing::info!(
        target: "audit",
        event = "balance_change",
        club_id = club_id,
        user_id = user_id,
        changed_by = changed_by,
        amount = amount,
        new_balance = new_balance,
        "Balance change: {} for user {} in club {} by {}",
        amount,
        user_id,
        club_id,
        changed_by
    );
}

/// Log a tournament event
pub fn log_tournament_event(tournament_id: &str, event: &str, details: &str) {
    tracing::info!(
        target: "audit",
        event = "tournament",
        tournament_id = tournament_id,
        tournament_event = event,
        details = details,
        "Tournament {}: {} - {}",
        tournament_id,
        event,
        details
    );
}

/// Log an authentication event
pub fn log_auth_event(username: &str, event: &str, success: bool) {
    if success {
        tracing::info!(
            target: "audit",
            event = "auth",
            username = username,
            auth_event = event,
            success = success,
            "Auth: {} - {} (success={})",
            event,
            username,
            success
        );
    } else {
        tracing::warn!(
            target: "audit",
            event = "auth",
            username = username,
            auth_event = event,
            success = success,
            "Auth: {} - {} (success={})",
            event,
            username,
            success
        );
    }
}

/// Log a security event (rate limiting, unauthorized access, etc.)
pub fn log_security_event(user_id: &str, event: &str, details: &str) {
    tracing::warn!(
        target: "audit",
        event = "security",
        user_id = user_id,
        security_event = event,
        details = details,
        "Security: {} - {} - {}",
        event,
        user_id,
        details
    );
}
