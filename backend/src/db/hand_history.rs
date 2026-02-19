//! Database persistence for hand history logs.

use crate::game::deck::Card;
use crate::game::table::hand_history::HandHistoryLog;
use sqlx::SqlitePool;

/// Card serialization for JSON storage (minimal: just rank + suit).
fn cards_to_json(cards: &[Card]) -> String {
    let arr: Vec<serde_json::Value> = cards
        .iter()
        .map(|c| serde_json::json!({"r": c.rank, "s": c.suit}))
        .collect();
    serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
}

/// Persist a completed hand to the database. Runs async, should not block game loop.
pub async fn save_hand_history(pool: &SqlitePool, log: &HandHistoryLog) -> Result<i64, sqlx::Error> {
    let community_json = cards_to_json(&log.community_cards);

    let hand_id: i64 = sqlx::query_scalar(
        "INSERT INTO hand_histories (table_id, tournament_id, hand_number, small_blind, big_blind, ante, dealer_seat, community_cards, pot_total, completed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
         RETURNING id",
    )
    .bind(&log.table_id)
    .bind(&log.tournament_id)
    .bind(log.hand_number)
    .bind(log.small_blind)
    .bind(log.big_blind)
    .bind(log.ante)
    .bind(log.dealer_seat as i64)
    .bind(&community_json)
    .bind(log.pot_total)
    .fetch_one(pool)
    .await?;

    // Insert players
    for p in &log.players {
        let hole_json = cards_to_json(&p.hole_cards);
        sqlx::query(
            "INSERT INTO hand_players (hand_id, seat, user_id, username, hole_cards, starting_stack, final_stack, is_winner, pot_won, winning_hand_desc, showed_cards, folded)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(hand_id)
        .bind(p.seat as i64)
        .bind(&p.user_id)
        .bind(&p.username)
        .bind(&hole_json)
        .bind(p.starting_stack)
        .bind(p.final_stack)
        .bind(p.is_winner as i64)
        .bind(p.pot_won)
        .bind(&p.winning_hand_desc)
        .bind(p.showed_cards as i64)
        .bind(p.folded as i64)
        .execute(pool)
        .await?;
    }

    // Insert actions
    for a in &log.actions {
        sqlx::query(
            "INSERT INTO hand_actions (hand_id, street, seq, seat, player_name, action_type, amount)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(hand_id)
        .bind(&a.street)
        .bind(a.seq)
        .bind(a.seat as i64)
        .bind(&a.player_name)
        .bind(&a.action_type)
        .bind(a.amount)
        .execute(pool)
        .await?;
    }

    Ok(hand_id)
}

/// A hand summary returned to clients, with visibility filtering applied.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandHistorySummary {
    pub hand_id: i64,
    pub hand_number: i64,
    pub small_blind: i64,
    pub big_blind: i64,
    pub ante: i64,
    pub dealer_seat: i64,
    pub community_cards: Vec<Card>,
    pub pot_total: i64,
    pub completed_at: String,
    pub players: Vec<HandHistoryPlayer>,
    pub actions: Vec<HandHistoryAction>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandHistoryPlayer {
    pub seat: i64,
    pub username: String,
    pub hole_cards: Option<Vec<Card>>, // None if not visible to requester
    pub starting_stack: i64,
    pub final_stack: i64,
    pub is_winner: bool,
    pub pot_won: i64,
    pub winning_hand_desc: Option<String>,
    pub folded: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HandHistoryAction {
    pub street: String,
    pub seq: i32,
    pub seat: i64,
    pub player_name: String,
    pub action_type: String,
    pub amount: i64,
}

fn parse_cards_json(json_str: &str) -> Vec<Card> {
    let arr: Vec<serde_json::Value> = serde_json::from_str(json_str).unwrap_or_default();
    arr.iter()
        .filter_map(|v| {
            Some(Card::new(v["r"].as_u64()? as u8, v["s"].as_u64()? as u8))
        })
        .collect()
}

/// Fetch hand history for a table, filtered for a specific player's visibility.
pub async fn get_hand_history(
    pool: &SqlitePool,
    table_id: &str,
    requesting_user_id: &str,
    limit: i64,
    offset: i64,
) -> Result<Vec<HandHistorySummary>, sqlx::Error> {
    // Fetch hands
    let rows: Vec<(i64, i64, i64, i64, i64, i64, String, i64, String)> = sqlx::query_as(
        "SELECT id, hand_number, small_blind, big_blind, ante, dealer_seat, community_cards, pot_total, COALESCE(completed_at, '') as completed_at
         FROM hand_histories
         WHERE table_id = ?
         ORDER BY id DESC
         LIMIT ? OFFSET ?",
    )
    .bind(table_id)
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await?;

    let mut summaries = Vec::with_capacity(rows.len());

    for (hand_id, hand_number, sb, bb, ante, dealer_seat, cc_json, pot, completed_at) in &rows {
        // Fetch players for this hand
        let player_rows: Vec<(i64, String, String, String, i64, i64, i64, i64, Option<String>, i64, i64)> = sqlx::query_as(
            "SELECT seat, user_id, username, hole_cards, starting_stack, final_stack, is_winner, pot_won, winning_hand_desc, showed_cards, folded
             FROM hand_players WHERE hand_id = ? ORDER BY seat",
        )
        .bind(hand_id)
        .fetch_all(pool)
        .await?;

        let players: Vec<HandHistoryPlayer> = player_rows
            .iter()
            .map(|(seat, user_id, username, hc_json, start_stack, final_stack, is_winner, pot_won, hand_desc, showed, folded)| {
                // Visibility: player sees own cards, or cards that were shown (showdown/voluntary)
                let visible = user_id == requesting_user_id || *showed != 0;
                let hole_cards = if visible {
                    Some(parse_cards_json(hc_json))
                } else {
                    None
                };
                HandHistoryPlayer {
                    seat: *seat,
                    username: username.clone(),
                    hole_cards,
                    starting_stack: *start_stack,
                    final_stack: *final_stack,
                    is_winner: *is_winner != 0,
                    pot_won: *pot_won,
                    winning_hand_desc: hand_desc.clone(),
                    folded: *folded != 0,
                }
            })
            .collect();

        // Fetch actions
        let action_rows: Vec<(String, i32, i64, String, String, i64)> = sqlx::query_as(
            "SELECT street, seq, seat, player_name, action_type, amount
             FROM hand_actions WHERE hand_id = ? ORDER BY seq",
        )
        .bind(hand_id)
        .fetch_all(pool)
        .await?;

        let actions: Vec<HandHistoryAction> = action_rows
            .iter()
            .map(|(street, seq, seat, name, atype, amount)| HandHistoryAction {
                street: street.clone(),
                seq: *seq,
                seat: *seat,
                player_name: name.clone(),
                action_type: atype.clone(),
                amount: *amount,
            })
            .collect();

        summaries.push(HandHistorySummary {
            hand_id: *hand_id,
            hand_number: *hand_number,
            small_blind: *sb,
            big_blind: *bb,
            ante: *ante,
            dealer_seat: *dealer_seat,
            community_cards: parse_cards_json(cc_json),
            pot_total: *pot,
            completed_at: completed_at.clone(),
            players,
            actions,
        });
    }

    Ok(summaries)
}
