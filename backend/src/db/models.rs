use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub id: String,
    pub username: String,
    pub email: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub created_at: String,
}

impl User {
    pub fn new(username: String, email: String, password_hash: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            username,
            email,
            password_hash,
            created_at: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Club {
    pub id: String,
    pub name: String,
    pub admin_id: String,
    pub created_at: String,
}

impl Club {
    pub fn new(name: String, admin_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            admin_id,
            created_at: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ClubMember {
    pub club_id: String,
    pub user_id: String,
    pub balance: i64, // in smallest unit (cents)
    pub joined_at: String,
}

impl ClubMember {
    pub fn new(club_id: String, user_id: String) -> Self {
        Self {
            club_id,
            user_id,
            balance: 0,
            joined_at: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Table {
    pub id: String,
    pub club_id: String,
    pub name: String,
    pub small_blind: i64,
    pub big_blind: i64,
    pub min_buyin: i64,
    pub max_buyin: i64,
    pub max_players: i32,
    pub variant_id: String,
    pub format_id: String,
    pub created_at: String,
}

impl Table {
    pub fn new(
        club_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        min_buyin: i64,
        max_buyin: i64,
        max_players: i32,
    ) -> Self {
        Self::with_variant_and_format(
            club_id,
            name,
            small_blind,
            big_blind,
            min_buyin,
            max_buyin,
            max_players,
            "holdem".to_string(),
            "cash".to_string(),
        )
    }

    pub fn with_variant_and_format(
        club_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        min_buyin: i64,
        max_buyin: i64,
        max_players: i32,
        variant_id: String,
        format_id: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            club_id,
            name,
            small_blind,
            big_blind,
            min_buyin,
            max_buyin,
            max_players,
            variant_id,
            format_id,
            created_at: Utc::now().to_rfc3339(),
        }
    }
}

#[allow(dead_code)] // Prepared for session history tracking
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TableSession {
    pub id: String,
    pub table_id: String,
    pub user_id: String,
    pub club_id: String,
    pub seat_number: i32,
    pub stack: i64,
    pub joined_at: String,
    pub left_at: Option<String>,
}

#[allow(dead_code)]
impl TableSession {
    pub fn new(
        table_id: String,
        user_id: String,
        club_id: String,
        seat_number: i32,
        stack: i64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            table_id,
            user_id,
            club_id,
            seat_number,
            stack,
            joined_at: Utc::now().to_rfc3339(),
            left_at: None,
        }
    }
}

#[allow(dead_code)] // Prepared for transaction ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionType {
    AdminCredit,
    AdminDebit,
    Buyin,
    Cashout,
}

#[allow(dead_code)]
impl TransactionType {
    pub fn as_str(&self) -> &str {
        match self {
            TransactionType::AdminCredit => "admin_credit",
            TransactionType::AdminDebit => "admin_debit",
            TransactionType::Buyin => "buyin",
            TransactionType::Cashout => "cashout",
        }
    }
}

impl std::str::FromStr for TransactionType {
    type Err = ();
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "admin_credit" => Ok(TransactionType::AdminCredit),
            "admin_debit" => Ok(TransactionType::AdminDebit),
            "buyin" => Ok(TransactionType::Buyin),
            "cashout" => Ok(TransactionType::Cashout),
            _ => Err(()),
        }
    }
}

#[allow(dead_code)] // Prepared for transaction ledger
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Transaction {
    pub id: String,
    pub club_id: String,
    pub user_id: String,
    pub amount: i64,
    pub transaction_type: String,
    pub description: Option<String>,
    pub created_at: String,
}

#[allow(dead_code)]
impl Transaction {
    pub fn new(
        club_id: String,
        user_id: String,
        amount: i64,
        transaction_type: TransactionType,
        description: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            club_id,
            user_id,
            amount,
            transaction_type: transaction_type.as_str().to_string(),
            description,
            created_at: Utc::now().to_rfc3339(),
        }
    }
}

// ============================================================================
// Tournament Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Tournament {
    pub id: String,
    pub club_id: String,
    pub name: String,
    pub format_id: String,
    pub variant_id: String,
    pub buy_in: i64,
    pub starting_stack: i64,
    pub prize_pool: i64,
    pub max_players: i32,
    pub registered_players: i32,
    pub remaining_players: i32,
    pub current_blind_level: i32,
    pub level_duration_secs: i64,
    pub level_start_time: Option<String>,
    pub status: String,
    pub scheduled_start: Option<String>,
    pub pre_seat_secs: i64,
    pub actual_start: Option<String>,
    pub finished_at: Option<String>,
    pub created_at: String,
}

impl Tournament {
    pub fn new(
        club_id: String,
        name: String,
        format_id: String,
        variant_id: String,
        buy_in: i64,
        starting_stack: i64,
        max_players: i32,
        level_duration_secs: i64,
        pre_seat_secs: i64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            club_id,
            name,
            format_id,
            variant_id,
            buy_in,
            starting_stack,
            prize_pool: 0,
            max_players,
            registered_players: 0,
            remaining_players: 0,
            current_blind_level: 0,
            level_duration_secs,
            level_start_time: None,
            status: "registering".to_string(),
            scheduled_start: None,
            pre_seat_secs,
            actual_start: None,
            finished_at: None,
            created_at: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TournamentRegistration {
    pub tournament_id: String,
    pub user_id: String,
    pub registered_at: String,
    pub starting_table_id: Option<String>,
    pub eliminated_at: Option<String>,
    pub finish_position: Option<i32>,
    pub prize_amount: i64,
}

impl TournamentRegistration {
    pub fn new(tournament_id: String, user_id: String) -> Self {
        Self {
            tournament_id,
            user_id,
            registered_at: Utc::now().to_rfc3339(),
            starting_table_id: None,
            eliminated_at: None,
            finish_position: None,
            prize_amount: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TournamentBlindLevel {
    pub tournament_id: String,
    pub level_number: i32,
    pub small_blind: i64,
    pub big_blind: i64,
    pub ante: i64,
}

impl TournamentBlindLevel {
    pub fn new(
        tournament_id: String,
        level_number: i32,
        small_blind: i64,
        big_blind: i64,
        ante: i64,
    ) -> Self {
        Self {
            tournament_id,
            level_number,
            small_blind,
            big_blind,
            ante,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TournamentTable {
    pub tournament_id: String,
    pub table_id: String,
    pub table_number: i32,
    pub is_active: i32,
}

impl TournamentTable {
    pub fn new(tournament_id: String, table_id: String, table_number: i32) -> Self {
        Self {
            tournament_id,
            table_id,
            table_number,
            is_active: 1,
        }
    }
}
