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
        Self {
            id: Uuid::new_v4().to_string(),
            club_id,
            name,
            small_blind,
            big_blind,
            min_buyin,
            max_buyin,
            max_players,
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
