//! Game-related error types
//!
//! Using typed errors instead of String provides:
//! - Better error handling and matching
//! - Clearer API contracts
//! - Potential for localization
//! - Better debugging information

// Allow dead code during development - error variants will be used as integration continues
#![allow(dead_code)]

use std::fmt;

/// Errors that can occur during game operations
#[derive(Debug, Clone, PartialEq)]
pub enum GameError {
    // Table errors
    TableFull,
    TableNotFound,
    InvalidTableId,
    InvalidSeat { seat: usize, max_seats: usize },
    SeatOccupied { seat: usize },

    // Player errors
    PlayerAlreadySeated,
    PlayerNotAtTable,
    PlayerNotFound { user_id: String },
    NotEnoughChips { required: i64, available: i64 },

    // Action errors
    NotYourTurn,
    CannotAct,
    CannotCheck { current_bet: i64 },
    RaiseTooSmall { min_raise: i64, attempted: i64 },
    InvalidAction { reason: String },

    // Game state errors
    InvalidPhase { expected: String, actual: String },
    GameInProgress,
    GameNotInProgress,

    // Generic
    InternalError(String),
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Table errors
            GameError::TableFull => write!(f, "Table is full"),
            GameError::TableNotFound => write!(f, "Table not found"),
            GameError::InvalidTableId => write!(f, "Invalid table ID"),
            GameError::InvalidSeat { seat, max_seats } => {
                write!(f, "Invalid seat number {}. Max seats: {}", seat, max_seats)
            }
            GameError::SeatOccupied { seat } => {
                write!(f, "Seat {} is already occupied", seat)
            }

            // Player errors
            GameError::PlayerAlreadySeated => write!(f, "You are already at this table"),
            GameError::PlayerNotAtTable => write!(f, "You are not at this table"),
            GameError::PlayerNotFound { user_id } => {
                write!(f, "Player not found: {}", user_id)
            }
            GameError::NotEnoughChips {
                required,
                available,
            } => {
                write!(
                    f,
                    "Not enough chips. Required: {}, Available: {}",
                    required, available
                )
            }

            // Action errors
            GameError::NotYourTurn => write!(f, "Not your turn"),
            GameError::CannotAct => write!(f, "You cannot act"),
            GameError::CannotCheck { current_bet } => {
                write!(f, "Cannot check, must call {} or raise", current_bet)
            }
            GameError::RaiseTooSmall {
                min_raise,
                attempted,
            } => {
                write!(
                    f,
                    "Raise amount {} is too small. Minimum raise: {}",
                    attempted, min_raise
                )
            }
            GameError::InvalidAction { reason } => {
                write!(f, "Invalid action: {}", reason)
            }

            // Game state errors
            GameError::InvalidPhase { expected, actual } => {
                write!(
                    f,
                    "Invalid phase. Expected: {}, Actual: {}",
                    expected, actual
                )
            }
            GameError::GameInProgress => {
                write!(f, "Cannot perform action while game is in progress")
            }
            GameError::GameNotInProgress => write!(f, "Game is not in progress"),

            // Generic
            GameError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for GameError {}

/// Result type for game operations
pub type GameResult<T> = Result<T, GameError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GameError::RaiseTooSmall {
            min_raise: 100,
            attempted: 50,
        };
        assert_eq!(
            err.to_string(),
            "Raise amount 50 is too small. Minimum raise: 100"
        );

        let err = GameError::NotYourTurn;
        assert_eq!(err.to_string(), "Not your turn");
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(GameError::TableFull, GameError::TableFull);
        assert_ne!(GameError::TableFull, GameError::NotYourTurn);
    }
}
