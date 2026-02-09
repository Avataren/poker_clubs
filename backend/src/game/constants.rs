//! Game-related constants and default configuration values
//!
//! Centralizing these values makes it easier to:
//! - Configure different game variants
//! - Adjust for testing
//! - Support future customization per-club or per-tournament

// Allow dead code during development - not all constants are used yet
#![allow(dead_code)]

/// Default maximum number of seats at a table
pub const DEFAULT_MAX_SEATS: usize = 9;

/// Minimum players required to start a hand
pub const MIN_PLAYERS_TO_START: usize = 2;

/// Default starting balance for new club members (in smallest currency unit)
pub const DEFAULT_STARTING_BALANCE: i64 = 10000;

/// Default buy-in multipliers relative to big blind
pub const DEFAULT_MIN_BUYIN_BB: i64 = 20; // 20 big blinds
pub const DEFAULT_MAX_BUYIN_BB: i64 = 100; // 100 big blinds

/// Timing constants (in milliseconds)
pub const DEFAULT_STREET_DELAY_MS: u64 = 1000; // Delay between flop/turn/river
pub const DEFAULT_SHOWDOWN_DELAY_MS: u64 = 5000; // Delay to show results before next hand
pub const DEFAULT_FOLD_WIN_DELAY_MS: u64 = 2000; // Shorter delay for uncontested (fold) wins
/// Bot uncontested wins should still wait briefly so clients can render fold/muck animation.
pub const BOT_FOLD_WIN_DELAY_MS: u64 = 600;
/// Minimum think time after a phase change before a bot can act.
/// Prevents clients from skipping transient states when bot-only hands progress quickly.
pub const BOT_ACTION_THINK_DELAY_MS: u64 = 350;
/// MTT-only Waiting window between hands.
/// Gives deferred tournament balancing moves a short window to apply before the next hand.
pub const MTT_WAITING_REBALANCE_MS: u64 = 1000;

/// Maximum number of raises allowed per betting round (fixed-limit rule)
pub const MAX_RAISES_PER_ROUND: usize = 4;

/// Number of players for heads-up special blind/button rules
pub const HEADS_UP_PLAYER_COUNT: usize = 2;

/// Broadcast channel capacity
pub const BROADCAST_CHANNEL_CAPACITY: usize = 100;

/// Number of hole cards dealt per variant
pub const HOLDEM_HOLE_CARDS: usize = 2;
pub const OMAHA_HOLE_CARDS: usize = 4;

/// Community cards per street
pub const FLOP_CARDS: usize = 3;
pub const TURN_CARDS: usize = 1;
pub const RIVER_CARDS: usize = 1;
