//! Game Format System
//!
//! Defines different ways a poker game can be structured:
//! - Cash Game: Players can buy-in, leave, top-up anytime
//! - Sit & Go (SNG): Fixed players, starts when full, plays to one winner
//! - Multi-Table Tournament (MTT): Scheduled start, many tables, consolidates
//!
//! Each format has different rules for:
//! - When a game starts
//! - Blind structure (static vs. increasing)
//! - When players are eliminated
//! - How the game ends

// Allow dead code during development - these formats will be integrated incrementally
#![allow(dead_code)]

use super::constants::{DEFAULT_MAX_SEATS, DEFAULT_SHOWDOWN_DELAY_MS, DEFAULT_STREET_DELAY_MS};

/// Blind level for tournaments
#[derive(Debug, Clone)]
pub struct BlindLevel {
    pub small_blind: i64,
    pub big_blind: i64,
    pub ante: i64,
    /// Duration in seconds (0 = indefinite for cash games)
    pub duration_secs: u64,
}

impl BlindLevel {
    pub fn new(small_blind: i64, big_blind: i64) -> Self {
        Self {
            small_blind,
            big_blind,
            ante: 0,
            duration_secs: 0,
        }
    }

    pub fn with_ante(mut self, ante: i64) -> Self {
        self.ante = ante;
        self
    }

    pub fn with_duration(mut self, duration_secs: u64) -> Self {
        self.duration_secs = duration_secs;
        self
    }
}

/// Blind schedule for tournaments
#[derive(Debug, Clone)]
pub struct BlindSchedule {
    pub levels: Vec<BlindLevel>,
    pub current_level: usize,
}

impl BlindSchedule {
    pub fn new(levels: Vec<BlindLevel>) -> Self {
        Self {
            levels,
            current_level: 0,
        }
    }

    /// Create a standard tournament blind schedule
    pub fn standard_tournament(starting_bb: i64, level_duration_secs: u64) -> Self {
        let multipliers = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100];
        let levels = multipliers
            .iter()
            .map(|&mult| {
                BlindLevel::new(starting_bb * mult / 2, starting_bb * mult)
                    .with_duration(level_duration_secs)
            })
            .collect();
        Self::new(levels)
    }

    /// Get the current blind level
    pub fn current(&self) -> &BlindLevel {
        &self.levels[self.current_level.min(self.levels.len() - 1)]
    }

    /// Advance to the next level
    pub fn advance(&mut self) -> bool {
        if self.current_level < self.levels.len() - 1 {
            self.current_level += 1;
            true
        } else {
            false
        }
    }
}

/// Prize pool distribution
#[derive(Debug, Clone)]
pub struct PrizeStructure {
    /// Percentages for each position (index 0 = 1st place)
    pub payouts: Vec<f64>,
}

impl PrizeStructure {
    /// Standard payout for heads-up SNG
    pub fn heads_up() -> Self {
        Self {
            payouts: vec![100.0],
        }
    }

    /// Standard payout for 3-player SNG
    pub fn three_player() -> Self {
        Self {
            payouts: vec![65.0, 35.0],
        }
    }

    /// Standard payout for 6-max SNG
    pub fn six_max() -> Self {
        Self {
            payouts: vec![65.0, 35.0],
        }
    }

    /// Standard payout for 9-player SNG
    pub fn nine_player() -> Self {
        Self {
            payouts: vec![50.0, 30.0, 20.0],
        }
    }

    /// Calculate payout for a position
    pub fn payout_for_position(&self, position: usize, total_prize_pool: i64) -> i64 {
        if position < self.payouts.len() {
            (total_prize_pool as f64 * self.payouts[position] / 100.0) as i64
        } else {
            0
        }
    }
}

/// Configuration for a game format
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Display name
    pub name: String,
    /// Maximum seats at the table
    pub max_seats: usize,
    /// Starting chip stack (for tournaments)
    pub starting_stack: i64,
    /// Buy-in amount (for tournaments)
    pub buy_in: i64,
    /// Blind schedule
    pub blind_schedule: BlindSchedule,
    /// Prize structure (for tournaments)
    pub prize_structure: Option<PrizeStructure>,
    /// Timing
    pub street_delay_ms: u64,
    pub showdown_delay_ms: u64,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            name: "Cash Game".to_string(),
            max_seats: DEFAULT_MAX_SEATS,
            starting_stack: 0,
            buy_in: 0,
            blind_schedule: BlindSchedule::new(vec![BlindLevel::new(50, 100)]),
            prize_structure: None,
            street_delay_ms: DEFAULT_STREET_DELAY_MS,
            showdown_delay_ms: DEFAULT_SHOWDOWN_DELAY_MS,
        }
    }
}

/// Status of a tournament/SNG
#[derive(Debug, Clone, PartialEq)]
pub enum TournamentStatus {
    /// Waiting for registrations
    Registering,
    /// Countdown window before start
    Seating,
    /// Game is in progress
    Running,
    /// Game is paused (for breaks)
    Paused,
    /// Game is finished
    Finished,
    /// Game was cancelled
    Cancelled,
}

/// Core trait for game formats
pub trait GameFormat: Send + Sync + std::fmt::Debug {
    /// Get the format name
    fn name(&self) -> &str;

    /// Get the format type ID (cash, sng, mtt)
    fn format_id(&self) -> &'static str;

    /// Get the format configuration
    fn config(&self) -> &FormatConfig;

    /// Can players join/register?
    fn can_join(&self) -> bool;

    /// Can players leave with their chips?
    fn can_cash_out(&self) -> bool;

    /// Can players add chips?
    fn can_top_up(&self) -> bool;

    /// Should blinds increase?
    fn has_increasing_blinds(&self) -> bool;

    /// Get current blinds
    fn current_blinds(&self) -> (i64, i64);

    /// Is a player eliminated when they lose all chips?
    fn eliminates_players(&self) -> bool;

    /// Number of players remaining to end the game (1 for tournaments)
    fn players_to_end(&self) -> usize {
        1
    }

    /// Should the game auto-start when min players reached?
    /// For cash games: yes (start with 2+ players)
    /// For tournaments: no (wait for explicit start or all players seated)
    fn should_auto_start(&self) -> bool {
        true // Default for cash games
    }

    /// Clone into a boxed trait object
    fn clone_box(&self) -> Box<dyn GameFormat>;
}

/// Cash game format - open entry, no eliminations
#[derive(Debug, Clone)]
pub struct CashGame {
    config: FormatConfig,
}

impl CashGame {
    pub fn new(small_blind: i64, big_blind: i64, max_seats: usize) -> Self {
        Self {
            config: FormatConfig {
                name: format!("Cash ${}/{}", small_blind, big_blind),
                max_seats,
                blind_schedule: BlindSchedule::new(vec![BlindLevel::new(small_blind, big_blind)]),
                ..Default::default()
            },
        }
    }
}

impl GameFormat for CashGame {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn format_id(&self) -> &'static str {
        "cash"
    }

    fn config(&self) -> &FormatConfig {
        &self.config
    }

    fn can_join(&self) -> bool {
        true // Always open for cash games
    }

    fn can_cash_out(&self) -> bool {
        true
    }

    fn can_top_up(&self) -> bool {
        true
    }

    fn has_increasing_blinds(&self) -> bool {
        false
    }

    fn current_blinds(&self) -> (i64, i64) {
        let level = self.config.blind_schedule.current();
        (level.small_blind, level.big_blind)
    }

    fn eliminates_players(&self) -> bool {
        false // Players can rebuy in cash games
    }

    fn should_auto_start(&self) -> bool {
        true // Cash games start as soon as 2+ players are seated
    }

    fn clone_box(&self) -> Box<dyn GameFormat> {
        Box::new(self.clone())
    }
}

/// Sit & Go format - fixed players, one table
#[derive(Debug, Clone)]
pub struct SitAndGo {
    config: FormatConfig,
    status: TournamentStatus,
    registered_players: usize,
}

impl SitAndGo {
    pub fn new(
        buy_in: i64,
        starting_stack: i64,
        max_players: usize,
        level_duration_secs: u64,
    ) -> Self {
        let blind_schedule = BlindSchedule::standard_tournament(
            starting_stack / 100, // Starting BB = 1% of starting stack
            level_duration_secs,
        );

        Self {
            config: FormatConfig {
                name: format!("SNG ${} ({}-max)", buy_in, max_players),
                max_seats: max_players,
                starting_stack,
                buy_in,
                blind_schedule,
                prize_structure: Some(match max_players {
                    2 => PrizeStructure::heads_up(),
                    3 => PrizeStructure::three_player(),
                    4..=6 => PrizeStructure::six_max(),
                    _ => PrizeStructure::nine_player(),
                }),
                ..Default::default()
            },
            status: TournamentStatus::Registering,
            registered_players: 0,
        }
    }

    pub fn register_player(&mut self) -> bool {
        if self.status == TournamentStatus::Registering
            && self.registered_players < self.config.max_seats
        {
            self.registered_players += 1;
            if self.registered_players == self.config.max_seats {
                self.status = TournamentStatus::Running;
            }
            true
        } else {
            false
        }
    }

    pub fn status(&self) -> &TournamentStatus {
        &self.status
    }
}

impl GameFormat for SitAndGo {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn format_id(&self) -> &'static str {
        "sng"
    }

    fn config(&self) -> &FormatConfig {
        &self.config
    }

    fn can_join(&self) -> bool {
        self.status == TournamentStatus::Registering
    }

    fn can_cash_out(&self) -> bool {
        false // Must play to the end
    }

    fn can_top_up(&self) -> bool {
        false // No rebuys in standard SNG
    }

    fn has_increasing_blinds(&self) -> bool {
        true
    }

    fn current_blinds(&self) -> (i64, i64) {
        let level = self.config.blind_schedule.current();
        (level.small_blind, level.big_blind)
    }

    fn eliminates_players(&self) -> bool {
        true
    }

    fn should_auto_start(&self) -> bool {
        false // SNGs do NOT auto-start - must be explicitly started
    }

    fn clone_box(&self) -> Box<dyn GameFormat> {
        Box::new(self.clone())
    }
}

/// Multi-table tournament format (foundation for future)
#[derive(Debug, Clone)]
pub struct MultiTableTournament {
    config: FormatConfig,
    status: TournamentStatus,
    total_entries: usize,
    remaining_players: usize,
}

impl MultiTableTournament {
    pub fn new(name: String, buy_in: i64, starting_stack: i64, level_duration_secs: u64) -> Self {
        let blind_schedule =
            BlindSchedule::standard_tournament(starting_stack / 100, level_duration_secs);

        Self {
            config: FormatConfig {
                name,
                max_seats: DEFAULT_MAX_SEATS,
                starting_stack,
                buy_in,
                blind_schedule,
                prize_structure: None, // Set based on entries
                ..Default::default()
            },
            status: TournamentStatus::Registering,
            total_entries: 0,
            remaining_players: 0,
        }
    }

    pub fn status(&self) -> &TournamentStatus {
        &self.status
    }
}

impl GameFormat for MultiTableTournament {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn format_id(&self) -> &'static str {
        "mtt"
    }

    fn config(&self) -> &FormatConfig {
        &self.config
    }

    fn can_join(&self) -> bool {
        self.status == TournamentStatus::Registering
    }

    fn can_cash_out(&self) -> bool {
        false
    }

    fn can_top_up(&self) -> bool {
        false // Standard MTT - no rebuys
    }

    fn has_increasing_blinds(&self) -> bool {
        true
    }

    fn current_blinds(&self) -> (i64, i64) {
        let level = self.config.blind_schedule.current();
        (level.small_blind, level.big_blind)
    }

    fn eliminates_players(&self) -> bool {
        true
    }

    fn should_auto_start(&self) -> bool {
        false // MTTs do NOT auto-start - must be explicitly started
    }

    fn clone_box(&self) -> Box<dyn GameFormat> {
        Box::new(self.clone())
    }
}

/// Factory function to create a format from its ID
/// Note: SNG and MTT require specific parameters, so this creates with defaults
/// For cash games, caller should typically use CashGame::new() directly with blinds
pub fn format_from_id(
    id: &str,
    small_blind: i64,
    big_blind: i64,
    max_seats: usize,
) -> Option<Box<dyn GameFormat>> {
    match id {
        "cash" => Some(Box::new(CashGame::new(small_blind, big_blind, max_seats))),
        "sng" => Some(Box::new(SitAndGo::new(
            big_blind * 100, // Default buy-in = 100 big blinds
            big_blind * 100, // Starting stack = buy-in
            max_seats,
            300, // 5 minute levels
        ))),
        "mtt" => Some(Box::new(MultiTableTournament::new(
            "Tournament".to_string(),
            big_blind * 100,
            big_blind * 100,
            600, // 10 minute levels
        ))),
        _ => None,
    }
}

/// Get all available format IDs
pub fn available_formats() -> Vec<&'static str> {
    vec!["cash", "sng", "mtt"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cash_game_format() {
        let cash = CashGame::new(50, 100, 9);

        assert!(cash.can_join());
        assert!(cash.can_cash_out());
        assert!(cash.can_top_up());
        assert!(!cash.has_increasing_blinds());
        assert!(!cash.eliminates_players());
        assert_eq!(cash.current_blinds(), (50, 100));
    }

    #[test]
    fn test_sng_format() {
        let mut sng = SitAndGo::new(100, 1500, 6, 300);

        assert!(sng.can_join());
        assert!(!sng.can_cash_out());
        assert!(!sng.can_top_up());
        assert!(sng.has_increasing_blinds());
        assert!(sng.eliminates_players());
        assert_eq!(*sng.status(), TournamentStatus::Registering);

        // Register players
        for _ in 0..5 {
            assert!(sng.register_player());
            assert_eq!(*sng.status(), TournamentStatus::Registering);
        }

        // Last player starts the tournament
        assert!(sng.register_player());
        assert_eq!(*sng.status(), TournamentStatus::Running);

        // Can't register more
        assert!(!sng.register_player());
        assert!(!sng.can_join());
    }

    #[test]
    fn test_blind_schedule() {
        let mut schedule = BlindSchedule::standard_tournament(100, 600);

        assert_eq!(schedule.current().small_blind, 50);
        assert_eq!(schedule.current().big_blind, 100);

        assert!(schedule.advance());
        assert_eq!(schedule.current().small_blind, 100);
        assert_eq!(schedule.current().big_blind, 200);
    }

    #[test]
    fn test_prize_structure() {
        let prize = PrizeStructure::nine_player();
        let pool = 9000;

        assert_eq!(prize.payout_for_position(0, pool), 4500); // 50%
        assert_eq!(prize.payout_for_position(1, pool), 2700); // 30%
        assert_eq!(prize.payout_for_position(2, pool), 1800); // 20%
        assert_eq!(prize.payout_for_position(3, pool), 0); // Not in the money
    }
}
