mod actions;
mod blinds;
mod dealing;
mod phase;
mod player_mgmt;
mod showdown;
mod state;
mod tournament;

pub use state::{PublicPlayerState, PublicPot, PublicTableState, TournamentInfo};

use super::{
    constants::{
        DEFAULT_MAX_SEATS, DEFAULT_SHOWDOWN_DELAY_MS, DEFAULT_STREET_DELAY_MS,
        HEADS_UP_PLAYER_COUNT, MAX_RAISES_PER_ROUND, MIN_PLAYERS_TO_START,
        MTT_WAITING_REBALANCE_MS,
    },
    deck::{Card, Deck},
    error::{GameError, GameResult},
    format::{CashGame, GameFormat},
    hand::{determine_winners, evaluate_hand, HandRank, LowHandRank},
    player::{Player, PlayerAction, PlayerState},
    pot::PotManager,
    variant::{PokerVariant, TexasHoldem},
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current timestamp in milliseconds since UNIX epoch.
/// Returns 0 on system clock error (should never happen in practice).
pub(crate) fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|e| {
            tracing::error!("System clock error: {}", e);
            0
        })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GamePhase {
    Waiting,  // Waiting for players
    PreFlop,  // Hole cards dealt, pre-flop betting
    Flop,     // 3 community cards, betting
    Turn,     // 4th community card, betting
    River,    // 5th community card, betting
    Showdown, // Reveal and determine winner
}

impl GamePhase {
    /// Returns the set of phases this phase can transition to.
    pub fn valid_transitions(&self) -> &[GamePhase] {
        match self {
            GamePhase::Waiting => &[GamePhase::PreFlop],
            GamePhase::PreFlop => &[GamePhase::Flop, GamePhase::Showdown],
            GamePhase::Flop => &[GamePhase::Turn, GamePhase::Showdown],
            GamePhase::Turn => &[GamePhase::River, GamePhase::Showdown],
            GamePhase::River => &[GamePhase::Showdown],
            GamePhase::Showdown => &[GamePhase::Waiting, GamePhase::PreFlop],
        }
    }

    /// Attempt to transition to a target phase. Returns error if the transition is invalid.
    pub fn transition_to(&self, target: GamePhase) -> Result<GamePhase, GameError> {
        if self.valid_transitions().contains(&target) {
            Ok(target)
        } else {
            tracing::error!(
                "Invalid phase transition: {:?} -> {:?} (valid: {:?})",
                self,
                target,
                self.valid_transitions()
            );
            Err(GameError::InvalidPhaseTransition {
                from: format!("{:?}", self),
                to: format!("{:?}", target),
            })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PokerTable {
    pub table_id: String,
    pub name: String,
    pub small_blind: i64,
    pub big_blind: i64,
    pub players: Vec<Player>,
    pub max_seats: usize,
    pub last_winner_message: Option<String>,
    pub winning_hand: Option<String>, // Description of the winning hand
    pub dealer_seat: usize,
    pub current_player: usize,
    pub phase: GamePhase,
    pub community_cards: Vec<Card>,
    pub deck: Deck,
    pub pot: PotManager,
    pub current_bet: i64,
    pub min_raise: i64,
    pub ante: i64,                // Ante amount (0 if no ante)
    pub raises_this_round: usize, // Number of raises in the current betting round (for fixed-limit)
    pub last_phase_change_time: Option<u64>,
    pub street_delay_ms: u64,          // Delay between flop/turn/river
    pub showdown_delay_ms: u64,        // Delay to show results
    pub tournament_id: Option<String>, // If this is a tournament table
    /// Buffer of user_ids eliminated since last drain (used by tournament lifecycle)
    #[serde(skip)]
    pub pending_eliminations: Vec<String>,
    pub won_without_showdown: bool,
    #[serde(skip, default = "default_variant")]
    variant: Box<dyn PokerVariant>,
    #[serde(skip, default = "default_format")]
    format: Box<dyn GameFormat>,
}

fn default_variant() -> Box<dyn PokerVariant> {
    Box::new(TexasHoldem)
}

fn default_format() -> Box<dyn GameFormat> {
    Box::new(CashGame::new(50, 100, DEFAULT_MAX_SEATS))
}

impl Clone for PokerTable {
    fn clone(&self) -> Self {
        Self {
            table_id: self.table_id.clone(),
            name: self.name.clone(),
            small_blind: self.small_blind,
            big_blind: self.big_blind,
            players: self.players.clone(),
            max_seats: self.max_seats,
            last_winner_message: self.last_winner_message.clone(),
            winning_hand: self.winning_hand.clone(),
            dealer_seat: self.dealer_seat,
            current_player: self.current_player,
            phase: self.phase.clone(),
            community_cards: self.community_cards.clone(),
            deck: self.deck.clone(),
            pot: self.pot.clone(),
            current_bet: self.current_bet,
            min_raise: self.min_raise,
            ante: self.ante,
            raises_this_round: self.raises_this_round,
            last_phase_change_time: self.last_phase_change_time,
            street_delay_ms: self.street_delay_ms,
            showdown_delay_ms: self.showdown_delay_ms,
            tournament_id: self.tournament_id.clone(),
            pending_eliminations: self.pending_eliminations.clone(),
            won_without_showdown: self.won_without_showdown,
            variant: self.variant.clone_box(),
            format: self.format.clone_box(),
        }
    }
}

impl PokerTable {
    /// Build the per-player bet vector used by pot calculations.
    pub(crate) fn player_bets(&self) -> Vec<(usize, i64, bool)> {
        self.players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.total_bet_this_hand > 0)
            .map(|(idx, p)| (idx, p.total_bet_this_hand, p.is_active_in_hand()))
            .collect()
    }

    pub fn new(table_id: String, name: String, small_blind: i64, big_blind: i64) -> Self {
        Self::with_max_seats(table_id, name, small_blind, big_blind, DEFAULT_MAX_SEATS)
    }

    pub fn with_max_seats(
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        max_seats: usize,
    ) -> Self {
        Self::with_variant(
            table_id,
            name,
            small_blind,
            big_blind,
            max_seats,
            Box::new(TexasHoldem),
        )
    }

    /// Create a table with a specific poker variant
    pub fn with_variant(
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        max_seats: usize,
        variant: Box<dyn PokerVariant>,
    ) -> Self {
        let format = Box::new(CashGame::new(small_blind, big_blind, max_seats));
        Self::with_variant_and_format(
            table_id,
            name,
            small_blind,
            big_blind,
            max_seats,
            variant,
            format,
        )
    }

    /// Create a table with a specific variant and game format
    pub fn with_variant_and_format(
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        max_seats: usize,
        variant: Box<dyn PokerVariant>,
        format: Box<dyn GameFormat>,
    ) -> Self {
        Self {
            table_id,
            name,
            small_blind,
            big_blind,
            players: Vec::new(),
            max_seats,
            last_winner_message: None,
            winning_hand: None,
            dealer_seat: 0,
            current_player: 0,
            phase: GamePhase::Waiting,
            community_cards: Vec::new(),
            deck: Deck::new(),
            pot: PotManager::new(),
            current_bet: 0,
            min_raise: big_blind,
            ante: 0,
            raises_this_round: 0,
            last_phase_change_time: None,
            street_delay_ms: DEFAULT_STREET_DELAY_MS,
            showdown_delay_ms: DEFAULT_SHOWDOWN_DELAY_MS,
            tournament_id: None,
            pending_eliminations: Vec::new(),
            won_without_showdown: false,
            variant,
            format,
        }
    }

    /// Set the tournament ID for this table
    pub fn set_tournament_id(&mut self, tournament_id: Option<String>) {
        self.tournament_id = tournament_id;
    }

    /// Get the variant ID of this table
    #[allow(dead_code)]
    pub fn variant_id(&self) -> &'static str {
        self.variant.id()
    }

    /// Get the variant name of this table
    #[allow(dead_code)]
    pub fn variant_name(&self) -> &'static str {
        self.variant.name()
    }

    /// Get the format ID of this table
    #[allow(dead_code)]
    pub fn format_id(&self) -> &'static str {
        self.format.format_id()
    }

    /// Get the format name of this table
    #[allow(dead_code)]
    pub fn format_name(&self) -> &str {
        self.format.name()
    }

    /// Check if the format allows players to cash out
    #[allow(dead_code)]
    pub fn can_cash_out(&self) -> bool {
        self.format.can_cash_out()
    }

    /// Check if the format allows top-ups
    #[allow(dead_code)]
    pub fn can_top_up(&self) -> bool {
        self.format.can_top_up()
    }

    /// Force start a new hand (used for tournaments after all players are seated)
    pub fn force_start_hand(&mut self) {
        if self.phase == GamePhase::Waiting {
            self.start_new_hand();
        }
    }

    pub(crate) fn active_players_count(&self) -> usize {
        self.players
            .iter()
            .filter(|p| p.state == PlayerState::Active || p.state == PlayerState::WaitingForHand)
            .count()
    }

    fn first_player_index_by_seat<F>(&self, mut eligible: F) -> Option<usize>
    where
        F: FnMut(&Player) -> bool,
    {
        if self.players.is_empty() || self.max_seats == 0 {
            return None;
        }

        for seat in 0..self.max_seats {
            if let Some((idx, player)) = self
                .players
                .iter()
                .enumerate()
                .find(|(_, p)| p.seat == seat)
            {
                if eligible(player) {
                    return Some(idx);
                }
            }
        }

        None
    }

    fn next_player_index_by_seat<F>(&self, after: usize, mut eligible: F) -> Option<usize>
    where
        F: FnMut(&Player) -> bool,
    {
        if self.players.is_empty() || self.max_seats == 0 {
            return None;
        }

        let start_idx = after.min(self.players.len() - 1);
        let start_seat = self.players[start_idx].seat;

        for offset in 1..=self.max_seats {
            let seat = (start_seat + offset) % self.max_seats;
            if let Some((idx, player)) = self
                .players
                .iter()
                .enumerate()
                .find(|(_, p)| p.seat == seat)
            {
                if eligible(player) {
                    return Some(idx);
                }
            }
        }

        None
    }

    pub(crate) fn next_active_player(&self, after: usize) -> usize {
        if self.players.is_empty() {
            tracing::warn!("next_active_player called with no players");
            return 0;
        }

        tracing::debug!(
            "next_active_player: after_idx={}, after_seat={}, num_players={}",
            after,
            self.players[after.min(self.players.len() - 1)].seat,
            self.players.len()
        );

        // In tournaments, include sitting out players so they can be auto-folded.
        // In all game types, include disconnected players so they can be auto-folded.
        let next = self.next_player_index_by_seat(after, |player| {
            if self.format.eliminates_players() {
                player.can_act()
                    || player.state == PlayerState::SittingOut
                    || player.state == PlayerState::Disconnected
            } else {
                player.can_act() || player.state == PlayerState::Disconnected
            }
        });

        if let Some(idx) = next {
            tracing::info!(
                "next_active_player: returning idx={} (seat {}, {})",
                idx,
                self.players[idx].seat,
                self.players[idx].username
            );
            idx
        } else {
            tracing::warn!(
                "next_active_player: No active players found! Returning fallback {}",
                after
            );
            after.min(self.players.len() - 1)
        }
    }

    /// Find the first player eligible for dealer button starting from seat 0
    /// Dealer button: must have chips and not be sitting out, eliminated, or disconnected
    pub(crate) fn first_eligible_player_for_button(&self) -> usize {
        let next = self.first_player_index_by_seat(|player| player.is_eligible_for_button());

        if let Some(idx) = next {
            tracing::info!(
                "first_eligible_player_for_button: returning idx={} (seat {}, {})",
                idx,
                self.players[idx].seat,
                self.players[idx].username
            );
            idx
        } else {
            tracing::warn!(
                "first_eligible_player_for_button: No eligible players found! Returning 0"
            );
            0 // Fallback
        }
    }

    /// Find the next player eligible for dealer button
    /// Dealer button: must have chips and not be sitting out, eliminated, or disconnected
    pub(crate) fn next_eligible_player_for_button(&self, after: usize) -> usize {
        if self.players.is_empty() {
            tracing::warn!("next_eligible_player_for_button: No players, returning 0");
            return 0;
        }

        tracing::debug!(
            "next_eligible_player_for_button: after_idx={}, after_seat={}, num_players={}",
            after,
            self.players[after.min(self.players.len() - 1)].seat,
            self.players.len()
        );

        let next = self.next_player_index_by_seat(after, |player| player.is_eligible_for_button());

        if let Some(idx) = next {
            tracing::info!(
                "next_eligible_player_for_button: returning idx={} (seat {}, {})",
                idx,
                self.players[idx].seat,
                self.players[idx].username
            );
            idx
        } else {
            tracing::warn!(
                "next_eligible_player_for_button: No eligible players found! Returning fallback {}",
                after
            );
            after.min(self.players.len() - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::variant::{variant_from_id, FixedLimitHoldem, OmahaHi, PotLimitOmaha};

    #[test]
    fn test_table_default_variant() {
        let table = PokerTable::new("t1".to_string(), "Test Table".to_string(), 5, 10);
        assert_eq!(table.variant_id(), "holdem");
        assert_eq!(table.variant_name(), "Texas Hold'em");
    }

    #[test]
    fn test_table_with_omaha_variant() {
        let table = PokerTable::with_variant(
            "t2".to_string(),
            "Omaha Table".to_string(),
            5,
            10,
            9,
            Box::new(OmahaHi),
        );
        assert_eq!(table.variant_id(), "omaha");
        assert_eq!(table.variant_name(), "Omaha");
    }

    #[test]
    fn test_table_with_variant_factory() {
        let variant = variant_from_id("omaha_hilo").expect("Should find variant");
        let table = PokerTable::with_variant(
            "t3".to_string(),
            "Omaha Hi-Lo Table".to_string(),
            5,
            10,
            9,
            variant,
        );
        assert_eq!(table.variant_id(), "omaha_hilo");
    }

    #[test]
    fn test_table_with_plo_variant() {
        let table = PokerTable::with_variant(
            "t3b".to_string(),
            "PLO Table".to_string(),
            5,
            10,
            9,
            Box::new(PotLimitOmaha),
        );
        assert_eq!(table.variant_id(), "plo");
        assert_eq!(table.variant_name(), "Pot Limit Omaha");
    }

    #[test]
    fn test_plo_raise_cannot_exceed_pot_limit() {
        let mut table = PokerTable::with_variant(
            "t3c".to_string(),
            "PLO Table".to_string(),
            25,
            50,
            2,
            Box::new(PotLimitOmaha),
        );

        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();

        table.pot = PotManager::new();
        table.current_player = 0;
        table.current_bet = 50;
        table.min_raise = 50;
        table.players[0].current_bet = 25;
        table.players[1].current_bet = 50;
        table.players[0].stack = 1000;
        table.players[1].stack = 1000;
        table.pot.add_bet(0, 25);
        table.pot.add_bet(1, 50);

        let result = table.handle_action("p1", PlayerAction::Raise(150));
        assert!(
            matches!(result, Err(GameError::RaiseTooLarge { .. })),
            "unexpected result: {:?}",
            result
        );
    }

    #[test]
    fn test_public_state_includes_variant() {
        let table = PokerTable::with_variant(
            "t4".to_string(),
            "Test".to_string(),
            5,
            10,
            6,
            Box::new(OmahaHi),
        );
        let state = table.get_public_state(None);
        assert_eq!(state.variant_id, "omaha");
        assert_eq!(state.variant_name, "Omaha");
    }

    #[test]
    fn test_table_clone_preserves_variant() {
        let table = PokerTable::with_variant(
            "t5".to_string(),
            "Clone Test".to_string(),
            5,
            10,
            9,
            Box::new(OmahaHi),
        );
        let cloned = table.clone();
        assert_eq!(cloned.variant_id(), "omaha");
        assert_eq!(cloned.variant_name(), "Omaha");
    }

    #[test]
    fn test_table_default_format() {
        let table = PokerTable::new("t6".to_string(), "Test".to_string(), 5, 10);
        assert_eq!(table.format_id(), "cash");
        assert!(table.can_cash_out());
        assert!(table.can_top_up());
    }

    #[test]
    fn test_public_state_includes_format() {
        let table = PokerTable::new("t7".to_string(), "Test".to_string(), 25, 50);
        let state = table.get_public_state(None);
        assert_eq!(state.format_id, "cash");
        assert!(state.can_cash_out);
        assert!(state.can_top_up);
    }

    #[test]
    fn test_table_with_sng_format() {
        use crate::game::format::SitAndGo;

        let sng = SitAndGo::new(100, 1500, 6, 300);
        let table = PokerTable::with_variant_and_format(
            "t8".to_string(),
            "SNG Test".to_string(),
            25,
            50,
            6,
            Box::new(TexasHoldem),
            Box::new(sng),
        );

        assert_eq!(table.format_id(), "sng");
        assert!(!table.can_cash_out());
        assert!(!table.can_top_up());
    }

    #[test]
    fn test_first_hand_dealer_position() {
        // Test that the first hand assigns dealer to the first eligible player
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);

        // Add 3 players
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000)
            .unwrap();

        // Table should have started the hand automatically
        assert_eq!(table.phase, GamePhase::PreFlop);

        // Dealer should be at position 0 (first eligible player)
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(table.players[0].username, "Player 1");
    }

    #[test]
    fn test_blinds_posted_correctly() {
        // Test that SB and BB are posted by the correct players in a 3-player game
        use crate::game::format::SitAndGo;

        // Create table with SNG format to prevent auto-start
        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        // Add 3 players at seats 0, 1, 2
        println!("Adding players...");
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        println!(
            "After adding p1, phase={:?}, players.len()={}",
            table.phase,
            table.players.len()
        );

        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        println!(
            "After adding p2, phase={:?}, players.len()={}",
            table.phase,
            table.players.len()
        );

        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000)
            .unwrap();
        println!(
            "After adding p3, phase={:?}, players.len()={}",
            table.phase,
            table.players.len()
        );

        // Game should NOT have auto-started (SNG format)
        assert_eq!(table.phase, GamePhase::Waiting);

        // Force start the hand
        table.force_start_hand();

        // Verify game started
        assert_eq!(table.phase, GamePhase::PreFlop);

        // Debug: print all players and their bets
        println!("Dealer at array index: {}", table.dealer_seat);
        for (idx, player) in table.players.iter().enumerate() {
            println!(
                "Player[{}]: name={}, seat={}, bet={}, stack={}",
                idx, player.username, player.seat, player.current_bet, player.stack
            );
        }

        // Dealer at array position 0, so:
        // - SB should be at array position 1 (Player 2)
        // - BB should be at array position 2 (Player 3)
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(
            table.players[1].current_bet, 50,
            "Player at index 1 should have posted SB"
        );
        assert_eq!(
            table.players[2].current_bet, 100,
            "Player at index 2 should have posted BB"
        );
        assert_eq!(
            table.players[0].current_bet, 0,
            "Dealer at index 0 should not have posted"
        );

        // Verify stacks are reduced correctly
        assert_eq!(
            table.players[0].stack, 1000,
            "Dealer stack should be unchanged"
        );
        assert_eq!(
            table.players[1].stack, 950,
            "SB stack should be reduced by 50"
        );
        assert_eq!(
            table.players[2].stack, 900,
            "BB stack should be reduced by 100"
        );
    }

    #[test]
    fn test_first_to_act_after_blinds() {
        // Test that the first player to act is the one after BB
        use crate::game::format::SitAndGo;

        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        // Add 4 players
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000)
            .unwrap();
        table
            .take_seat("p4".to_string(), "Player 4".to_string(), 3, 1000)
            .unwrap();

        // Force start
        table.force_start_hand();

        // Dealer at position 0, SB at 1, BB at 2
        // First to act should be position 3 (after BB)
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(table.current_player, 3, "First to act should be position 3");
        assert_eq!(table.players[table.current_player].username, "Player 4");
    }

    #[test]
    fn test_heads_up_blind_positions_simple() {
        // Test heads-up setup in isolation
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);

        // Add exactly 2 players
        println!("Adding first player...");
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        println!(
            "Phase after p1: {:?}, players.len()={}",
            table.phase,
            table.players.len()
        );

        println!("Adding second player...");
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        println!(
            "Phase after p2: {:?}, players.len()={}",
            table.phase,
            table.players.len()
        );

        // Game should have started
        assert_eq!(table.phase, GamePhase::PreFlop);
        assert_eq!(table.players.len(), 2);

        println!("Dealer at index: {}", table.dealer_seat);
        for (idx, p) in table.players.iter().enumerate() {
            println!(
                "Player[{}]: seat={}, name={}, bet={}, stack={}",
                idx, p.seat, p.username, p.current_bet, p.stack
            );
        }

        // In heads-up: dealer posts SB, other player posts BB
        // dealer_seat should be 0
        assert_eq!(table.dealer_seat, 0, "Dealer should be at index 0");

        // Dealer (index 0) should post SB (50)
        // Non-dealer (index 1) should post BB (100)
        assert_eq!(
            table.players[0].current_bet, 50,
            "Dealer should post SB in heads-up"
        );
        assert_eq!(
            table.players[1].current_bet, 100,
            "Non-dealer should post BB in heads-up"
        );
    }

    #[test]
    fn test_heads_up_blind_positions() {
        // In heads-up (2 players), dealer posts SB and acts first pre-flop
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);

        // Add 2 players
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();

        // Dealer at position 0
        assert_eq!(table.dealer_seat, 0);

        // In heads-up: dealer (pos 0) posts SB, other player (pos 1) posts BB
        assert_eq!(
            table.players[0].current_bet, 50,
            "Dealer should post SB in heads-up"
        );
        assert_eq!(
            table.players[1].current_bet, 100,
            "Non-dealer should post BB in heads-up"
        );

        // In heads-up, dealer acts first pre-flop (after posting SB)
        assert_eq!(
            table.current_player, 0,
            "Dealer should act first in heads-up"
        );
    }

    #[test]
    fn test_nine_player_sng_blinds() {
        // Test a 9-player SNG starting positions
        use crate::game::format::SitAndGo;

        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        // Add 9 players
        for i in 0..9 {
            table
                .take_seat(format!("p{}", i), format!("Player {}", i + 1), i, 1000)
                .unwrap();
        }

        // Force start
        table.force_start_hand();

        // Dealer should be at position 0
        assert_eq!(table.dealer_seat, 0);

        // SB at position 1, BB at position 2
        assert_eq!(
            table.players[1].current_bet, 50,
            "Position 1 should post SB"
        );
        assert_eq!(
            table.players[2].current_bet, 100,
            "Position 2 should post BB"
        );

        // Verify only SB and BB have posted
        assert_eq!(table.players[0].current_bet, 0);
        for i in 3..9 {
            assert_eq!(
                table.players[i].current_bet, 0,
                "Position {} should not have posted",
                i
            );
        }

        // First to act should be position 3
        assert_eq!(table.current_player, 3);

        // All players should have cards
        for i in 0..9 {
            assert_eq!(
                table.players[i].hole_cards.len(),
                2,
                "Player {} should have 2 cards",
                i
            );
        }
    }

    #[test]
    fn test_dealer_advances_between_hands() {
        // Test that dealer button moves correctly between hands
        use crate::game::format::SitAndGo;

        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        // Add 3 players
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000)
            .unwrap();

        // Force start first hand
        table.force_start_hand();

        // First hand - dealer at position 0
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(
            table.players[1].current_bet, 50,
            "First hand: Position 1 should post SB"
        );
        assert_eq!(
            table.players[2].current_bet, 100,
            "First hand: Position 2 should post BB"
        );

        // Fast-forward to showdown and start new hand
        table.phase = GamePhase::Showdown;
        table.start_new_hand();

        // Second hand - dealer should move to position 1
        assert_eq!(table.dealer_seat, 1);
        assert_eq!(
            table.players[2].current_bet, 50,
            "Second hand: Position 2 should now post SB"
        );
        assert_eq!(
            table.players[0].current_bet, 100,
            "Second hand: Position 0 should now post BB"
        );
    }

    #[test]
    fn test_tournament_button_and_blinds_rotate_by_physical_seat_order() {
        // Regression: player vec order can differ from seat order (e.g. table balancing moves).
        // Button/SB/BB must rotate by physical seats, not insertion order.
        use crate::game::format::SitAndGo;

        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        // Intentionally seat players in non-sorted insertion order.
        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 5, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 8, 1000)
            .unwrap();

        table.force_start_hand();
        let dealer_seat_num = table.players[table.dealer_seat].seat;
        let sb_idx = table.next_player_for_blind(table.dealer_seat);
        let bb_idx = table.next_player_for_blind(sb_idx);
        assert_eq!(
            dealer_seat_num, 1,
            "First dealer should be lowest eligible seat"
        );
        assert_eq!(
            table.players[sb_idx].seat, 5,
            "SB should be next clockwise seat"
        );
        assert_eq!(
            table.players[bb_idx].seat, 8,
            "BB should be next clockwise seat"
        );
        assert_eq!(
            table.players[table.current_player].seat, 1,
            "First to act preflop should be seat after BB"
        );

        table.phase = GamePhase::Showdown;
        table.start_new_hand();
        let dealer_seat_num = table.players[table.dealer_seat].seat;
        let sb_idx = table.next_player_for_blind(table.dealer_seat);
        let bb_idx = table.next_player_for_blind(sb_idx);
        assert_eq!(dealer_seat_num, 5, "Dealer should rotate clockwise");
        assert_eq!(table.players[sb_idx].seat, 8, "SB should rotate clockwise");
        assert_eq!(table.players[bb_idx].seat, 1, "BB should rotate clockwise");

        table.phase = GamePhase::Showdown;
        table.start_new_hand();
        let dealer_seat_num = table.players[table.dealer_seat].seat;
        let sb_idx = table.next_player_for_blind(table.dealer_seat);
        let bb_idx = table.next_player_for_blind(sb_idx);
        assert_eq!(dealer_seat_num, 8, "Dealer should continue rotating");
        assert_eq!(table.players[sb_idx].seat, 1, "SB should continue rotating");
        assert_eq!(table.players[bb_idx].seat, 5, "BB should continue rotating");
    }

    #[test]
    fn test_waiting_between_hands_still_rotates_dealer_button() {
        // Regression: MTT flow transitions Showdown -> Waiting -> start_new_hand.
        // Dealer must rotate on this Waiting path, not reset to first seat each hand.
        use crate::game::format::SitAndGo;

        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000)
            .unwrap();

        table.force_start_hand();
        assert_eq!(table.players[table.dealer_seat].seat, 0);

        // Simulate showdown completion entering Waiting with a timestamp.
        table.phase = GamePhase::Waiting;
        table.last_phase_change_time = Some(1);

        table.start_new_hand();

        // Dealer should advance to next seat, not reset to seat 0.
        assert_eq!(table.players[table.dealer_seat].seat, 1);
    }

    #[test]
    fn test_all_players_receive_hole_cards() {
        // Test that all players receive the correct number of hole cards
        use crate::game::format::SitAndGo;

        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );

        // Add 5 players
        for i in 0..5 {
            table
                .take_seat(format!("p{}", i), format!("Player {}", i + 1), i, 1000)
                .unwrap();
        }

        // Force start
        table.force_start_hand();

        // All 5 players should have 2 hole cards each
        for i in 0..5 {
            assert_eq!(
                table.players[i].hole_cards.len(),
                2,
                "Player {} should have 2 hole cards",
                i + 1
            );
        }

        // Verify no duplicate cards between players
        let mut all_cards = Vec::new();
        for player in &table.players {
            for card in &player.hole_cards {
                assert!(
                    !all_cards.contains(card),
                    "Duplicate card dealt: {:?}",
                    card
                );
                all_cards.push(card.clone());
            }
        }
    }

    // Helper to create a fixed-limit table in PreFlop with 3 players ready to act
    fn setup_fixed_limit_table() -> PokerTable {
        use crate::game::format::SitAndGo;

        // FL Hold'em with small_bet=100, big_bet=200 (blinds 50/100)
        let fl = FixedLimitHoldem::new(100, 200);
        let sng_format = Box::new(SitAndGo::new(100, 5000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "fl_test".to_string(),
            "FL Test".to_string(),
            50,
            100,
            9,
            Box::new(fl),
            sng_format,
        );

        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 5000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 5000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 5000)
            .unwrap();
        table.force_start_hand();

        assert_eq!(table.phase, GamePhase::PreFlop);
        table
    }

    #[test]
    fn test_fixed_limit_raise_exact_amount_preflop() {
        let mut table = setup_fixed_limit_table();
        // PreFlop: small_bet=100 required for raises
        // current_player should be at position 0 (after BB)
        let current_user = table.players[table.current_player].user_id.clone();
        let result = table.handle_action(&current_user, PlayerAction::Raise(100));
        assert!(
            result.is_ok(),
            "Raising exactly small_bet should succeed: {:?}",
            result
        );
        assert_eq!(table.raises_this_round, 1);
    }

    #[test]
    fn test_fixed_limit_raise_wrong_amount_preflop() {
        let mut table = setup_fixed_limit_table();
        let current_user = table.players[table.current_player].user_id.clone();
        // Try raising 200 (big_bet) on PreFlop -- should fail, need small_bet=100
        let result = table.handle_action(&current_user, PlayerAction::Raise(200));
        assert!(
            matches!(
                result,
                Err(GameError::RaiseNotExact {
                    required: 100,
                    attempted: 200
                })
            ),
            "Wrong raise amount should fail: {:?}",
            result
        );
    }

    #[test]
    fn test_fixed_limit_max_raises_per_round() {
        let mut table = setup_fixed_limit_table();
        // Do 4 raises (bet + 3 re-raises), then the 5th should fail
        for i in 0..4 {
            let current_user = table.players[table.current_player].user_id.clone();
            let result = table.handle_action(&current_user, PlayerAction::Raise(100));
            assert!(
                result.is_ok(),
                "Raise {} should succeed: {:?}",
                i + 1,
                result
            );
        }
        assert_eq!(table.raises_this_round, 4);

        // 5th raise should fail
        let current_user = table.players[table.current_player].user_id.clone();
        let result = table.handle_action(&current_user, PlayerAction::Raise(100));
        assert!(
            matches!(result, Err(GameError::MaxRaisesReached { max_raises: 4 })),
            "5th raise should be rejected: {:?}",
            result
        );
    }

    #[test]
    fn test_fixed_limit_raises_reset_on_new_street() {
        let mut table = setup_fixed_limit_table();
        // Raise once on PreFlop
        let user = table.players[table.current_player].user_id.clone();
        table
            .handle_action(&user, PlayerAction::Raise(100))
            .unwrap();
        assert_eq!(table.raises_this_round, 1);

        // Have remaining players call to advance to Flop
        loop {
            if table.phase != GamePhase::PreFlop {
                break;
            }
            let user = table.players[table.current_player].user_id.clone();
            table.handle_action(&user, PlayerAction::Call).unwrap();
        }

        // Should be on Flop now with raises reset
        assert_eq!(table.phase, GamePhase::Flop);
        assert_eq!(table.raises_this_round, 0);
    }

    #[test]
    fn test_fixed_limit_big_bet_on_turn() {
        let mut table = setup_fixed_limit_table();

        // Advance to flop: everyone calls preflop
        loop {
            if table.phase != GamePhase::PreFlop {
                break;
            }
            let user = table.players[table.current_player].user_id.clone();
            table.handle_action(&user, PlayerAction::Call).unwrap();
        }
        assert_eq!(table.phase, GamePhase::Flop);

        // Advance to turn: everyone checks on flop
        loop {
            if table.phase != GamePhase::Flop {
                break;
            }
            let user = table.players[table.current_player].user_id.clone();
            table.handle_action(&user, PlayerAction::Check).unwrap();
        }
        assert_eq!(table.phase, GamePhase::Turn);

        // On Turn, must raise exactly big_bet=200
        let user = table.players[table.current_player].user_id.clone();
        // small_bet should fail
        let result = table.handle_action(&user, PlayerAction::Raise(100));
        assert!(
            matches!(
                result,
                Err(GameError::RaiseNotExact {
                    required: 200,
                    attempted: 100
                })
            ),
            "small_bet on turn should fail: {:?}",
            result
        );
        // big_bet should succeed
        let result = table.handle_action(&user, PlayerAction::Raise(200));
        assert!(
            result.is_ok(),
            "big_bet on turn should succeed: {:?}",
            result
        );
    }

    #[test]
    fn test_preflop_all_fold_to_raise_ends_hand_without_board() {
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);

        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 5000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 5000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 5000)
            .unwrap();

        assert_eq!(table.phase, GamePhase::PreFlop);

        let raiser = table.players[table.current_player].user_id.clone();
        table
            .handle_action(&raiser, PlayerAction::Raise(100))
            .unwrap();

        // Everyone else folds; raiser should not need to act again.
        while table.phase == GamePhase::PreFlop {
            let to_act = table.players[table.current_player].user_id.clone();
            assert_ne!(
                to_act, raiser,
                "Raiser should not get another turn once all other players fold"
            );
            table.handle_action(&to_act, PlayerAction::Fold).unwrap();
        }

        assert_eq!(table.phase, GamePhase::Showdown);
        assert!(table.won_without_showdown);
        assert!(
            table.community_cards.is_empty(),
            "No board cards should be dealt on fold-win"
        );

        let winners: Vec<&Player> = table.players.iter().filter(|p| p.is_winner).collect();
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].user_id, raiser);
    }

    #[test]
    fn test_preflop_all_fold_to_big_blind_auto_wins_without_acting() {
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);

        table
            .take_seat("p1".to_string(), "Player 1".to_string(), 0, 5000)
            .unwrap();
        table
            .take_seat("p2".to_string(), "Player 2".to_string(), 1, 5000)
            .unwrap();
        table
            .take_seat("p3".to_string(), "Player 3".to_string(), 2, 5000)
            .unwrap();
        table
            .take_seat("p4".to_string(), "Player 4".to_string(), 3, 5000)
            .unwrap();

        assert_eq!(table.phase, GamePhase::PreFlop);

        let bb_user = table
            .players
            .iter()
            .find(|p| p.current_bet == table.big_blind)
            .expect("Expected exactly one big blind poster in 4-handed preflop")
            .user_id
            .clone();

        // Fold everyone who gets to act before the BB.
        while table.phase == GamePhase::PreFlop {
            let to_act = table.players[table.current_player].user_id.clone();
            assert_ne!(
                to_act, bb_user,
                "Big blind should win immediately once all other players fold"
            );
            table.handle_action(&to_act, PlayerAction::Fold).unwrap();
        }

        // BB should win immediately without needing to act.
        assert_eq!(table.phase, GamePhase::Showdown);
        assert!(table.won_without_showdown);
        assert!(
            table.community_cards.is_empty(),
            "No board cards should be dealt on fold-win to BB"
        );

        let winners: Vec<&Player> = table.players.iter().filter(|p| p.is_winner).collect();
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].user_id, bb_user);
    }
}
