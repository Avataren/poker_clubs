//! Betting Engine
//!
//! Handles all betting-related logic separate from the main table state.
//! This includes validation of actions, execution of bets, and tracking
//! of betting rounds.
//!
//! Benefits of separation:
//! - Easier testing of betting logic in isolation
//! - Cleaner table code focused on game flow
//! - Reusable across different game formats (cash, tournament)
//! - Foundation for different betting structures (NL, PL, FL)

// Allow dead code during development - these components will be integrated incrementally
#![allow(dead_code)]

use super::error::{GameError, GameResult};
use super::player::{Player, PlayerAction};
use super::pot::PotManager;
use super::variant::BettingStructure;

/// Tracks the state of the current betting round
#[derive(Debug, Clone)]
pub struct BettingRound {
    /// The current bet amount that must be matched
    pub current_bet: i64,
    /// The minimum raise amount
    pub min_raise: i64,
    /// The big blind for this game
    pub big_blind: i64,
    /// Betting structure (NL, PL, FL)
    pub structure: BettingStructure,
}

impl BettingRound {
    pub fn new(big_blind: i64, structure: BettingStructure) -> Self {
        Self {
            current_bet: 0,
            min_raise: big_blind,
            big_blind,
            structure,
        }
    }

    /// Reset for a new betting round (new street)
    pub fn reset(&mut self) {
        self.current_bet = 0;
        self.min_raise = self.big_blind;
    }

    /// Set initial bet after blinds are posted
    pub fn set_after_blinds(&mut self, big_blind_amount: i64) {
        self.current_bet = big_blind_amount;
        self.min_raise = big_blind_amount;
    }
}

/// Validates betting actions without modifying state
pub struct BettingValidator;

impl BettingValidator {
    /// Validate that a player can perform the given action
    pub fn validate_action(
        player: &Player,
        action: &PlayerAction,
        round: &BettingRound,
        pot: &PotManager,
    ) -> GameResult<()> {
        if !player.can_act() {
            return Err(GameError::CannotAct);
        }

        match action {
            PlayerAction::Fold => {
                // Can always fold
                Ok(())
            }
            PlayerAction::Check => {
                if round.current_bet > player.current_bet {
                    return Err(GameError::CannotCheck {
                        current_bet: round.current_bet,
                    });
                }
                Ok(())
            }
            PlayerAction::Call => {
                // Can always call (may result in all-in if insufficient chips)
                Ok(())
            }
            PlayerAction::Raise(amount) => Self::validate_raise(player, *amount, round, pot),
            PlayerAction::AllIn => {
                if let BettingStructure::PotLimit = &round.structure {
                    let to_call = round.current_bet - player.current_bet;
                    let max_raise = pot.total() + to_call;
                    let desired_total = player.current_bet + player.stack;
                    let max_total = round.current_bet + max_raise;

                    if desired_total > max_total {
                        let attempted_raise = desired_total.saturating_sub(round.current_bet);
                        return Err(GameError::RaiseTooLarge {
                            max_raise,
                            attempted: attempted_raise,
                        });
                    }
                }
                Ok(())
            }
            PlayerAction::ShowCards(_) => {
                // ShowCards is handled separately, not a betting action
                Err(GameError::InvalidAction {
                    reason: "ShowCards is not a betting action".to_string(),
                })
            }
        }
    }

    fn validate_raise(
        player: &Player,
        amount: i64,
        round: &BettingRound,
        pot: &PotManager,
    ) -> GameResult<()> {
        if amount <= 0 {
            return Err(GameError::RaiseTooSmall {
                min_raise: round.min_raise,
                attempted: amount,
            });
        }

        // Check minimum raise
        if amount < round.min_raise {
            return Err(GameError::RaiseTooSmall {
                min_raise: round.min_raise,
                attempted: amount,
            });
        }

        // For pot-limit, check maximum
        if let BettingStructure::PotLimit = &round.structure {
            let to_call = round.current_bet - player.current_bet;
            let max_raise = pot.total() + to_call;
            if amount > max_raise {
                return Err(GameError::RaiseTooLarge {
                    max_raise,
                    attempted: amount,
                });
            }
        }

        // For fixed-limit, raise must be exactly the bet size
        // Note: The actual street-aware enforcement is done in table/actions.rs
        // which has access to the current game phase. This standalone validator
        // checks that the amount is at least one of the valid bet sizes.
        if let BettingStructure::FixedLimit { small_bet, big_bet } = &round.structure {
            if amount != *small_bet && amount != *big_bet {
                return Err(GameError::RaiseNotExact {
                    required: *small_bet, // default to small_bet in standalone context
                    attempted: amount,
                });
            }
        }

        // Check if player has enough chips
        let to_call = round.current_bet - player.current_bet;
        let total_needed = to_call + amount;
        if total_needed > player.stack {
            // This is okay - they'll just go all-in
        }

        Ok(())
    }
}

/// Executes betting actions, modifying player and pot state
pub struct BettingExecutor;

impl BettingExecutor {
    fn execute_bet(
        player: &mut Player,
        pot: &mut PotManager,
        round: &mut BettingRound,
        target_total: i64,
        requested_raise: i64,
    ) -> (i64, bool) {
        let current_player_bet = player.current_bet;
        let total_to_bet = target_total.saturating_sub(current_player_bet);
        let actual = player.place_bet(total_to_bet);
        pot.add_bet(player.seat, actual);

        let new_total = current_player_bet + actual;
        let is_raise = new_total > round.current_bet;

        if is_raise {
            let effective_raise = new_total - round.current_bet;
            if effective_raise >= round.min_raise && requested_raise > 0 {
                round.min_raise = requested_raise;
            }
            round.current_bet = new_total;
        }

        (actual, is_raise)
    }

    /// Execute a validated action
    ///
    /// Returns (chips_added_to_pot, is_raise)
    pub fn execute_action(
        player: &mut Player,
        action: PlayerAction,
        pot: &mut PotManager,
        round: &mut BettingRound,
    ) -> (i64, bool) {
        match action {
            PlayerAction::Fold => {
                player.fold();
                (0, false)
            }
            PlayerAction::Check => {
                // No chips move
                (0, false)
            }
            PlayerAction::Call => {
                let to_call = round.current_bet - player.current_bet;
                let actual = player.place_bet(to_call);
                pot.add_bet(player.seat, actual);
                (actual, false)
            }
            PlayerAction::Raise(amount) => {
                let raise_to = round.current_bet + amount;
                Self::execute_bet(player, pot, round, raise_to, amount)
            }
            PlayerAction::AllIn => {
                let target_total = player.current_bet + player.stack;
                let requested_raise = target_total.saturating_sub(round.current_bet);
                Self::execute_bet(player, pot, round, target_total, requested_raise)
            }
            PlayerAction::ShowCards(_) => {
                // ShowCards is handled separately, not a betting action
                (0, false)
            }
        }
    }

    /// Post blinds at the start of a hand
    ///
    /// Returns (small_blind_posted, big_blind_posted)
    pub fn post_blinds(
        players: &mut [Player],
        pot: &mut PotManager,
        round: &mut BettingRound,
        dealer_seat: usize,
        small_blind: i64,
        big_blind: i64,
    ) -> (i64, i64) {
        let num_players = players.len();
        if num_players == 0 {
            return (0, 0);
        }

        // Small blind (next player after dealer), heads-up: dealer posts small blind
        let sb_seat = if num_players == 2 {
            dealer_seat
        } else {
            (dealer_seat + 1) % num_players
        };
        let sb_amount = players[sb_seat].place_bet(small_blind);
        pot.add_bet(sb_seat, sb_amount);

        // Big blind
        let bb_seat = if num_players == 2 {
            (dealer_seat + 1) % num_players
        } else {
            (dealer_seat + 2) % num_players
        };
        let bb_amount = players[bb_seat].place_bet(big_blind);
        pot.add_bet(bb_seat, bb_amount);

        round.set_after_blinds(big_blind);

        (sb_amount, bb_amount)
    }
}

/// Combined betting engine that provides a clean interface
pub struct BettingEngine {
    pub round: BettingRound,
}

impl BettingEngine {
    pub fn new(big_blind: i64, structure: BettingStructure) -> Self {
        Self {
            round: BettingRound::new(big_blind, structure),
        }
    }

    /// Validate and execute an action
    pub fn process_action(
        &mut self,
        player: &mut Player,
        action: PlayerAction,
        pot: &mut PotManager,
    ) -> GameResult<bool> {
        // Validate first
        BettingValidator::validate_action(player, &action, &self.round, pot)?;

        // Execute
        let (_chips, is_raise) =
            BettingExecutor::execute_action(player, action, pot, &mut self.round);

        Ok(is_raise)
    }

    /// Get the current bet amount
    pub fn current_bet(&self) -> i64 {
        self.round.current_bet
    }

    /// Get the minimum raise amount
    pub fn min_raise(&self) -> i64 {
        self.round.min_raise
    }

    /// Reset for a new street
    pub fn new_street(&mut self) {
        self.round.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::player::PlayerState;

    fn create_test_player(seat: usize, stack: i64) -> Player {
        Player::new(
            format!("user_{}", seat),
            format!("Player {}", seat),
            seat,
            stack,
        )
    }

    #[test]
    fn test_betting_round_new() {
        let round = BettingRound::new(100, BettingStructure::NoLimit);
        assert_eq!(round.current_bet, 0);
        assert_eq!(round.min_raise, 100);
        assert_eq!(round.big_blind, 100);
    }

    #[test]
    fn test_betting_round_reset() {
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);
        round.current_bet = 500;
        round.min_raise = 300;

        round.reset();

        assert_eq!(round.current_bet, 0);
        assert_eq!(round.min_raise, 100);
    }

    #[test]
    fn test_validate_check_when_no_bet() {
        let player = create_test_player(0, 1000);
        let round = BettingRound::new(100, BettingStructure::NoLimit);
        let pot = PotManager::new();

        let result = BettingValidator::validate_action(&player, &PlayerAction::Check, &round, &pot);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_check_fails_when_bet() {
        let player = create_test_player(0, 1000);
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);
        round.current_bet = 200;
        let pot = PotManager::new();

        let result = BettingValidator::validate_action(&player, &PlayerAction::Check, &round, &pot);
        assert!(matches!(result, Err(GameError::CannotCheck { .. })));
    }

    #[test]
    fn test_validate_raise_too_small() {
        let player = create_test_player(0, 1000);
        let round = BettingRound::new(100, BettingStructure::NoLimit);
        let pot = PotManager::new();

        let result =
            BettingValidator::validate_action(&player, &PlayerAction::Raise(50), &round, &pot);
        assert!(matches!(result, Err(GameError::RaiseTooSmall { .. })));
    }

    #[test]
    fn test_validate_raise_too_large_pot_limit() {
        let mut player = create_test_player(0, 1000);
        let mut round = BettingRound::new(50, BettingStructure::PotLimit);
        round.current_bet = 50;
        round.min_raise = 50;
        player.current_bet = 25;

        let mut pot = PotManager::new();
        pot.add_bet(0, 25);
        pot.add_bet(1, 50);

        let result =
            BettingValidator::validate_action(&player, &PlayerAction::Raise(150), &round, &pot);
        assert!(matches!(result, Err(GameError::RaiseTooLarge { .. })));
    }

    #[test]
    fn test_validate_action_fails_for_inactive_player() {
        let mut player = create_test_player(0, 1000);
        player.state = PlayerState::Folded;
        let round = BettingRound::new(100, BettingStructure::NoLimit);
        let pot = PotManager::new();

        let result = BettingValidator::validate_action(&player, &PlayerAction::Check, &round, &pot);
        assert!(matches!(result, Err(GameError::CannotAct)));
    }

    #[test]
    fn test_execute_fold() {
        let mut player = create_test_player(0, 1000);
        let mut pot = PotManager::new();
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);

        let (chips, is_raise) =
            BettingExecutor::execute_action(&mut player, PlayerAction::Fold, &mut pot, &mut round);

        assert_eq!(chips, 0);
        assert!(!is_raise);
        assert_eq!(player.state, PlayerState::Folded);
    }

    #[test]
    fn test_execute_call() {
        let mut player = create_test_player(0, 1000);
        let mut pot = PotManager::new();
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);
        round.current_bet = 200;

        let (chips, is_raise) =
            BettingExecutor::execute_action(&mut player, PlayerAction::Call, &mut pot, &mut round);

        assert_eq!(chips, 200);
        assert!(!is_raise);
        assert_eq!(player.stack, 800);
        assert_eq!(player.current_bet, 200);
    }

    #[test]
    fn test_execute_raise() {
        let mut player = create_test_player(0, 1000);
        let mut pot = PotManager::new();
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);
        round.current_bet = 100; // Big blind

        let (chips, is_raise) = BettingExecutor::execute_action(
            &mut player,
            PlayerAction::Raise(200), // Raise by 200 (to 300 total)
            &mut pot,
            &mut round,
        );

        assert_eq!(chips, 300);
        assert!(is_raise);
        assert_eq!(player.stack, 700);
        assert_eq!(round.current_bet, 300);
        assert_eq!(round.min_raise, 200);
    }

    #[test]
    fn test_execute_raise_short_all_in_does_not_update_min_raise() {
        let mut player = create_test_player(0, 150);
        let mut pot = PotManager::new();
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);
        round.current_bet = 100;
        round.min_raise = 100;

        let (chips, is_raise) = BettingExecutor::execute_action(
            &mut player,
            PlayerAction::Raise(200),
            &mut pot,
            &mut round,
        );

        assert_eq!(chips, 150);
        assert!(is_raise);
        assert_eq!(round.current_bet, 150);
        assert_eq!(round.min_raise, 100);
    }

    #[test]
    fn test_betting_engine_process_action() {
        let mut engine = BettingEngine::new(100, BettingStructure::NoLimit);
        let mut player = create_test_player(0, 1000);
        let mut pot = PotManager::new();

        // Try to check when no bet - should succeed
        let result = engine.process_action(&mut player, PlayerAction::Check, &mut pot);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Not a raise
    }

    #[test]
    fn test_post_blinds_heads_up() {
        let mut players = vec![create_test_player(0, 1000), create_test_player(1, 1000)];
        let mut pot = PotManager::new();
        let mut round = BettingRound::new(100, BettingStructure::NoLimit);

        let (sb_amount, bb_amount) =
            BettingExecutor::post_blinds(&mut players, &mut pot, &mut round, 0, 50, 100);

        assert_eq!(sb_amount, 50);
        assert_eq!(bb_amount, 100);
        assert_eq!(players[0].current_bet, 50);
        assert_eq!(players[1].current_bet, 100);
    }
}
