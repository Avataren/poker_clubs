//! Bot decision-making strategies.
//!
//! Each strategy implements the `BotStrategy` trait to decide what action
//! to take given the current game state visible to the bot.

use crate::bot::evaluate::{estimate_hand_strength, preflop_hand_strength};
use crate::game::deck::Card;
use crate::game::{GamePhase, PlayerAction};
use rand::Rng;

/// Everything the bot can see about the current game state.
pub struct BotGameView {
    pub hole_cards: Vec<Card>,
    pub community_cards: Vec<Card>,
    pub pot_total: i64,
    pub current_bet: i64,
    pub my_current_bet: i64,
    pub my_stack: i64,
    pub phase: GamePhase,
    pub big_blind: i64,
    pub num_active_opponents: usize,
}

/// Trait for bot decision-making.
pub trait BotStrategy: Send + Sync {
    fn decide(&self, view: &BotGameView) -> PlayerAction;
    fn name(&self) -> &str;
}

/// A simple strategy that uses hand strength and pot odds.
///
/// `aggression` controls how loose/aggressive the bot plays:
/// - 0.0 = very tight/passive (folds most hands, rarely raises)
/// - 0.5 = balanced
/// - 1.0 = very loose/aggressive (plays many hands, raises often)
pub struct SimpleStrategy {
    pub aggression: f64,
}

impl SimpleStrategy {
    pub fn new(aggression: f64) -> Self {
        Self {
            aggression: aggression.clamp(0.0, 1.0),
        }
    }

    pub fn tight() -> Self {
        Self::new(0.2)
    }
    pub fn balanced() -> Self {
        Self::new(0.5)
    }
    pub fn aggressive() -> Self {
        Self::new(0.8)
    }
}

impl BotStrategy for SimpleStrategy {
    fn name(&self) -> &str {
        "simple"
    }

    fn decide(&self, view: &BotGameView) -> PlayerAction {
        let mut rng = rand::thread_rng();

        let strength = if view.community_cards.is_empty() {
            // Preflop: use quick heuristic
            preflop_hand_strength(&view.hole_cards)
        } else {
            // Postflop: Monte Carlo (fast, 200 iterations)
            estimate_hand_strength(
                &view.hole_cards,
                &view.community_cards,
                view.num_active_opponents.max(1),
                200,
            )
        };

        // Aggression adjusts thresholds:
        // Higher aggression = plays more hands and raises more
        let raise_threshold = 0.70 - self.aggression * 0.20; // 0.50..0.70

        let to_call = view.current_bet - view.my_current_bet;
        let can_check = to_call <= 0;

        // Pot odds: ratio of call amount to total pot
        let pot_odds = if to_call > 0 && view.pot_total > 0 {
            to_call as f64 / (view.pot_total + to_call) as f64
        } else {
            0.0
        };

        // Required strength to call: pot_odds + a margin based on tightness
        // Tight (0.2): need strength > pot_odds + 0.12
        // Aggressive (0.8): need strength > pot_odds - 0.06
        let call_margin = 0.15 - self.aggression * 0.22; // -0.07..0.13
        let min_call_strength = (pot_odds + call_margin).max(0.0);

        // Add some randomness (bluff or slowplay ~10% of the time)
        let bluff_roll: f64 = rng.gen();
        let is_bluffing = bluff_roll < 0.05 + self.aggression * 0.08;
        let is_slowplaying = bluff_roll > 0.95 - self.aggression * 0.05;

        if can_check {
            // No bet to face
            if strength > raise_threshold && !is_slowplaying {
                // Strong hand: raise (or occasional all-in with nuts)
                if strength >= 0.95 && rng.gen_bool(0.15 + self.aggression * 0.15) {
                    // Very strong hand: sometimes go all-in
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, strength, &mut rng);
                    PlayerAction::Raise(raise_size)
                }
            } else if is_bluffing && view.phase != GamePhase::PreFlop {
                // Occasional bluff bet
                let raise_size = self.calculate_raise(view, 0.5, &mut rng);
                PlayerAction::Raise(raise_size)
            } else {
                PlayerAction::Check
            }
        } else {
            // Facing a bet — use pot odds to decide
            if strength > raise_threshold && !is_slowplaying {
                // Strong hand: raise or re-raise
                if strength >= 0.95 && rng.gen_bool(0.20 + self.aggression * 0.20) {
                    // Very strong hand (nuts): go all-in more often
                    PlayerAction::AllIn
                } else if to_call >= view.my_stack / 2 {
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, strength, &mut rng);
                    PlayerAction::Raise(raise_size)
                }
            } else if is_bluffing {
                // Bluff raise
                if to_call >= view.my_stack / 2 {
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, 0.5, &mut rng);
                    PlayerAction::Raise(raise_size)
                }
            } else if strength >= min_call_strength {
                // Decent hand: call
                PlayerAction::Call
            } else if to_call <= view.big_blind && strength > 0.25 {
                // Cheap call: worth it with marginal hand
                PlayerAction::Call
            } else {
                PlayerAction::Fold
            }
        }
    }
}

impl SimpleStrategy {
    fn calculate_raise(&self, view: &BotGameView, strength: f64, rng: &mut impl Rng) -> i64 {
        let bb = view.big_blind.max(1);
        let pot = view.pot_total.max(bb);

        // Preflop: use BB-based sizing (2.5-4x BB)
        if view.community_cards.is_empty() {
            let multiplier = 2.5 + strength * 1.5 + self.aggression * 0.5;
            let raise = (bb as f64 * multiplier) as i64;
            return raise.max(bb * 2).min(view.my_stack);
        }

        // Postflop: use pot-based sizing like real players
        // Choose bet size based on hand strength and aggression
        let base_percentage = if strength >= 0.85 {
            // Very strong hands: vary between 50% pot to overbet
            if rng.gen_bool(0.3) {
                1.0 + self.aggression * 0.5 // pot or overbet (for value)
            } else if rng.gen_bool(0.5) {
                0.75 // 75% pot
            } else {
                0.5 // 50% pot
            }
        } else if strength >= 0.70 {
            // Strong hands: 50-75% pot
            if rng.gen_bool(0.6) {
                0.75
            } else {
                0.5
            }
        } else if strength >= 0.55 {
            // Medium hands: 33-50% pot
            if rng.gen_bool(0.5) {
                0.5
            } else {
                0.33
            }
        } else {
            // Bluff or weak: 33-50% pot (smaller for bluffs)
            if rng.gen_bool(0.7) {
                0.33
            } else {
                0.5
            }
        };

        // Add small variance (0.9x to 1.15x)
        let variance: f64 = rng.gen_range(0.9..1.15);
        let raise = (pot as f64 * base_percentage * variance) as i64;

        // Ensure minimum bet is at least 1 BB, and don't exceed stack
        let min_bet = bb.max(view.current_bet + bb);
        raise.max(min_bet).min(view.my_stack)
    }
}

/// A calling station: calls everything, never folds, never raises.
/// Useful for testing.
pub struct CallingStation;

impl BotStrategy for CallingStation {
    fn name(&self) -> &str {
        "calling_station"
    }

    fn decide(&self, view: &BotGameView) -> PlayerAction {
        let to_call = view.current_bet - view.my_current_bet;
        if to_call <= 0 {
            PlayerAction::Check
        } else if to_call >= view.my_stack {
            PlayerAction::AllIn
        } else {
            PlayerAction::Call
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_view(
        hole: Vec<Card>,
        community: Vec<Card>,
        pot: i64,
        current_bet: i64,
        my_bet: i64,
        my_stack: i64,
        phase: GamePhase,
    ) -> BotGameView {
        BotGameView {
            hole_cards: hole,
            community_cards: community,
            pot_total: pot,
            current_bet,
            my_current_bet: my_bet,
            my_stack,
            phase,
            big_blind: 50,
            num_active_opponents: 1,
        }
    }

    #[test]
    fn test_simple_strategy_checks_with_weak_hand_no_bet() {
        let strategy = SimpleStrategy::tight();
        // Weak hand, no bet to face
        let view = make_view(
            vec![Card::new(2, 0), Card::new(7, 3)],
            vec![Card::new(3, 1), Card::new(9, 2), Card::new(13, 0)],
            100,
            0,
            0,
            1000,
            GamePhase::Flop,
        );
        // With a weak hand and no bet, a tight player should check
        let action = strategy.decide(&view);
        // Could be Check or a rare bluff — just verify it's not Fold
        assert!(
            !matches!(action, PlayerAction::Fold),
            "Should not fold when can check"
        );
    }

    #[test]
    fn test_simple_strategy_folds_junk_facing_raise() {
        let strategy = SimpleStrategy::tight();
        // Very weak hand (2-3o) on A-K-Q board — virtually no chance to win
        let view = make_view(
            vec![Card::new(2, 0), Card::new(3, 3)],
            vec![Card::new(14, 1), Card::new(13, 2), Card::new(12, 0)],
            500,
            400,
            0,
            1000,
            GamePhase::Flop,
        );
        // Run multiple times — tight strategy should usually fold junk vs big raise
        let mut fold_count = 0;
        for _ in 0..20 {
            if matches!(strategy.decide(&view), PlayerAction::Fold) {
                fold_count += 1;
            }
        }
        assert!(
            fold_count > 10,
            "Tight bot should fold junk vs big raise most of the time, folded {}/20",
            fold_count
        );
    }

    #[test]
    fn test_calling_station_never_folds() {
        let strategy = CallingStation;
        let view = make_view(
            vec![Card::new(2, 0), Card::new(7, 3)],
            vec![],
            500,
            400,
            0,
            1000,
            GamePhase::PreFlop,
        );
        for _ in 0..20 {
            let action = strategy.decide(&view);
            assert!(
                !matches!(action, PlayerAction::Fold),
                "Calling station should never fold"
            );
        }
    }

    #[test]
    fn test_simple_strategy_raises_strong_hand() {
        let strategy = SimpleStrategy::aggressive();
        // Pocket aces preflop, no bet
        let view = make_view(
            vec![Card::new(14, 0), Card::new(14, 1)],
            vec![],
            75,
            0,
            0,
            1000,
            GamePhase::PreFlop,
        );
        let mut raise_count = 0;
        for _ in 0..20 {
            if matches!(strategy.decide(&view), PlayerAction::Raise(_)) {
                raise_count += 1;
            }
        }
        assert!(
            raise_count > 10,
            "Aggressive bot should usually raise AA, raised {}/20",
            raise_count
        );
    }
}
