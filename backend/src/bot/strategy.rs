//! Bot decision-making strategies.
//!
//! Each strategy implements the `BotStrategy` trait to decide what action
//! to take given the current game state visible to the bot.

use crate::bot::evaluate::{estimate_hand_strength, preflop_hand_strength, preflop_tier};
use crate::game::deck::Card;
use crate::game::hand::evaluate_hand;
use crate::game::{GamePhase, PlayerAction};
use rand::Rng;

/// Position categories for bot decision-making.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotPosition {
    Early,
    Middle,
    Late,
    Blind,
}

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
    pub min_raise: i64,
    pub num_active_opponents: usize,
    pub position: BotPosition,
    pub was_preflop_raiser: bool,
    pub is_big_blind: bool,
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

        // --- Preflop: chart + position-based decisions ---
        if view.community_cards.is_empty() && view.hole_cards.len() == 2 {
            return self.decide_preflop(view, &mut rng);
        }

        // --- Postflop ---
        let iterations = match view.phase {
            GamePhase::Flop => 160,
            GamePhase::Turn => 200,
            GamePhase::River => 240,
            _ => 180,
        };
        let strength = estimate_hand_strength(
            &view.hole_cards,
            &view.community_cards,
            view.num_active_opponents.max(1),
            iterations,
        );

        let adjusted_strength = adjust_for_draws_and_board(strength, view).clamp(0.0, 1.0);

        let raise_threshold = 0.70 - self.aggression * 0.20; // 0.50..0.70

        let to_call = view.current_bet - view.my_current_bet;
        let can_check = to_call <= 0;

        let pot_odds = if to_call > 0 && view.pot_total > 0 {
            to_call as f64 / (view.pot_total + to_call) as f64
        } else {
            0.0
        };

        let call_margin = 0.15 - self.aggression * 0.22;
        let min_call_strength = (pot_odds + call_margin).max(0.0);

        let bluff_roll: f64 = rng.gen();
        let is_bluffing = bluff_roll < 0.05 + self.aggression * 0.08;
        let is_slowplaying = bluff_roll > 0.95 - self.aggression * 0.05;
        let spr = if view.pot_total > 0 {
            view.my_stack as f64 / view.pot_total as f64
        } else {
            10.0
        };

        // Continuation bet: if we were the preflop raiser and it checks to us on the flop
        let should_cbet = view.was_preflop_raiser
            && view.phase == GamePhase::Flop
            && can_check
            && rng.gen_bool(0.65);

        if can_check {
            if adjusted_strength > raise_threshold && !is_slowplaying {
                if (adjusted_strength >= 0.95 && rng.gen_bool(0.15 + self.aggression * 0.15))
                    || (spr <= 2.0 && adjusted_strength > 0.65)
                {
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, adjusted_strength, &mut rng);
                    self.raise_action(view, raise_size)
                }
            } else if should_cbet {
                // C-bet with standard sizing regardless of hand strength
                let raise_size = self.calculate_raise(view, 0.55, &mut rng);
                self.raise_action(view, raise_size)
            } else if is_bluffing {
                let raise_size = self.calculate_raise(view, 0.5, &mut rng);
                self.raise_action(view, raise_size)
            } else {
                PlayerAction::Check
            }
        } else if adjusted_strength > raise_threshold && !is_slowplaying {
            if (adjusted_strength >= 0.95 && rng.gen_bool(0.20 + self.aggression * 0.20))
                || (spr <= 2.0 && adjusted_strength > 0.65)
                || to_call >= view.my_stack / 2
            {
                PlayerAction::AllIn
            } else {
                let raise_size = self.calculate_raise(view, adjusted_strength, &mut rng);
                self.raise_action(view, raise_size)
            }
        } else if is_bluffing {
            if to_call >= view.my_stack / 2 {
                PlayerAction::AllIn
            } else {
                let raise_size = self.calculate_raise(view, 0.5, &mut rng);
                self.raise_action(view, raise_size)
            }
        } else if adjusted_strength >= min_call_strength
            || (to_call <= view.big_blind && adjusted_strength > 0.25)
        {
            PlayerAction::Call
        } else {
            PlayerAction::Fold
        }
    }
}

impl SimpleStrategy {
    /// Position-aware preflop decision using chart tiers.
    fn decide_preflop(&self, view: &BotGameView, rng: &mut impl Rng) -> PlayerAction {
        let tier = preflop_tier(
            view.hole_cards[0].rank,
            view.hole_cards[1].rank,
            view.hole_cards[0].suit == view.hole_cards[1].suit,
        );

        // Aggression widens range by ~1 tier, tightness narrows by ~1 tier
        // aggression=0.2 -> adjust=+0.6 (tighter), aggression=0.8 -> adjust=-0.6 (looser)
        let tier_adjust = ((0.5 - self.aggression) * 2.0).round() as i8;
        let effective_tier = (tier as i8 + tier_adjust).max(0) as u8;

        // Max playable tier depends on position
        let max_tier = match view.position {
            BotPosition::Early => 2,  // tight: only premium/strong
            BotPosition::Middle => 4, // standard range
            BotPosition::Late => 6,   // wide range
            BotPosition::Blind => {
                // BB defends wider vs small raises, SB plays tighter
                let to_call = (view.current_bet - view.my_current_bet).max(0);
                if to_call <= view.big_blind * 2 {
                    5 // BB defend range
                } else {
                    3 // SB or facing big raise
                }
            }
        };

        let to_call = (view.current_bet - view.my_current_bet).max(0);
        let can_check = to_call == 0;
        let bb = view.big_blind.max(1);

        // Big blind preflop option: if nobody raised, never fold.
        // Normal flow checks (to_call == 0); fallback for inconsistent blind
        // tracking is to call the blind instead of folding.
        if view.is_big_blind && view.current_bet <= bb {
            return if can_check {
                PlayerAction::Check
            } else if to_call >= view.my_stack {
                PlayerAction::AllIn
            } else {
                PlayerAction::Call
            };
        }

        // Fold if hand is outside our playable range
        if effective_tier > max_tier {
            return if can_check {
                PlayerAction::Check
            } else {
                PlayerAction::Fold
            };
        }

        // Steal attempt: in late position, if folded to us, open wider
        let folded_to_us = can_check || (to_call <= bb && view.position == BotPosition::Late);
        let is_steal = view.position == BotPosition::Late && folded_to_us && effective_tier <= 7;

        // Determine if we're facing a raise (someone raised above BB)
        let facing_raise = to_call > bb;
        let facing_3bet = to_call > bb * 5; // rough heuristic: >5BB means 3-bet

        if can_check || (to_call <= bb && !facing_raise) {
            // No raise to face — open raise or check
            if effective_tier <= 1 {
                // Premium: always raise
                let raise_size = self.calculate_preflop_raise(view, false, rng);
                self.raise_action(view, raise_size)
            } else if effective_tier <= max_tier || is_steal {
                // Playable hand: raise
                let raise_size = self.calculate_preflop_raise(view, false, rng);
                self.raise_action(view, raise_size)
            } else {
                PlayerAction::Check
            }
        } else if facing_3bet {
            // Facing a 3-bet: only continue with strong hands
            if effective_tier <= 1 {
                // 4-bet with premium
                let raise_size = self.calculate_preflop_raise(view, true, rng);
                self.raise_action(view, raise_size)
            } else if effective_tier <= 3 {
                // Call with good hands
                PlayerAction::Call
            } else {
                PlayerAction::Fold
            }
        } else if facing_raise {
            // Facing a single raise
            if effective_tier <= 1 {
                // 3-bet with premium
                let raise_size = self.calculate_preflop_raise(view, true, rng);
                self.raise_action(view, raise_size)
            } else if effective_tier <= 2 && rng.gen_bool(0.4 + self.aggression * 0.2) {
                // Sometimes 3-bet with strong hands
                let raise_size = self.calculate_preflop_raise(view, true, rng);
                self.raise_action(view, raise_size)
            } else if effective_tier <= max_tier {
                // Call with playable hands
                PlayerAction::Call
            } else if to_call <= bb * 2 && effective_tier <= max_tier + 1 {
                // Cheap call with marginal hand
                PlayerAction::Call
            } else {
                PlayerAction::Fold
            }
        } else {
            // Default: use strength-based logic
            let strength = preflop_hand_strength(&view.hole_cards);
            if strength > 0.60 {
                PlayerAction::Call
            } else {
                PlayerAction::Fold
            }
        }
    }

    /// Calculate preflop raise size with standard poker sizing.
    fn calculate_preflop_raise(
        &self,
        view: &BotGameView,
        is_reraise: bool,
        rng: &mut impl Rng,
    ) -> i64 {
        let bb = view.big_blind.max(1);
        let to_call = (view.current_bet - view.my_current_bet).max(0);
        let min_raise = view.min_raise.max(bb);
        let stack_after_call = (view.my_stack - to_call).max(0);

        let raise = if is_reraise {
            if view.current_bet > bb * 8 {
                // Facing a 3-bet, 4-bet: 2.2-2.5x the current bet
                let multiplier = rng.gen_range(2.2..2.5);
                let target = (view.current_bet as f64 * multiplier) as i64;
                (target - view.current_bet).max(min_raise)
            } else {
                // 3-bet: 3x-3.5x the raise
                let multiplier =
                    if view.position == BotPosition::Blind || view.position == BotPosition::Early {
                        rng.gen_range(3.2..3.6) // OOP: bigger
                    } else {
                        rng.gen_range(2.8..3.2) // IP: smaller
                    };
                let target = (view.current_bet as f64 * multiplier) as i64;
                (target - view.current_bet).max(min_raise)
            }
        } else {
            // Open raise: 2.5-3x BB + 1BB per limper
            let base_multiplier = rng.gen_range(2.5..3.0);
            // Count limpers: excess in pot beyond blinds
            let expected_blind_pot = bb + bb / 2; // BB + SB
            let limper_money = (view.pot_total - expected_blind_pot).max(0);
            let limper_extra = (limper_money / bb).min(4); // cap at 4 extra BB
            let target = (bb as f64 * base_multiplier) as i64 + limper_extra * bb;
            (target - view.current_bet).max(min_raise)
        };

        // Add ±10% variance
        let variance: f64 = rng.gen_range(0.9..1.1);
        let raise = (raise as f64 * variance) as i64;
        raise.max(min_raise).min(stack_after_call)
    }

    /// Calculate postflop raise with street-dependent sizing.
    fn calculate_raise(&self, view: &BotGameView, strength: f64, rng: &mut impl Rng) -> i64 {
        let bb = view.big_blind.max(1);
        let to_call = (view.current_bet - view.my_current_bet).max(0);
        let pot = (view.pot_total + to_call).max(bb);
        let min_raise = view.min_raise.max(bb);
        let stack_after_call = (view.my_stack - to_call).max(0);

        // Preflop: delegate to preflop raise calculator
        if view.community_cards.is_empty() {
            return self.calculate_preflop_raise(view, view.current_bet > bb, rng);
        }

        // Street-dependent base sizing
        let street_base = match view.phase {
            GamePhase::Flop => rng.gen_range(0.50..0.66),
            GamePhase::Turn => rng.gen_range(0.66..0.75),
            GamePhase::River => rng.gen_range(0.75..1.00),
            _ => 0.60,
        };

        let base_percentage = if strength >= 0.85 {
            // Very strong: use street sizing, occasionally overbet on river
            if view.phase == GamePhase::River && rng.gen_bool(0.25) {
                rng.gen_range(1.2..1.5) // overbet for value
            } else {
                street_base
            }
        } else if strength >= 0.70 {
            // Strong: standard street sizing
            street_base
        } else if strength >= 0.55 {
            // Medium: slightly smaller than street standard
            street_base * 0.8
        } else {
            // Bluff/weak: use same sizing as value bets (balanced)
            street_base * 0.9
        };

        // Add small variance (±10%)
        let variance: f64 = rng.gen_range(0.9..1.1);
        let raise = (pot as f64 * base_percentage * variance) as i64;

        let min_bet = min_raise.max(bb);
        raise.max(min_bet).min(stack_after_call)
    }

    fn raise_action(&self, view: &BotGameView, raise_size: i64) -> PlayerAction {
        let to_call = (view.current_bet - view.my_current_bet).max(0);
        let stack_after_call = (view.my_stack - to_call).max(0);
        if raise_size < view.min_raise || raise_size >= stack_after_call {
            PlayerAction::AllIn
        } else {
            PlayerAction::Raise(raise_size)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StraightDraw {
    None,
    Gutshot,
    OpenEnded,
}

#[derive(Debug, Clone, Copy)]
struct DrawInfo {
    flush_draw: bool,
    straight_draw: StraightDraw,
}

fn adjust_for_draws_and_board(base_strength: f64, view: &BotGameView) -> f64 {
    if view.community_cards.is_empty() {
        return base_strength;
    }

    let draw_info = detect_draws(&view.hole_cards, &view.community_cards);
    let board_info = board_texture(&view.community_cards);
    let made_rank = evaluate_hand(&view.hole_cards, &view.community_cards).rank_value;

    let mut strength = base_strength;
    if draw_info.flush_draw {
        strength += 0.08;
    }
    strength += match draw_info.straight_draw {
        StraightDraw::OpenEnded => 0.06,
        StraightDraw::Gutshot => 0.03,
        StraightDraw::None => 0.0,
    };

    let opponent_pressure = (view.num_active_opponents as f64).min(4.0) * 0.02;
    if board_info.is_wet && made_rank <= 1 {
        strength -= opponent_pressure;
    } else if made_rank >= 2 && !board_info.is_wet {
        strength += 0.02;
    }

    strength
}

fn detect_draws(hole_cards: &[Card], community_cards: &[Card]) -> DrawInfo {
    let mut suit_counts = [0u8; 4];
    let mut ranks: Vec<u8> = Vec::new();
    for card in hole_cards.iter().chain(community_cards.iter()) {
        if let Some(count) = suit_counts.get_mut(card.suit as usize) {
            *count += 1;
        }
        ranks.push(card.rank);
    }

    let has_flush_suit = hole_cards
        .iter()
        .any(|card| suit_counts.get(card.suit as usize).copied().unwrap_or(0) == 4);
    let flush_draw = community_cards.len() < 5 && has_flush_suit;

    let straight_draw = if community_cards.len() >= 3 && community_cards.len() < 5 {
        straight_draw_from_cards(hole_cards, &ranks)
    } else {
        StraightDraw::None
    };

    DrawInfo {
        flush_draw,
        straight_draw,
    }
}

fn straight_draw_from_cards(hole_cards: &[Card], ranks: &[u8]) -> StraightDraw {
    let mut unique: Vec<u8> = ranks.iter().copied().collect();
    if unique.contains(&14) {
        unique.push(1);
    }
    unique.sort_unstable();
    unique.dedup();

    let mut hole_ranks: Vec<u8> = hole_cards.iter().map(|card| card.rank).collect();
    if hole_ranks.contains(&14) {
        hole_ranks.push(1);
    }

    let rank_set: std::collections::HashSet<u8> = unique.iter().copied().collect();
    let mut best = StraightDraw::None;

    for start in 1..=10 {
        let window: Vec<u8> = (start..=start + 4).collect();
        let count = window.iter().filter(|r| rank_set.contains(r)).count();
        if count == 5 {
            return StraightDraw::None;
        }
        if count == 4 && hole_ranks.iter().any(|r| window.contains(r)) {
            let missing_low = !rank_set.contains(&start);
            let missing_high = !rank_set.contains(&(start + 4));
            if missing_low || missing_high {
                best = StraightDraw::OpenEnded;
            } else if best == StraightDraw::None {
                best = StraightDraw::Gutshot;
            }
        }
    }

    best
}

struct BoardTexture {
    is_wet: bool,
}

fn board_texture(community_cards: &[Card]) -> BoardTexture {
    if community_cards.is_empty() {
        return BoardTexture { is_wet: false };
    }

    let mut suit_counts = [0u8; 4];
    let mut ranks: Vec<u8> = Vec::new();
    for card in community_cards {
        if let Some(count) = suit_counts.get_mut(card.suit as usize) {
            *count += 1;
        }
        ranks.push(card.rank);
    }

    let has_flushy = suit_counts.iter().any(|&count| count >= 3);
    let has_pair = ranks
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        < ranks.len();

    ranks.sort_unstable();
    ranks.dedup();
    let mut has_straighty = false;
    for window in ranks.windows(3) {
        if window.len() == 3 && window[2] - window[0] <= 4 {
            has_straighty = true;
            break;
        }
    }

    BoardTexture {
        is_wet: has_flushy || has_pair || has_straighty,
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
    use rand::seq::SliceRandom;

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
            min_raise: 50,
            num_active_opponents: 1,
            position: BotPosition::Middle,
            was_preflop_raiser: false,
            is_big_blind: false,
        }
    }

    fn make_preflop_view(
        hole: Vec<Card>,
        pot: i64,
        current_bet: i64,
        my_bet: i64,
        my_stack: i64,
        position: BotPosition,
    ) -> BotGameView {
        BotGameView {
            hole_cards: hole,
            community_cards: vec![],
            pot_total: pot,
            current_bet,
            my_current_bet: my_bet,
            my_stack,
            phase: GamePhase::PreFlop,
            big_blind: 50,
            min_raise: 50,
            num_active_opponents: 3,
            position,
            was_preflop_raiser: false,
            is_big_blind: false,
        }
    }

    #[test]
    fn test_simple_strategy_checks_with_weak_hand_no_bet() {
        let strategy = SimpleStrategy::tight();
        // Weak hand, no bet to face — postflop
        let view = make_view(
            vec![Card::new(2, 0), Card::new(7, 3)],
            vec![Card::new(3, 1), Card::new(9, 2), Card::new(13, 0)],
            100,
            0,
            0,
            1000,
            GamePhase::Flop,
        );
        let action = strategy.decide(&view);
        assert!(
            !matches!(action, PlayerAction::Fold),
            "Should not fold when can check"
        );
    }

    #[test]
    fn test_simple_strategy_folds_junk_facing_raise() {
        let strategy = SimpleStrategy::tight();
        // Very weak hand (2-3o) on A-K-Q board — postflop
        let view = make_view(
            vec![Card::new(2, 0), Card::new(3, 3)],
            vec![Card::new(14, 1), Card::new(13, 2), Card::new(12, 0)],
            500,
            400,
            0,
            1000,
            GamePhase::Flop,
        );
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
        // Pocket aces preflop, late position, no bet
        let view = make_preflop_view(
            vec![Card::new(14, 0), Card::new(14, 1)],
            75,
            0,
            0,
            1000,
            BotPosition::Late,
        );
        let mut raise_count = 0;
        for _ in 0..20 {
            let action = strategy.decide(&view);
            if matches!(action, PlayerAction::Raise(_) | PlayerAction::AllIn) {
                raise_count += 1;
            }
        }
        assert!(
            raise_count >= 15,
            "Aggressive bot should always raise AA, raised {}/20",
            raise_count
        );
    }

    #[test]
    fn test_draw_adjustment_prefers_calling_flush_draw() {
        let strategy = SimpleStrategy::balanced();
        let flush_draw_view = BotGameView {
            hole_cards: vec![Card::new(14, 2), Card::new(9, 2)],
            community_cards: vec![Card::new(2, 2), Card::new(7, 2), Card::new(13, 1)],
            pot_total: 200,
            current_bet: 300,
            my_current_bet: 0,
            my_stack: 1200,
            phase: GamePhase::Flop,
            big_blind: 50,
            min_raise: 50,
            num_active_opponents: 1,
            position: BotPosition::Middle,
            was_preflop_raiser: false,
            is_big_blind: false,
        };

        let no_draw_view = BotGameView {
            hole_cards: vec![Card::new(14, 3), Card::new(9, 1)],
            community_cards: vec![Card::new(2, 2), Card::new(7, 2), Card::new(13, 1)],
            pot_total: 200,
            current_bet: 300,
            my_current_bet: 0,
            my_stack: 1200,
            phase: GamePhase::Flop,
            big_blind: 50,
            min_raise: 50,
            num_active_opponents: 1,
            position: BotPosition::Middle,
            was_preflop_raiser: false,
            is_big_blind: false,
        };

        let mut draw_folds = 0;
        let mut no_draw_folds = 0;
        for _ in 0..40 {
            if matches!(strategy.decide(&flush_draw_view), PlayerAction::Fold) {
                draw_folds += 1;
            }
            if matches!(strategy.decide(&no_draw_view), PlayerAction::Fold) {
                no_draw_folds += 1;
            }
        }

        assert!(
            draw_folds < no_draw_folds,
            "Flush draws should reduce folding (draw folds {} vs no-draw folds {})",
            draw_folds,
            no_draw_folds
        );
    }

    #[test]
    fn test_min_raise_respected() {
        let strategy = SimpleStrategy::aggressive();
        let view = BotGameView {
            hole_cards: vec![Card::new(14, 0), Card::new(14, 1)],
            community_cards: vec![],
            pot_total: 150,
            current_bet: 100,
            my_current_bet: 0,
            my_stack: 2000,
            phase: GamePhase::PreFlop,
            big_blind: 50,
            min_raise: 150,
            num_active_opponents: 2,
            position: BotPosition::Late,
            was_preflop_raiser: false,
            is_big_blind: false,
        };

        let mut saw_raise = false;
        for _ in 0..20 {
            if let PlayerAction::Raise(amount) = strategy.decide(&view) {
                saw_raise = true;
                assert!(
                    amount >= view.min_raise,
                    "Raise amount {} should respect min raise {}",
                    amount,
                    view.min_raise
                );
            }
        }
        assert!(saw_raise, "Expected to see at least one raise with AA");
    }

    #[test]
    fn test_preflop_chart_tiers() {
        use crate::bot::evaluate::preflop_tier;

        // AA is tier 0 (premium)
        assert_eq!(preflop_tier(14, 14, false), 0);
        // KK is tier 0
        assert_eq!(preflop_tier(13, 13, false), 0);
        // AKs is tier 0
        assert_eq!(preflop_tier(14, 13, true), 0);
        // AKo is tier 1
        assert_eq!(preflop_tier(14, 13, false), 1);
        // JJ is tier 1
        assert_eq!(preflop_tier(11, 11, false), 1);
        // 72o is tier 8 (trash)
        assert_eq!(preflop_tier(7, 2, false), 8);
        // 32o is tier 8 (trash)
        assert_eq!(preflop_tier(3, 2, false), 8);
        // Tier ordering: AA <= AKs < JJ < 99 < 72o
        assert!(preflop_tier(14, 14, false) <= preflop_tier(14, 13, true));
        assert!(preflop_tier(14, 13, true) <= preflop_tier(11, 11, false));
        assert!(preflop_tier(11, 11, false) < preflop_tier(9, 9, false));
        assert!(preflop_tier(9, 9, false) < preflop_tier(7, 2, false));

        // Suited > offsuit for same ranks
        assert!(preflop_tier(14, 12, true) < preflop_tier(14, 12, false));
        assert!(preflop_tier(13, 11, true) < preflop_tier(13, 11, false));
    }

    #[test]
    fn test_position_affects_decisions() {
        let strategy = SimpleStrategy::balanced();
        // K9o is tier 6 — playable in Late but not in Early
        let k9o = vec![Card::new(13, 0), Card::new(9, 1)];

        let early_view = make_preflop_view(k9o.clone(), 75, 0, 0, 1000, BotPosition::Early);

        let late_view = make_preflop_view(k9o, 75, 0, 0, 1000, BotPosition::Late);

        let mut early_folds = 0;
        let mut late_folds = 0;
        for _ in 0..30 {
            if matches!(strategy.decide(&early_view), PlayerAction::Check) {
                early_folds += 1;
            }
            if matches!(strategy.decide(&late_view), PlayerAction::Check) {
                late_folds += 1;
            }
        }
        // In early position, K9o (tier 6) should be checked/folded more than in late
        assert!(
            early_folds > late_folds,
            "Should play tighter in early position (early checks: {}, late checks: {})",
            early_folds,
            late_folds
        );
    }

    #[test]
    fn test_big_blind_never_folds_unraised_preflop() {
        let strategy = SimpleStrategy::tight();
        let view = BotGameView {
            hole_cards: vec![Card::new(7, 0), Card::new(2, 1)],
            community_cards: vec![],
            pot_total: 75,
            current_bet: 50,
            my_current_bet: 50,
            my_stack: 1000,
            phase: GamePhase::PreFlop,
            big_blind: 50,
            min_raise: 50,
            num_active_opponents: 3,
            position: BotPosition::Blind,
            was_preflop_raiser: false,
            is_big_blind: true,
        };

        for _ in 0..20 {
            assert!(matches!(strategy.decide(&view), PlayerAction::Check));
        }
    }

    #[test]
    fn test_big_blind_unraised_fallback_calls_instead_of_folding() {
        let strategy = SimpleStrategy::tight();
        let view = BotGameView {
            hole_cards: vec![Card::new(7, 0), Card::new(2, 1)],
            community_cards: vec![],
            pot_total: 75,
            current_bet: 50,
            my_current_bet: 0,
            my_stack: 1000,
            phase: GamePhase::PreFlop,
            big_blind: 50,
            min_raise: 50,
            num_active_opponents: 3,
            position: BotPosition::Blind,
            was_preflop_raiser: false,
            is_big_blind: true,
        };

        for _ in 0..20 {
            assert!(matches!(strategy.decide(&view), PlayerAction::Call));
        }
    }

    #[test]
    fn test_cbet_on_flop() {
        let strategy = SimpleStrategy::balanced();
        // Bot was preflop raiser, weak hand on flop, it checks to them
        let view = BotGameView {
            hole_cards: vec![Card::new(14, 0), Card::new(9, 1)],
            community_cards: vec![Card::new(3, 2), Card::new(7, 3), Card::new(5, 0)],
            pot_total: 300,
            current_bet: 0,
            my_current_bet: 0,
            my_stack: 1500,
            phase: GamePhase::Flop,
            big_blind: 50,
            min_raise: 50,
            num_active_opponents: 1,
            position: BotPosition::Late,
            was_preflop_raiser: true,
            is_big_blind: false,
        };

        let mut bet_count = 0;
        for _ in 0..40 {
            let action = strategy.decide(&view);
            if matches!(action, PlayerAction::Raise(_) | PlayerAction::AllIn) {
                bet_count += 1;
            }
        }
        // Should c-bet a significant portion of the time (~65% + bluff %)
        assert!(
            bet_count >= 15,
            "Preflop raiser should c-bet frequently on flop, bet {}/40",
            bet_count
        );
    }

    #[test]
    fn test_simulated_tournament_battle() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let positions = [BotPosition::Early, BotPosition::Middle, BotPosition::Late];
        let strategies: Vec<Box<dyn BotStrategy>> = vec![
            Box::new(SimpleStrategy::tight()),
            Box::new(SimpleStrategy::balanced()),
            Box::new(CallingStation),
        ];

        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let mut wins = vec![0usize; strategies.len()];

        for _ in 0..60 {
            let mut deck: Vec<Card> = (0u8..4)
                .flat_map(|suit| (2u8..=14).map(move |rank| Card::new(rank, suit)))
                .collect();
            deck.shuffle(&mut rng);

            let community = deck.drain(0..5).collect::<Vec<_>>();
            let mut hands = Vec::new();
            for _ in 0..strategies.len() {
                hands.push(deck.drain(0..2).collect::<Vec<_>>());
            }

            let mut active_players = Vec::new();
            for (idx, strategy) in strategies.iter().enumerate() {
                let view = BotGameView {
                    hole_cards: hands[idx].clone(),
                    community_cards: vec![],
                    pot_total: 150,
                    current_bet: 100,
                    my_current_bet: 0,
                    my_stack: 1500,
                    phase: GamePhase::PreFlop,
                    big_blind: 50,
                    min_raise: 100,
                    num_active_opponents: strategies.len() - 1,
                    position: positions[idx % positions.len()],
                    was_preflop_raiser: false,
                    is_big_blind: false,
                };

                let action = strategy.decide(&view);
                if !matches!(action, PlayerAction::Fold) {
                    active_players.push(idx);
                }
            }

            if active_players.is_empty() {
                continue;
            }

            let mut best_idx = active_players[0];
            let mut best_rank = evaluate_hand(&hands[best_idx], &community);
            for idx in active_players.iter().skip(1).copied() {
                let rank = evaluate_hand(&hands[idx], &community);
                if rank > best_rank {
                    best_rank = rank;
                    best_idx = idx;
                }
            }

            wins[best_idx] += 1;
        }

        assert!(
            wins.iter().all(|&count| count > 0),
            "Each strategy should win at least one simulated hand: {:?}",
            wins
        );
    }
}
