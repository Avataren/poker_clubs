//! Bot decision-making strategies.
//!
//! Each strategy implements the `BotStrategy` trait to decide what action
//! to take given the current game state visible to the bot.

use crate::bot::evaluate::{estimate_hand_strength, preflop_hand_strength};
use crate::game::deck::Card;
use crate::game::hand::evaluate_hand;
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
    pub min_raise: i64,
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
            // Postflop: Monte Carlo with adaptive iteration counts
            let iterations = match view.phase {
                GamePhase::Flop => 160,
                GamePhase::Turn => 200,
                GamePhase::River => 240,
                _ => 180,
            };
            estimate_hand_strength(
                &view.hole_cards,
                &view.community_cards,
                view.num_active_opponents.max(1),
                iterations,
            )
        };

        let adjusted_strength = adjust_for_draws_and_board(strength, view).clamp(0.0, 1.0);

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
        let spr = if view.pot_total > 0 {
            view.my_stack as f64 / view.pot_total as f64
        } else {
            10.0
        };

        if can_check {
            // No bet to face
            if adjusted_strength > raise_threshold && !is_slowplaying {
                // Strong hand: raise (or occasional all-in with nuts)
                if adjusted_strength >= 0.95 && rng.gen_bool(0.15 + self.aggression * 0.15) {
                    // Very strong hand: sometimes go all-in
                    PlayerAction::AllIn
                } else if spr <= 2.0 && adjusted_strength > 0.65 {
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, adjusted_strength, &mut rng);
                    self.raise_action(view, raise_size)
                }
            } else if is_bluffing && view.phase != GamePhase::PreFlop {
                // Occasional bluff bet
                let raise_size = self.calculate_raise(view, 0.5, &mut rng);
                self.raise_action(view, raise_size)
            } else {
                PlayerAction::Check
            }
        } else {
            // Facing a bet — use pot odds to decide
            if adjusted_strength > raise_threshold && !is_slowplaying {
                // Strong hand: raise or re-raise
                if adjusted_strength >= 0.95 && rng.gen_bool(0.20 + self.aggression * 0.20) {
                    // Very strong hand (nuts): go all-in more often
                    PlayerAction::AllIn
                } else if spr <= 2.0 && adjusted_strength > 0.65 {
                    PlayerAction::AllIn
                } else if to_call >= view.my_stack / 2 {
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, adjusted_strength, &mut rng);
                    self.raise_action(view, raise_size)
                }
            } else if is_bluffing {
                // Bluff raise
                if to_call >= view.my_stack / 2 {
                    PlayerAction::AllIn
                } else {
                    let raise_size = self.calculate_raise(view, 0.5, &mut rng);
                    self.raise_action(view, raise_size)
                }
            } else if adjusted_strength >= min_call_strength {
                // Decent hand: call
                PlayerAction::Call
            } else if to_call <= view.big_blind && adjusted_strength > 0.25 {
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
        let to_call = (view.current_bet - view.my_current_bet).max(0);
        let pot = (view.pot_total + to_call).max(bb);
        let min_raise = view.min_raise.max(bb);
        let stack_after_call = (view.my_stack - to_call).max(0);

        // Preflop: use BB-based sizing (2.5-4x BB)
        if view.community_cards.is_empty() {
            let opponent_modifier = if view.num_active_opponents >= 4 {
                -0.2
            } else {
                0.1
            };
            let multiplier = 2.5 + strength * 1.5 + self.aggression * 0.5 + opponent_modifier;
            let raise = (bb as f64 * multiplier) as i64;
            return raise.max(min_raise).min(stack_after_call);
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

        // Ensure minimum bet is at least 1 BB, and don't exceed stack-after-call
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

    let has_flush_suit = hole_cards.iter().any(|card| {
        suit_counts.get(card.suit as usize).copied().unwrap_or(0) == 4
    });
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
    let has_pair = ranks.iter().copied().collect::<std::collections::HashSet<_>>().len()
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
            raise_count >= 6,
            "Aggressive bot should often raise AA, raised {}/20",
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
    fn test_simulated_tournament_battle() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

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
