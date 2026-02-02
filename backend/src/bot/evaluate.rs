//! Hand strength evaluation helpers for bot decision-making.
//!
//! Provides Monte Carlo simulation to estimate win probability
//! and quick preflop hand ranking.

use crate::game::deck::Card;
use crate::game::hand::evaluate_hand;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Build a full 52-card deck, excluding the given cards.
fn remaining_deck(exclude: &[Card]) -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for suit in 0..4u8 {
        for rank in 2..=14u8 {
            let card = Card::new(rank, suit);
            if !exclude
                .iter()
                .any(|c| c.rank == card.rank && c.suit == card.suit)
            {
                deck.push(card);
            }
        }
    }
    deck
}

/// Estimate hand strength as a value in 0.0..=1.0 using Monte Carlo simulation.
///
/// Deals random opponent hands and remaining community cards, then counts
/// how often our hand wins or ties. Higher values mean stronger hands.
pub fn estimate_hand_strength(
    hole_cards: &[Card],
    community_cards: &[Card],
    num_opponents: usize,
    iterations: usize,
) -> f64 {
    if hole_cards.is_empty() || num_opponents == 0 {
        return 0.5;
    }

    let mut known: Vec<Card> = Vec::new();
    known.extend_from_slice(hole_cards);
    known.extend_from_slice(community_cards);

    let deck = remaining_deck(&known);
    let community_remaining = 5 - community_cards.len();
    // Each opponent needs 2 cards, plus we need to fill community
    let cards_needed = community_remaining + num_opponents * 2;

    if deck.len() < cards_needed {
        return 0.5;
    }

    let mut rng = ChaCha20Rng::from_entropy();
    let mut wins = 0.0;

    for _ in 0..iterations {
        let mut shuffled = deck.clone();
        shuffled.shuffle(&mut rng);

        // Deal remaining community cards
        let mut full_community: Vec<Card> = community_cards.to_vec();
        for card in shuffled.iter().take(community_remaining) {
            full_community.push(*card);
        }

        // Evaluate our hand
        let our_rank = evaluate_hand(hole_cards, &full_community);

        // Deal and evaluate opponent hands
        let mut we_win = true;
        let mut tie_count = 0;
        for opp in 0..num_opponents {
            let opp_start = community_remaining + opp * 2;
            let opp_cards = &shuffled[opp_start..opp_start + 2];
            let opp_rank = evaluate_hand(opp_cards, &full_community);

            if opp_rank > our_rank {
                we_win = false;
                break;
            } else if opp_rank == our_rank {
                tie_count += 1;
            }
        }

        if we_win {
            if tie_count > 0 {
                wins += 1.0 / (tie_count + 1) as f64;
            } else {
                wins += 1.0;
            }
        }
    }

    wins / iterations as f64
}

/// Quick preflop hand strength estimate (0.0..=1.0) without Monte Carlo.
///
/// Based on starting hand categories: pairs, suited connectors, high cards, etc.
pub fn preflop_hand_strength(hole_cards: &[Card]) -> f64 {
    if hole_cards.len() < 2 {
        return 0.0;
    }

    let r1 = hole_cards[0].rank;
    let r2 = hole_cards[1].rank;
    let high = r1.max(r2);
    let low = r1.min(r2);
    let suited = hole_cards[0].suit == hole_cards[1].suit;
    let pair = r1 == r2;
    let gap = high - low;
    let connected = gap == 1;

    if pair {
        // Pairs: 22=0.50, ..., AA=0.95
        return 0.50 + (low as f64 - 2.0) * 0.0375;
    }

    let mut strength = 0.0;

    // Base from high card (14=Ace → 0.40, 2 → 0.08)
    strength += (high as f64 - 2.0) * 0.027 + 0.08;

    // Bonus for second card
    strength += (low as f64 - 2.0) * 0.012;

    // Suited bonus
    if suited {
        strength += 0.06;
    }

    // Connected/close bonus
    if connected {
        strength += 0.04;
    } else if gap == 2 {
        strength += 0.02;
    }

    strength.clamp(0.0, 0.90)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preflop_aces_strongest() {
        let aces = vec![Card::new(14, 0), Card::new(14, 1)]; // AA
        let deuces = vec![Card::new(2, 0), Card::new(2, 1)]; // 22
        let junk = vec![Card::new(2, 0), Card::new(7, 1)]; // 27o

        let aa = preflop_hand_strength(&aces);
        let twos = preflop_hand_strength(&deuces);
        let bad = preflop_hand_strength(&junk);

        assert!(
            aa > twos,
            "AA ({}) should be stronger than 22 ({})",
            aa,
            twos
        );
        assert!(
            twos > bad,
            "22 ({}) should be stronger than 27o ({})",
            twos,
            bad
        );
        assert!(aa > 0.90, "AA should be > 0.90, got {}", aa);
    }

    #[test]
    fn test_preflop_suited_better_than_offsuit() {
        let suited = vec![Card::new(14, 0), Card::new(13, 0)]; // AKs
        let offsuit = vec![Card::new(14, 0), Card::new(13, 1)]; // AKo

        assert!(
            preflop_hand_strength(&suited) > preflop_hand_strength(&offsuit),
            "AKs should be stronger than AKo"
        );
    }

    #[test]
    fn test_monte_carlo_strong_vs_weak() {
        // AA vs random with no community cards
        let aces = vec![Card::new(14, 0), Card::new(14, 1)];
        let junk = vec![Card::new(2, 0), Card::new(7, 3)];

        let aa_strength = estimate_hand_strength(&aces, &[], 1, 500);
        let junk_strength = estimate_hand_strength(&junk, &[], 1, 500);

        assert!(
            aa_strength > junk_strength,
            "AA ({:.2}) should be stronger than 27o ({:.2})",
            aa_strength,
            junk_strength
        );
        assert!(
            aa_strength > 0.7,
            "AA should win > 70% vs 1 opponent, got {:.2}",
            aa_strength
        );
    }

    #[test]
    fn test_monte_carlo_made_hand() {
        // We have a flush on the flop
        let hole = vec![Card::new(14, 2), Card::new(10, 2)]; // Ah Th
        let community = vec![
            Card::new(5, 2), // 5h
            Card::new(8, 2), // 8h
            Card::new(2, 2), // 2h - flush!
        ];

        let strength = estimate_hand_strength(&hole, &community, 1, 500);
        assert!(
            strength > 0.8,
            "Ace-high flush should be very strong, got {:.2}",
            strength
        );
    }
}
