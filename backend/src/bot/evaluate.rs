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
/// For 2-card hands (Hold'em): uses a chart-based tier lookup modeled after
/// standard preflop opening charts.
/// For 4-card hands (Omaha): falls back to a formula-based heuristic.
pub fn preflop_hand_strength(hole_cards: &[Card]) -> f64 {
    if hole_cards.len() < 2 {
        return 0.0;
    }

    // For Omaha (4 cards), use formula-based fallback
    if hole_cards.len() != 2 {
        return preflop_hand_strength_formula(hole_cards);
    }

    let tier = preflop_tier(hole_cards[0].rank, hole_cards[1].rank, hole_cards[0].suit == hole_cards[1].suit);
    // Tier 0 = 0.95, tier 8 = 0.15
    (0.95 - tier as f64 * 0.10).clamp(0.0, 0.95)
}

/// Returns the preflop tier (0 = premium, 8 = trash) for a 2-card hand.
pub fn preflop_tier(rank1: u8, rank2: u8, suited: bool) -> u8 {
    let high = rank1.max(rank2);
    let low = rank1.min(rank2);
    let pair = rank1 == rank2;

    if pair {
        return match high {
            14 => 0, // AA
            13 => 0, // KK
            12 => 0, // QQ
            11 => 1, // JJ
            10 => 1, // TT
            9 => 2,  // 99
            8 => 2,  // 88
            7 => 3,  // 77
            6 => 3,  // 66
            5 => 4,  // 55
            4 => 4,  // 44
            3 => 5,  // 33
            _ => 5,  // 22
        };
    }

    if suited {
        match (high, low) {
            // Tier 0
            (14, 13) => 0, // AKs
            // Tier 1
            (14, 12) => 1, // AQs
            (14, 11) => 1, // AJs
            // Tier 2
            (14, 10) => 2, // ATs
            (13, 12) => 2, // KQs
            (13, 11) => 2, // KJs
            (12, 11) => 2, // QJs
            // Tier 3
            (14, lo) if (2..=9).contains(&lo) => 3, // A9s-A2s
            (13, 10) => 3, // KTs
            (12, 10) => 3, // QTs
            (11, 10) => 3, // JTs
            (10, 9) => 3,  // T9s
            // Tier 4
            (13, lo) if (7..=9).contains(&lo) => 4, // K9s-K7s
            (12, 9) => 4,  // Q9s
            (11, 9) => 4,  // J9s
            (10, 8) => 4,  // T8s
            (9, 8) => 4,   // 98s
            (8, 7) => 4,   // 87s
            (7, 6) => 4,   // 76s
            // Tier 5
            (13, lo) if (2..=6).contains(&lo) => 5, // K6s-K2s
            (12, lo) if (6..=8).contains(&lo) => 5, // Q8s-Q6s
            (11, 8) => 5,  // J8s
            (10, 7) => 5,  // T7s
            (9, 7) => 5,   // 97s
            (8, 6) => 5,   // 86s
            (7, 5) => 5,   // 75s
            (6, 5) => 5,   // 65s
            (5, 4) => 5,   // 54s
            // Tier 6
            (12, lo) if (2..=5).contains(&lo) => 6, // Q5s-Q2s
            (11, 7) => 6,  // J7s
            (10, 6) => 6,  // T6s
            (9, 6) => 6,   // 96s
            (8, 5) => 6,   // 85s
            (7, 4) => 6,   // 74s
            (6, 4) => 6,   // 64s
            (5, 3) => 6,   // 53s
            (4, 3) => 6,   // 43s
            // Tier 7
            (_, _) => 7,   // remaining suited hands
        }
    } else {
        // Offsuit hands
        match (high, low) {
            // Tier 1
            (14, 13) => 1, // AKo
            // Tier 2
            (14, 12) => 2, // AQo
            // Tier 3
            (14, 11) => 3, // AJo
            (14, 10) => 3, // ATo
            // Tier 4
            (13, 12) => 4, // KQo
            (13, 11) => 4, // KJo
            (12, 11) => 4, // QJo
            // Tier 5
            (13, 10) => 5, // KTo
            (12, 10) => 5, // QTo
            (11, 10) => 5, // JTo
            // Tier 6
            (14, 9) => 6,  // A9o
            (14, 8) => 6,  // A8o
            (13, 9) => 6,  // K9o
            (13, 8) => 6,  // K8o
            (12, 9) => 6,  // Q9o
            (11, 9) => 6,  // J9o
            (10, 9) => 6,  // T9o
            // Tier 7
            (14, lo) if (2..=7).contains(&lo) => 7, // A7o-A2o
            (13, 7) => 7,  // K7o
            (12, 8) => 7,  // Q8o
            (11, 8) => 7,  // J8o
            (10, 8) => 7,  // T8o
            (9, 8) => 7,   // 98o
            // Tier 8
            (_, _) => 8,   // everything else
        }
    }
}

/// Formula-based fallback for Omaha (4+ card) hands.
fn preflop_hand_strength_formula(hole_cards: &[Card]) -> f64 {
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
        return 0.50 + (low as f64 - 2.0) * 0.0375;
    }

    let mut strength = 0.0;
    strength += (high as f64 - 2.0) * 0.027 + 0.08;
    strength += (low as f64 - 2.0) * 0.012;
    if suited {
        strength += 0.06;
    }
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
        assert!(aa >= 0.95, "AA should be >= 0.95, got {}", aa);
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
    fn test_preflop_chart_tiers() {
        // Tier 0 hands should be strongest
        assert_eq!(preflop_tier(14, 14, false), 0); // AA
        assert_eq!(preflop_tier(13, 13, false), 0); // KK
        assert_eq!(preflop_tier(14, 13, true), 0);  // AKs

        // Tier 8 = trash
        assert_eq!(preflop_tier(7, 2, false), 8);  // 72o
        assert_eq!(preflop_tier(3, 2, false), 8);  // 32o
        assert_eq!(preflop_tier(8, 2, false), 8);  // 82o

        // Verify strength ordering maps correctly
        let aa = preflop_hand_strength(&[Card::new(14, 0), Card::new(14, 1)]);
        let jj = preflop_hand_strength(&[Card::new(11, 0), Card::new(11, 1)]);
        let trash = preflop_hand_strength(&[Card::new(7, 0), Card::new(2, 1)]);

        assert!(aa > jj, "AA ({}) > JJ ({})", aa, jj);
        assert!(jj > trash, "JJ ({}) > 72o ({})", jj, trash);
        assert!(aa >= 0.95, "AA = tier 0 = 0.95");
        assert!(trash <= 0.20, "72o = tier 8 = 0.15, got {}", trash);
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
