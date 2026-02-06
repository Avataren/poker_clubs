use crate::game::deck::Card;
use rs_poker::core::{Hand, Rank as RsRank, Rankable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandRank {
    pub rank_value: i32,
    /// Sub-rank within the hand category for proper comparison
    /// (e.g., AAQQ vs AA66 within TwoPair)
    sub_rank: u32,
    pub description: String,
    pub best_cards: Vec<Card>, // The 5 cards that make up the best hand
}

/// Equality is based on hand strength only (rank_value + sub_rank),
/// not on which specific cards/suits are used.
impl PartialEq for HandRank {
    fn eq(&self, other: &Self) -> bool {
        self.rank_value == other.rank_value && self.sub_rank == other.sub_rank
    }
}

impl Eq for HandRank {}

impl HandRank {
    pub fn from_hand(hand: &Hand) -> Self {
        let rs_rank = hand.rank();
        let (rank_value, sub_rank, description) = match &rs_rank {
            RsRank::HighCard(v) => (0, *v, "High Card"),
            RsRank::OnePair(v) => (1, *v, "Pair"),
            RsRank::TwoPair(v) => (2, *v, "Two Pair"),
            RsRank::ThreeOfAKind(v) => (3, *v, "Three of a Kind"),
            RsRank::Straight(v) => (4, *v, "Straight"),
            RsRank::Flush(v) => (5, *v, "Flush"),
            RsRank::FullHouse(v) => (6, *v, "Full House"),
            RsRank::FourOfAKind(v) => (7, *v, "Four of a Kind"),
            RsRank::StraightFlush(v) => (8, *v, "Straight Flush"),
        };

        // Get the best 5 cards from the hand
        let best_cards: Vec<Card> = hand.cards().iter().map(Card::from_rs_poker).collect();

        Self {
            rank_value,
            sub_rank,
            description: description.to_string(),
            best_cards,
        }
    }
}

impl PartialOrd for HandRank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HandRank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank_value
            .cmp(&other.rank_value)
            .then_with(|| self.sub_rank.cmp(&other.sub_rank))
    }
}

/// Represents a qualifying low hand in hi-lo games (8-or-better)
/// Lower values are better. The rank is the 5 cards sorted highest to lowest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LowHandRank {
    /// The 5 card ranks sorted highest to lowest (e.g., [8,6,4,3,2])
    /// Lower is better - compared lexicographically
    pub ranks: [u8; 5],
    pub description: String,
    pub best_cards: Vec<Card>,
}

impl Ord for LowHandRank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower is better: compare lexicographically, but reverse the ordering
        // so that a lower hand sorts as "less than" a higher hand.
        // e.g., [5,4,3,2,1] < [8,7,6,5,4] because [5,4,3,2,1] is a better (lower) hand
        self.ranks.cmp(&other.ranks)
    }
}

impl PartialOrd for LowHandRank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Evaluates the best Omaha low hand: MUST use exactly 2 hole cards and 3 community cards
/// A qualifying low hand has 5 unique card ranks, all 8 or below (aces count as 1).
/// Straights and flushes do NOT disqualify a low hand.
/// Returns None if no qualifying low exists.
pub fn evaluate_omaha_low_hand(
    hole_cards: &[Card],
    community_cards: &[Card],
) -> Option<LowHandRank> {
    if hole_cards.len() < 2 || community_cards.len() < 3 {
        return None;
    }

    let hole_combos = combinations(hole_cards, 2);
    let community_combos = combinations(community_cards, 3);

    let mut best_low: Option<LowHandRank> = None;

    for hole_combo in &hole_combos {
        for community_combo in &community_combos {
            // Collect the 5 cards
            let mut five_cards: Vec<&Card> = Vec::with_capacity(5);
            five_cards.extend(hole_combo.iter());
            five_cards.extend(community_combo.iter());

            // Convert ranks: Ace (14) counts as 1 for low evaluation
            let mut low_ranks: Vec<u8> = five_cards
                .iter()
                .map(|c| if c.rank == 14 { 1 } else { c.rank })
                .collect();

            // Sort and deduplicate to check for 5 unique ranks
            low_ranks.sort_unstable();
            low_ranks.dedup();

            // Must have 5 unique ranks, all <= 8
            if low_ranks.len() != 5 || low_ranks.iter().any(|&r| r > 8) {
                continue;
            }

            // Sort descending for comparison (highest card first)
            low_ranks.sort_unstable_by(|a, b| b.cmp(a));
            let ranks: [u8; 5] = [
                low_ranks[0],
                low_ranks[1],
                low_ranks[2],
                low_ranks[3],
                low_ranks[4],
            ];

            // Build description
            let description = format!(
                "{}-{}-{}-{}-{} low",
                ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]
            );

            // Build best_cards list
            let best_cards: Vec<Card> = five_cards.iter().copied().cloned().collect();

            let low_rank = LowHandRank {
                ranks,
                description,
                best_cards,
            };

            best_low = match best_low {
                None => Some(low_rank),
                Some(ref current) if low_rank < *current => Some(low_rank),
                _ => best_low,
            };
        }
    }

    best_low
}

/// Evaluates the best 5-card hand from a player's hole cards and community cards
pub fn evaluate_hand(hole_cards: &[Card], community_cards: &[Card]) -> HandRank {
    let mut all_cards = Vec::new();
    all_cards.extend_from_slice(hole_cards);
    all_cards.extend_from_slice(community_cards);

    // Find the best 5-card combination out of all available cards
    // Use rs_poker's native Rank comparison for proper ordering within hand categories
    let combos = combinations(&all_cards, 5);
    let best_hand = combos
        .into_iter()
        .map(|five_cards| {
            let rs_cards: Vec<rs_poker::core::Card> =
                five_cards.iter().map(|c| c.to_rs_poker()).collect();
            Hand::new_with_cards(rs_cards)
        })
        .max_by_key(|hand| hand.rank())
        .expect("should have at least one 5-card combination");
    HandRank::from_hand(&best_hand)
}

/// Evaluates the best Omaha hand: MUST use exactly 2 hole cards and 3 community cards
/// Returns None if not enough cards to make a valid hand
pub fn evaluate_omaha_hand(hole_cards: &[Card], community_cards: &[Card]) -> Option<HandRank> {
    // Omaha requires at least 4 hole cards and 3 community cards for a valid hand
    if hole_cards.len() < 2 || community_cards.len() < 3 {
        return None;
    }

    let mut best_rank: Option<HandRank> = None;

    // Generate all 2-card combinations from hole cards
    let hole_combos = combinations(hole_cards, 2);
    // Generate all 3-card combinations from community cards
    let community_combos = combinations(community_cards, 3);

    // Try every combination of 2 hole + 3 community
    for hole_combo in &hole_combos {
        for community_combo in &community_combos {
            let mut hand_cards: Vec<rs_poker::core::Card> = Vec::with_capacity(5);
            hand_cards.extend(hole_combo.iter().map(|c| c.to_rs_poker()));
            hand_cards.extend(community_combo.iter().map(|c| c.to_rs_poker()));

            let hand = Hand::new_with_cards(hand_cards);
            let rank = HandRank::from_hand(&hand);

            best_rank = match best_rank {
                None => Some(rank),
                Some(ref current) if rank > *current => Some(rank),
                _ => best_rank,
            };
        }
    }

    best_rank
}

/// Generate all k-combinations from a slice
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }

    let mut result = Vec::new();

    // Take the first item and combine with (k-1) combinations from rest
    let first = &items[0];
    let rest = &items[1..];

    for mut combo in combinations(rest, k - 1) {
        combo.insert(0, first.clone());
        result.push(combo);
    }

    // Also get combinations that don't include the first item
    result.extend(combinations(rest, k));

    result
}

/// Determines the winner(s) from multiple hands
/// Returns indices of winning players
pub fn determine_winners(hands: Vec<(usize, HandRank)>) -> Vec<usize> {
    if hands.is_empty() {
        return vec![];
    }

    // Find the best hand rank
    let best_rank = hands.iter().map(|(_, rank)| rank).max().unwrap().clone();

    // Debug logging for all hand comparisons
    tracing::info!("=== WINNER DETERMINATION ===");
    for (idx, rank) in &hands {
        tracing::info!(
            "Player {}: {} (rank_value={}, sub_rank={}, best_cards={:?})",
            idx,
            rank.description,
            rank.rank_value,
            rank.sub_rank,
            rank.best_cards
                .iter()
                .map(|c| format!(
                    "{}{}. ",
                    match c.rank {
                        14 => "A",
                        13 => "K",
                        12 => "Q",
                        11 => "J",
                        10 => "T",
                        r => return r.to_string(),
                    },
                    match c.suit {
                        0 => "♣",
                        1 => "♦",
                        2 => "♥",
                        3 => "♠",
                        _ => "?",
                    }
                ))
                .collect::<String>()
        );
    }
    tracing::info!(
        "Best rank: {} (rank_value={}, sub_rank={})",
        best_rank.description,
        best_rank.rank_value,
        best_rank.sub_rank
    );

    let winners: Vec<usize> = hands
        .iter()
        .filter(|(_, rank)| rank == &best_rank)
        .map(|(idx, _)| *idx)
        .collect();

    tracing::info!("Winners: {:?}", winners);

    // Return all players with the best hand (handles ties)
    hands
        .into_iter()
        .filter(move |(_, rank)| rank == &best_rank)
        .map(|(idx, _)| idx)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_royal_flush() {
        let hole_cards = vec![
            Card::new(14, 3), // Ace of Spades
            Card::new(13, 3), // King of Spades
        ];
        let community_cards = vec![
            Card::new(12, 3), // Queen of Spades
            Card::new(11, 3), // Jack of Spades
            Card::new(10, 3), // Ten of Spades
        ];

        let hand_rank = evaluate_hand(&hole_cards, &community_cards);
        assert_eq!(hand_rank.description, "Straight Flush");
    }

    #[test]
    fn test_pair() {
        let hole_cards = vec![
            Card::new(14, 3), // Ace of Spades
            Card::new(14, 2), // Ace of Hearts
        ];
        let community_cards = vec![
            Card::new(2, 1), // Two of Diamonds
            Card::new(5, 0), // Five of Clubs
            Card::new(9, 3), // Nine of Spades
        ];

        let hand_rank = evaluate_hand(&hole_cards, &community_cards);
        assert_eq!(hand_rank.description, "Pair");
    }

    #[test]
    fn test_determine_winners_single() {
        // Higher rank_value means better hand
        let hands = vec![
            (
                0,
                HandRank {
                    rank_value: 3,
                    sub_rank: 0,
                    description: "Three of a Kind".to_string(),
                    best_cards: vec![],
                },
            ),
            (
                1,
                HandRank {
                    rank_value: 1,
                    sub_rank: 0,
                    description: "Pair".to_string(),
                    best_cards: vec![],
                },
            ),
            (
                2,
                HandRank {
                    rank_value: 2,
                    sub_rank: 0,
                    description: "Two Pair".to_string(),
                    best_cards: vec![],
                },
            ),
        ];

        let winners = determine_winners(hands);
        assert_eq!(winners, vec![0]); // Player 0 has the best hand (highest rank_value)
    }

    #[test]
    fn test_determine_winners_tie() {
        let hands = vec![
            (
                0,
                HandRank {
                    rank_value: 3,
                    sub_rank: 100,
                    description: "Three of a Kind".to_string(),
                    best_cards: vec![],
                },
            ),
            (
                1,
                HandRank {
                    rank_value: 3,
                    sub_rank: 100,
                    description: "Three of a Kind".to_string(),
                    best_cards: vec![],
                },
            ),
        ];

        let winners = determine_winners(hands);
        assert_eq!(winners.len(), 2);
        assert!(winners.contains(&0));
        assert!(winners.contains(&1));
    }

    #[test]
    fn test_two_pair_comparison() {
        // QQ on board A3A86 should beat 76 on same board
        // QQ → best hand: AAQQ8 (two pair, aces and queens)
        // 76 → best hand: AA668 (two pair, aces and sixes)
        let community = vec![
            Card::new(14, 0), // A
            Card::new(3, 1),  // 3
            Card::new(14, 2), // A
            Card::new(8, 3),  // 8
            Card::new(6, 0),  // 6
        ];

        let qq_hand = evaluate_hand(
            &[Card::new(12, 1), Card::new(12, 3)], // QQ
            &community,
        );
        let s76_hand = evaluate_hand(
            &[Card::new(7, 1), Card::new(6, 2)], // 76
            &community,
        );

        assert_eq!(qq_hand.description, "Two Pair");
        assert_eq!(s76_hand.description, "Two Pair");
        assert!(
            qq_hand > s76_hand,
            "AAQQ should beat AA66: qq={:?} vs 76={:?}",
            qq_hand,
            s76_hand
        );
    }

    #[test]
    fn test_combinations() {
        let items = vec![1, 2, 3, 4];
        let combos = combinations(&items, 2);
        assert_eq!(combos.len(), 6); // C(4,2) = 6
        assert!(combos.contains(&vec![1, 2]));
        assert!(combos.contains(&vec![1, 3]));
        assert!(combos.contains(&vec![1, 4]));
        assert!(combos.contains(&vec![2, 3]));
        assert!(combos.contains(&vec![2, 4]));
        assert!(combos.contains(&vec![3, 4]));
    }

    #[test]
    fn test_omaha_hand_must_use_two_hole_cards() {
        // Omaha scenario: Player has AA in hole but board has KKKK
        // In Hold'em, this would be quads (K). In Omaha, player MUST use 2 hole cards
        // so the best hand is AAAKK (full house using 2 aces + 3 kings)
        let hole_cards = vec![
            Card::new(14, 0), // Ace of Clubs
            Card::new(14, 1), // Ace of Diamonds
            Card::new(2, 2),  // Two of Hearts
            Card::new(3, 3),  // Three of Spades
        ];
        let community_cards = vec![
            Card::new(13, 0), // King of Clubs
            Card::new(13, 1), // King of Diamonds
            Card::new(13, 2), // King of Hearts
            Card::new(13, 3), // King of Spades
            Card::new(7, 0),  // Seven of Clubs
        ];

        let omaha_rank = evaluate_omaha_hand(&hole_cards, &community_cards).unwrap();
        // In Omaha, best is Full House (AA + KKK) using 2 aces from hole + 3 kings from board
        assert_eq!(omaha_rank.description, "Full House");
    }

    #[test]
    fn test_omaha_vs_holdem_different_results() {
        // Scenario where Omaha and Hold'em give different results
        // Hole: Ah Kh 2c 3d - has A-high flush draw in hearts
        // Board: Qh Jh Th 5s 6s - has heart flush on board
        // Hold'em would use just the Ah for A-high flush
        // Omaha MUST use 2 hole cards, so uses Ah + Kh for A-K high flush
        let hole_cards = vec![
            Card::new(14, 2), // Ace of Hearts
            Card::new(13, 2), // King of Hearts
            Card::new(2, 0),  // Two of Clubs
            Card::new(3, 1),  // Three of Diamonds
        ];
        let community_cards = vec![
            Card::new(12, 2), // Queen of Hearts
            Card::new(11, 2), // Jack of Hearts
            Card::new(10, 2), // Ten of Hearts
            Card::new(5, 3),  // Five of Spades
            Card::new(6, 3),  // Six of Spades
        ];

        let omaha_rank = evaluate_omaha_hand(&hole_cards, &community_cards).unwrap();
        // Royal flush: A K Q J T all hearts - player uses Ah Kh + QhJhTh
        assert_eq!(omaha_rank.description, "Straight Flush");
    }

    #[test]
    fn test_flush_vs_flush_different_high_cards() {
        // Test that A-high flush beats K-high flush
        let community = vec![
            Card::new(10, 2), // Th
            Card::new(8, 2),  // 8h
            Card::new(6, 2),  // 6h
            Card::new(4, 2),  // 4h
            Card::new(2, 0),  // 2c
        ];

        // Player 1: Ah 3h = A-high flush (Ah Th 8h 6h 4h)
        let player1 = evaluate_hand(
            &[Card::new(14, 2), Card::new(3, 2)], // Ah 3h
            &community,
        );

        // Player 2: Kh 5h = K-high flush (Kh Th 8h 6h 5h)
        let player2 = evaluate_hand(
            &[Card::new(13, 2), Card::new(5, 2)], // Kh 5h
            &community,
        );

        assert_eq!(player1.description, "Flush");
        assert_eq!(player2.description, "Flush");
        assert!(player1 > player2, "A-high flush should beat K-high flush");

        let winners = determine_winners(vec![(0, player1.clone()), (1, player2.clone())]);
        assert_eq!(
            winners,
            vec![0],
            "Player 0 with A-high flush should win alone"
        );
    }

    #[test]
    fn test_flush_vs_flush_same_high_card_different_kicker() {
        // Both have K-high flush but different second cards
        let community = vec![
            Card::new(13, 1), // Kd
            Card::new(9, 1),  // 9d
            Card::new(7, 1),  // 7d
            Card::new(5, 1),  // 5d
            Card::new(2, 0),  // 2c
        ];

        // Player 1: Qd 3d = K-high flush with Q kicker
        let player1 = evaluate_hand(
            &[Card::new(12, 1), Card::new(3, 1)], // Qd 3d
            &community,
        );

        // Player 2: Jd 4d = K-high flush with J kicker
        let player2 = evaluate_hand(
            &[Card::new(11, 1), Card::new(4, 1)], // Jd 4d
            &community,
        );

        assert_eq!(player1.description, "Flush");
        assert_eq!(player2.description, "Flush");
        assert!(
            player1 > player2,
            "K-Q-9-7-5 flush should beat K-J-9-7-5 flush"
        );

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(
            winners,
            vec![0],
            "Player 0 with higher kicker should win alone"
        );
    }

    #[test]
    fn test_identical_flushes_split_pot() {
        // Both players make the exact same flush using board cards
        let community = vec![
            Card::new(14, 3), // As
            Card::new(13, 3), // Ks
            Card::new(11, 3), // Js
            Card::new(9, 3),  // 9s
            Card::new(7, 3),  // 7s
        ];

        // Player 1: 2s 3s (board flush is better)
        let player1 = evaluate_hand(&[Card::new(2, 3), Card::new(3, 3)], &community);

        // Player 2: 4s 5s (board flush is better)
        let player2 = evaluate_hand(&[Card::new(4, 3), Card::new(5, 3)], &community);

        assert_eq!(player1.description, "Flush");
        assert_eq!(player2.description, "Flush");
        assert_eq!(
            player1, player2,
            "Both should have identical A-K-J-9-7 flush"
        );

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(winners.len(), 2, "Should be a split pot");
        assert!(winners.contains(&0) && winners.contains(&1));
    }

    #[test]
    fn test_straight_vs_straight() {
        let community = vec![
            Card::new(9, 0), // 9c
            Card::new(8, 1), // 8d
            Card::new(7, 2), // 7h
            Card::new(6, 3), // 6s
            Card::new(2, 0), // 2c
        ];

        // Player 1: T5 = T-high straight (T9876)
        let player1 = evaluate_hand(&[Card::new(10, 1), Card::new(5, 2)], &community);

        // Player 2: 53 = 9-high straight (98765)
        let player2 = evaluate_hand(&[Card::new(5, 1), Card::new(3, 2)], &community);

        assert_eq!(player1.description, "Straight");
        assert_eq!(player2.description, "Straight");
        assert!(
            player1 > player2,
            "T-high straight should beat 9-high straight"
        );

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(winners, vec![0]);
    }

    #[test]
    fn test_full_house_vs_full_house() {
        let community = vec![
            Card::new(10, 0), // Tc
            Card::new(10, 1), // Td
            Card::new(10, 2), // Th
            Card::new(5, 3),  // 5s
            Card::new(3, 0),  // 3c
        ];

        // Player 1: 55 = TTT55 (tens full of fives)
        let player1 = evaluate_hand(&[Card::new(5, 0), Card::new(5, 1)], &community);

        // Player 2: 33 = TTT33 (tens full of threes)
        let player2 = evaluate_hand(&[Card::new(3, 1), Card::new(3, 2)], &community);

        assert_eq!(player1.description, "Full House");
        assert_eq!(player2.description, "Full House");
        assert!(
            player1 > player2,
            "Tens full of fives should beat tens full of threes"
        );

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(winners, vec![0]);
    }

    #[test]
    fn test_quads_vs_quads() {
        let community = vec![
            Card::new(10, 0), // Tc
            Card::new(10, 1), // Td
            Card::new(10, 2), // Th
            Card::new(10, 3), // Ts
            Card::new(3, 0),  // 3c
        ];

        // Player 1: AK = TTTTA (quads with ace kicker)
        let player1 = evaluate_hand(&[Card::new(14, 1), Card::new(13, 2)], &community);

        // Player 2: Q9 = TTTTQ (quads with queen kicker)
        let player2 = evaluate_hand(&[Card::new(12, 1), Card::new(9, 2)], &community);

        assert_eq!(player1.description, "Four of a Kind");
        assert_eq!(player2.description, "Four of a Kind");
        assert!(
            player1 > player2,
            "Quads with A kicker should beat quads with Q kicker"
        );

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(winners, vec![0]);
    }

    #[test]
    fn test_high_card_vs_high_card() {
        let community = vec![
            Card::new(13, 0), // Kc
            Card::new(11, 1), // Jd
            Card::new(9, 2),  // 9h
            Card::new(7, 3),  // 7s
            Card::new(5, 0),  // 5c
        ];

        // Player 1: A2 = AKJ97 high
        let player1 = evaluate_hand(&[Card::new(14, 1), Card::new(2, 2)], &community);

        // Player 2: Q3 = KQJ97 high
        let player2 = evaluate_hand(&[Card::new(12, 1), Card::new(3, 2)], &community);

        assert_eq!(player1.description, "High Card");
        assert_eq!(player2.description, "High Card");
        assert!(player1 > player2, "A-high should beat K-high");

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(winners, vec![0]);
    }

    #[test]
    fn test_flush_beats_straight() {
        let community = vec![
            Card::new(10, 2), // Th
            Card::new(9, 2),  // 9h
            Card::new(8, 2),  // 8h
            Card::new(7, 0),  // 7c
            Card::new(6, 1),  // 6d
        ];

        // Player 1: Ah 2h = A-high flush
        let player1 = evaluate_hand(&[Card::new(14, 2), Card::new(2, 2)], &community);

        // Player 2: JQ = Q-high straight
        let player2 = evaluate_hand(&[Card::new(11, 0), Card::new(12, 1)], &community);

        assert_eq!(player1.description, "Flush");
        assert_eq!(player2.description, "Straight");
        assert!(player1 > player2, "Flush should beat straight");

        let winners = determine_winners(vec![(0, player1), (1, player2)]);
        assert_eq!(winners, vec![0]);
    }

    #[test]
    fn test_three_way_different_flushes() {
        let community = vec![
            Card::new(12, 1), // Qd
            Card::new(10, 1), // Td
            Card::new(8, 1),  // 8d
            Card::new(6, 1),  // 6d
            Card::new(2, 0),  // 2c
        ];

        // Player 0: Ad 3d = A-high flush
        let player0 = evaluate_hand(&[Card::new(14, 1), Card::new(3, 1)], &community);

        // Player 1: Kd 4d = K-high flush
        let player1 = evaluate_hand(&[Card::new(13, 1), Card::new(4, 1)], &community);

        // Player 2: 9d 5d = Q-high flush
        let player2 = evaluate_hand(&[Card::new(9, 1), Card::new(5, 1)], &community);

        assert_eq!(player0.description, "Flush");
        assert_eq!(player1.description, "Flush");
        assert_eq!(player2.description, "Flush");

        assert!(player0 > player1, "A-high flush should beat K-high flush");
        assert!(player1 > player2, "K-high flush should beat Q-high flush");
        assert!(player0 > player2, "A-high flush should beat Q-high flush");

        let winners = determine_winners(vec![(0, player0), (1, player1), (2, player2)]);
        assert_eq!(
            winners,
            vec![0],
            "Only player 0 with A-high flush should win"
        );
    }

    #[test]
    fn test_two_pair_ace_kicker_vs_jack_kicker() {
        // Board: 9d 9h 6d Jc 4h
        // Bob: 6c Ad = 99 66 A (two pair with ace kicker)
        // Diana: 6s Th = 99 66 J (two pair with jack kicker)
        // Bob should win because A > J
        let community = vec![
            Card::new(9, 1),  // 9d
            Card::new(9, 2),  // 9h
            Card::new(6, 1),  // 6d
            Card::new(11, 0), // Jc
            Card::new(4, 2),  // 4h
        ];

        // Bob: 6c Ad
        let bob = evaluate_hand(
            &[Card::new(6, 0), Card::new(14, 1)], // 6c Ad
            &community,
        );

        // Diana: 6s Th
        let diana = evaluate_hand(
            &[Card::new(6, 3), Card::new(10, 2)], // 6s Th
            &community,
        );

        assert_eq!(bob.description, "Two Pair");
        assert_eq!(diana.description, "Two Pair");
        assert!(
            bob > diana,
            "Two pair 99 66 with A kicker should beat 99 66 with J kicker"
        );

        let winners = determine_winners(vec![(0, bob), (1, diana)]);
        assert_eq!(winners, vec![0], "Bob with ace kicker should win alone");
    }

    #[test]
    fn test_two_pair_board_better_than_both_players() {
        // Board: AA 77 K - both players have worse hole cards
        // Both players play the board for AA 77 K
        let community = vec![
            Card::new(14, 1), // Ad
            Card::new(14, 0), // Ac
            Card::new(7, 2),  // 7h
            Card::new(7, 3),  // 7s
            Card::new(13, 0), // Kc
        ];

        // Alice: JT (worse than board)
        let alice = evaluate_hand(
            &[Card::new(11, 1), Card::new(10, 3)], // Jd Ts
            &community,
        );

        // Bob: 85 (worse than board)
        let bob = evaluate_hand(
            &[Card::new(8, 0), Card::new(5, 0)], // 8c 5c
            &community,
        );

        assert_eq!(alice.description, "Two Pair");
        assert_eq!(bob.description, "Two Pair");
        assert_eq!(
            alice, bob,
            "Both should have identical AA 77 K using the board"
        );

        let winners = determine_winners(vec![(0, alice), (1, bob)]);
        assert_eq!(
            winners.len(),
            2,
            "Should be a split pot - both play the board"
        );
        assert!(winners.contains(&0) && winners.contains(&1));
    }

    #[test]

    fn test_omaha_returns_none_with_insufficient_cards() {
        let hole_cards = vec![
            Card::new(14, 0), // Ace
        ];
        let community_cards = vec![
            Card::new(13, 0), // King
            Card::new(12, 0), // Queen
        ];

        // Not enough cards for Omaha
        assert!(evaluate_omaha_hand(&hole_cards, &community_cards).is_none());
    }

    #[test]
    fn test_low_hand_wheel_is_best() {
        let hole_cards = vec![
            Card::new(14, 0), // Ace (plays as low)
            Card::new(2, 1),
            Card::new(10, 2),
            Card::new(11, 3),
        ];
        let community_cards = vec![
            Card::new(3, 0),
            Card::new(4, 1),
            Card::new(5, 2),
            Card::new(13, 3),
            Card::new(12, 0),
        ];
        let low = evaluate_omaha_low_hand(&hole_cards, &community_cards);
        assert!(low.is_some());
        let low = low.unwrap();
        assert_eq!(low.ranks, [5, 4, 3, 2, 1]); // Wheel: 5-4-3-2-A
    }

    #[test]
    fn test_low_hand_no_qualifying() {
        // All community cards are 9+, no qualifying low
        let hole_cards = vec![
            Card::new(14, 0),
            Card::new(2, 1),
            Card::new(3, 2),
            Card::new(4, 3),
        ];
        let community_cards = vec![
            Card::new(9, 0),
            Card::new(10, 1),
            Card::new(11, 2),
            Card::new(12, 3),
            Card::new(13, 0),
        ];
        let low = evaluate_omaha_low_hand(&hole_cards, &community_cards);
        assert!(low.is_none());
    }

    #[test]
    fn test_low_hand_comparison() {
        // 8-7-6-5-4 vs 7-5-4-3-2 - second is better (lower)
        let low1 = LowHandRank {
            ranks: [8, 7, 6, 5, 4],
            description: "8-7-6-5-4 low".to_string(),
            best_cards: vec![],
        };
        let low2 = LowHandRank {
            ranks: [7, 5, 4, 3, 2],
            description: "7-5-4-3-2 low".to_string(),
            best_cards: vec![],
        };
        assert!(low2 < low1); // Lower is better
    }

    #[test]
    fn test_low_hand_unique_ranks_required() {
        // Pairs don't qualify for low - need 5 unique ranks
        let hole_cards = vec![
            Card::new(2, 0),
            Card::new(2, 1),  // Pair of 2s
            Card::new(10, 2),
            Card::new(11, 3),
        ];
        let community_cards = vec![
            Card::new(3, 0),
            Card::new(3, 1), // Pair of 3s
            Card::new(8, 2),
            Card::new(13, 3),
            Card::new(12, 0),
        ];
        // The only combos with 2 hole cards all use at least one 2, and community has 3,3,8
        // With hole (2,10) or (2,11) + community, we can't make 5 unique low ranks
        // With hole (2,2) + community (3,3,8) = ranks 2,2,3,3,8 - not 5 unique
        // hole(2,10) + community(3,8,K) = 2,3,8,10,K - 10 and K don't qualify
        // hole(2,10) + community(3,3,8) = 2,3,3,8,10 - not unique, 10 > 8
        // hole(2,11) similarly fails
        // So no qualifying low should exist here
        let low = evaluate_omaha_low_hand(&hole_cards, &community_cards);
        assert!(low.is_none(), "Should not qualify: {:?}", low);
    }
}
