use crate::game::deck::Card;
use rs_poker::core::{Hand, Rankable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandRank {
    pub rank_value: i32,
    pub description: String,
    pub best_cards: Vec<Card>,  // The 5 cards that make up the best hand
}

impl HandRank {
    pub fn from_hand(hand: &Hand) -> Self {
        use rs_poker::core::Rank as RsRank;

        let rs_rank = hand.rank();
        let (rank_value, description) = match rs_rank {
            RsRank::HighCard(_) => (0, "High Card"),
            RsRank::OnePair(_) => (1, "Pair"),
            RsRank::TwoPair(_) => (2, "Two Pair"),
            RsRank::ThreeOfAKind(_) => (3, "Three of a Kind"),
            RsRank::Straight(_) => (4, "Straight"),
            RsRank::Flush(_) => (5, "Flush"),
            RsRank::FullHouse(_) => (6, "Full House"),
            RsRank::FourOfAKind(_) => (7, "Four of a Kind"),
            RsRank::StraightFlush(_) => (8, "Straight Flush"),
        };

        // Get the best 5 cards from the hand
        let best_cards: Vec<Card> = hand.cards()
            .iter()
            .map(Card::from_rs_poker)
            .collect();

        Self {
            rank_value,
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
        self.rank_value.cmp(&other.rank_value)
    }
}

/// Evaluates the best 5-card hand from a player's hole cards and community cards
pub fn evaluate_hand(hole_cards: &[Card], community_cards: &[Card]) -> HandRank {
    let mut all_cards = Vec::new();
    all_cards.extend_from_slice(hole_cards);
    all_cards.extend_from_slice(community_cards);

    // Find the best 5-card combination out of all available cards
    // Use rs_poker's native Rank comparison for proper ordering within hand categories
    let combos = combinations(&all_cards, 5);
    let best_hand = combos.into_iter()
        .map(|five_cards| {
            let rs_cards: Vec<rs_poker::core::Card> = five_cards.iter().map(|c| c.to_rs_poker()).collect();
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
            Card::new(2, 1),  // Two of Diamonds
            Card::new(5, 0),  // Five of Clubs
            Card::new(9, 3),  // Nine of Spades
        ];

        let hand_rank = evaluate_hand(&hole_cards, &community_cards);
        assert_eq!(hand_rank.description, "Pair");
    }

    #[test]
    fn test_determine_winners_single() {
        // Higher rank_value means better hand
        let hands = vec![
            (0, HandRank { rank_value: 5000, description: "Three of a Kind".to_string(), best_cards: vec![] }),
            (1, HandRank { rank_value: 1000, description: "Pair".to_string(), best_cards: vec![] }),
            (2, HandRank { rank_value: 3000, description: "Two Pair".to_string(), best_cards: vec![] }),
        ];

        let winners = determine_winners(hands);
        assert_eq!(winners, vec![0]); // Player 0 has the best hand (highest rank_value)
    }

    #[test]
    fn test_determine_winners_tie() {
        let hands = vec![
            (0, HandRank { rank_value: 1000, description: "Three of a Kind".to_string(), best_cards: vec![] }),
            (1, HandRank { rank_value: 1000, description: "Three of a Kind".to_string(), best_cards: vec![] }),
        ];

        let winners = determine_winners(hands);
        assert_eq!(winners.len(), 2);
        assert!(winners.contains(&0));
        assert!(winners.contains(&1));
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
}
