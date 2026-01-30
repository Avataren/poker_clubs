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
    all_cards.extend(hole_cards.iter().map(|c| c.to_rs_poker()));
    all_cards.extend(community_cards.iter().map(|c| c.to_rs_poker()));

    let hand = Hand::new_with_cards(all_cards);
    HandRank::from_hand(&hand)
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
}
