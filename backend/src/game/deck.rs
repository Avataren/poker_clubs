use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt;

// Simple card representation for our poker game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Card {
    pub rank: u8,  // 2-14 (Jack=11, Queen=12, King=13, Ace=14)
    pub suit: u8,  // 0-3 (Clubs, Diamonds, Hearts, Spades)
    pub highlighted: bool,  // Whether this card is part of the winning hand
    pub face_up: bool,  // Whether this card is visible (false = face down/hidden)
}

impl Card {
    pub fn new(rank: u8, suit: u8) -> Self {
        Self { rank, suit, highlighted: false, face_up: true }
    }

    fn suit_char(suit: u8) -> char {
        match suit {
            0 => '♣',
            1 => '♦',
            2 => '♥',
            3 => '♠',
            _ => '?',
        }
    }

    // Convert to rs_poker Card for hand evaluation
    pub fn to_rs_poker(&self) -> rs_poker::core::Card {
        use rs_poker::core::{Suit, Value};

        let rank = match self.rank {
            2 => Value::Two,
            3 => Value::Three,
            4 => Value::Four,
            5 => Value::Five,
            6 => Value::Six,
            7 => Value::Seven,
            8 => Value::Eight,
            9 => Value::Nine,
            10 => Value::Ten,
            11 => Value::Jack,
            12 => Value::Queen,
            13 => Value::King,
            14 => Value::Ace,
            _ => Value::Two,
        };

        let suit = match self.suit {
            0 => Suit::Club,
            1 => Suit::Diamond,
            2 => Suit::Heart,
            3 => Suit::Spade,
            _ => Suit::Club,
        };

        rs_poker::core::Card { value: rank, suit }
    }

    // Convert from rs_poker Card back to our Card
    pub fn from_rs_poker(card: &rs_poker::core::Card) -> Self {
        use rs_poker::core::{Value, Suit};

        let rank = match card.value {
            Value::Two => 2,
            Value::Three => 3,
            Value::Four => 4,
            Value::Five => 5,
            Value::Six => 6,
            Value::Seven => 7,
            Value::Eight => 8,
            Value::Nine => 9,
            Value::Ten => 10,
            Value::Jack => 11,
            Value::Queen => 12,
            Value::King => 13,
            Value::Ace => 14,
        };

        let suit = match card.suit {
            Suit::Club => 0,
            Suit::Diamond => 1,
            Suit::Heart => 2,
            Suit::Spade => 3,
        };

        Card::new(rank, suit)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rank_str = match self.rank {
            11 => "J".to_string(),
            12 => "Q".to_string(),
            13 => "K".to_string(),
            14 => "A".to_string(),
            n => n.to_string(),
        };
        write!(f, "{}{}", rank_str, Self::suit_char(self.suit))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deck {
    cards: Vec<Card>,
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

impl Deck {
    /// Creates a new standard 52-card deck
    pub fn new() -> Self {
        let mut cards = Vec::with_capacity(52);

        // 4 suits: Clubs=0, Diamonds=1, Hearts=2, Spades=3
        // 13 ranks: 2-10, Jack=11, Queen=12, King=13, Ace=14
        for suit in 0..4 {
            for rank in 2..=14 {
                cards.push(Card::new(rank, suit));
            }
        }

        Self { cards }
    }

    /// Shuffles the deck using Fisher-Yates algorithm with ChaCha20 RNG for cryptographic security
    pub fn shuffle(&mut self) {
        let mut rng = ChaCha20Rng::from_entropy();
        self.cards.shuffle(&mut rng);
    }

    /// Deals a single card from the deck
    pub fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }

    /// Deals multiple cards from the deck
    pub fn deal_multiple(&mut self, count: usize) -> Vec<Card> {
        let mut dealt = Vec::new();
        for _ in 0..count {
            if let Some(card) = self.deal() {
                dealt.push(card);
            }
        }
        dealt
    }

    /// Returns the number of remaining cards
    #[allow(dead_code)] // Useful for debugging and validation
    pub fn remaining(&self) -> usize {
        self.cards.len()
    }

    /// Resets the deck to a full 52-card deck and shuffles
    pub fn reset_and_shuffle(&mut self) {
        *self = Self::new();
        self.shuffle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_deck_has_52_cards() {
        let deck = Deck::new();
        assert_eq!(deck.remaining(), 52);
    }

    #[test]
    fn test_shuffle_maintains_card_count() {
        let mut deck = Deck::new();
        deck.shuffle();
        assert_eq!(deck.remaining(), 52);
    }

    #[test]
    fn test_deal_reduces_deck_size() {
        let mut deck = Deck::new();
        deck.deal();
        assert_eq!(deck.remaining(), 51);
    }

    #[test]
    fn test_deal_multiple() {
        let mut deck = Deck::new();
        let cards = deck.deal_multiple(5);
        assert_eq!(cards.len(), 5);
        assert_eq!(deck.remaining(), 47);
    }

    #[test]
    fn test_card_to_string() {
        let card = Card::new(14, 3); // Ace of Spades
        assert!(card.to_string().contains("A"));
    }
}
