use std::fmt;

/// Minimal card: rank 2-14, suit 0-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: u8, // 2-14 (J=11, Q=12, K=13, A=14)
    pub suit: u8, // 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
}

impl Card {
    pub fn new(rank: u8, suit: u8) -> Self {
        Self { rank, suit }
    }

    /// Unique index 0..51 for one-hot encoding.
    pub fn index(&self) -> usize {
        (self.suit as usize) * 13 + (self.rank as usize - 2)
    }

    pub fn from_index(idx: usize) -> Self {
        Self {
            suit: (idx / 13) as u8,
            rank: (idx % 13) as u8 + 2,
        }
    }

    pub fn to_rs_poker(&self) -> rs_poker::core::Card {
        use rs_poker::core::{Suit, Value};
        let value = match self.rank {
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
        rs_poker::core::Card { value, suit }
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
        let suit_ch = match self.suit {
            0 => 'c',
            1 => 'd',
            2 => 'h',
            3 => 's',
            _ => '?',
        };
        write!(f, "{}{}", rank_str, suit_ch)
    }
}

/// Standard 52-card deck backed by a Vec.
#[derive(Debug, Clone)]
pub struct Deck {
    pub cards: Vec<Card>,
}

impl Deck {
    pub fn new() -> Self {
        let mut cards = Vec::with_capacity(52);
        for suit in 0..4u8 {
            for rank in 2..=14u8 {
                cards.push(Card::new(rank, suit));
            }
        }
        Self { cards }
    }

    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        use rand::seq::SliceRandom;
        self.cards.shuffle(rng);
    }

    pub fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }

    pub fn remaining(&self) -> usize {
        self.cards.len()
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deck_52_cards() {
        let deck = Deck::new();
        assert_eq!(deck.remaining(), 52);
    }

    #[test]
    fn test_card_index_roundtrip() {
        for suit in 0..4u8 {
            for rank in 2..=14u8 {
                let card = Card::new(rank, suit);
                let idx = card.index();
                assert!(idx < 52);
                let back = Card::from_index(idx);
                assert_eq!(card, back);
            }
        }
    }
}
