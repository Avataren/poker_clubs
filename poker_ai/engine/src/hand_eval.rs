use crate::card::Card;
use rs_poker::core::{Hand, Rank as RsRank, Rankable};

/// Numeric hand rank: higher is better.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandRank {
    pub rank_value: i32, // 0=high card .. 8=straight flush
    pub sub_rank: u32,   // rs_poker sub-rank for ordering within category
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

impl HandRank {
    pub fn from_hand(hand: &Hand) -> Self {
        let rs_rank = hand.rank();
        let (rank_value, sub_rank) = match &rs_rank {
            RsRank::HighCard(v) => (0, *v),
            RsRank::OnePair(v) => (1, *v),
            RsRank::TwoPair(v) => (2, *v),
            RsRank::ThreeOfAKind(v) => (3, *v),
            RsRank::Straight(v) => (4, *v),
            RsRank::Flush(v) => (5, *v),
            RsRank::FullHouse(v) => (6, *v),
            RsRank::FourOfAKind(v) => (7, *v),
            RsRank::StraightFlush(v) => (8, *v),
        };
        Self {
            rank_value,
            sub_rank,
        }
    }

    /// Normalized rank 0.0..1.0 for feature encoding.
    pub fn normalized(&self) -> f32 {
        // rank_value 0-8, sub_rank is u32
        // Use rank_value as primary, sub_rank as fractional
        let base = self.rank_value as f32 / 8.0;
        let frac = (self.sub_rank as f32).min(100000.0) / 100000.0 / 8.0;
        (base + frac).min(1.0)
    }
}

/// Generate all k-combinations from a slice.
pub fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }
    let mut result = Vec::new();
    let first = &items[0];
    let rest = &items[1..];
    for mut combo in combinations(rest, k - 1) {
        combo.insert(0, first.clone());
        result.push(combo);
    }
    result.extend(combinations(rest, k));
    result
}

/// Evaluate best 5-card hand from hole cards + community cards.
pub fn evaluate_hand(hole_cards: &[Card], community_cards: &[Card]) -> HandRank {
    let mut all_cards = Vec::with_capacity(hole_cards.len() + community_cards.len());
    all_cards.extend_from_slice(hole_cards);
    all_cards.extend_from_slice(community_cards);

    let combos = combinations(&all_cards, 5);
    let best_hand = combos
        .into_iter()
        .map(|five| {
            let rs_cards: Vec<rs_poker::core::Card> = five.iter().map(|c| c.to_rs_poker()).collect();
            Hand::new_with_cards(rs_cards)
        })
        .max_by_key(|hand| hand.rank())
        .expect("should have at least one 5-card combination");
    HandRank::from_hand(&best_hand)
}

/// Determine winner indices from (player_idx, HandRank) pairs.
pub fn determine_winners(hands: &[(usize, HandRank)]) -> Vec<usize> {
    if hands.is_empty() {
        return vec![];
    }
    let best = hands.iter().map(|(_, r)| r).max().unwrap();
    hands
        .iter()
        .filter(|(_, r)| r == best)
        .map(|(idx, _)| *idx)
        .collect()
}

/// Monte Carlo hand strength estimation. Returns win probability 0.0..1.0.
pub fn estimate_hand_strength(
    hole_cards: &[Card],
    community_cards: &[Card],
    num_opponents: usize,
    iterations: usize,
    rng: &mut impl rand::Rng,
) -> f64 {
    use rand::seq::SliceRandom;

    if hole_cards.is_empty() || num_opponents == 0 {
        return 0.5;
    }

    let mut known: Vec<Card> = Vec::new();
    known.extend_from_slice(hole_cards);
    known.extend_from_slice(community_cards);

    let mut deck: Vec<Card> = Vec::with_capacity(52);
    for suit in 0..4u8 {
        for rank in 2..=14u8 {
            let c = Card::new(rank, suit);
            if !known.iter().any(|k| k.rank == c.rank && k.suit == c.suit) {
                deck.push(c);
            }
        }
    }

    let community_remaining = 5 - community_cards.len();
    let cards_needed = community_remaining + num_opponents * 2;
    if deck.len() < cards_needed {
        return 0.5;
    }

    let mut wins = 0.0;
    for _ in 0..iterations {
        let mut shuffled = deck.clone();
        shuffled.shuffle(rng);

        let mut full_community: Vec<Card> = community_cards.to_vec();
        for i in 0..community_remaining {
            full_community.push(shuffled[i]);
        }

        let our_rank = evaluate_hand(hole_cards, &full_community);
        let mut we_win = true;
        let mut tie_count = 0;
        for opp in 0..num_opponents {
            let start = community_remaining + opp * 2;
            let opp_cards = &shuffled[start..start + 2];
            let opp_rank = evaluate_hand(opp_cards, &full_community);
            if opp_rank > our_rank {
                we_win = false;
                break;
            } else if opp_rank == our_rank {
                tie_count += 1;
            }
        }
        if we_win {
            wins += if tie_count > 0 {
                1.0 / (tie_count + 1) as f64
            } else {
                1.0
            };
        }
    }

    wins / iterations as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_royal_flush() {
        let hole = vec![Card::new(14, 3), Card::new(13, 3)];
        let community = vec![Card::new(12, 3), Card::new(11, 3), Card::new(10, 3)];
        let rank = evaluate_hand(&hole, &community);
        assert_eq!(rank.rank_value, 8); // Straight flush
    }

    #[test]
    fn test_pair() {
        let hole = vec![Card::new(14, 3), Card::new(14, 2)];
        let community = vec![Card::new(2, 1), Card::new(5, 0), Card::new(9, 3)];
        let rank = evaluate_hand(&hole, &community);
        assert_eq!(rank.rank_value, 1);
    }

    #[test]
    fn test_determine_winners_tie() {
        let hands = vec![
            (0, HandRank { rank_value: 3, sub_rank: 100 }),
            (1, HandRank { rank_value: 3, sub_rank: 100 }),
        ];
        let winners = determine_winners(&hands);
        assert_eq!(winners.len(), 2);
    }
}
