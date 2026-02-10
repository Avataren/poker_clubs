use crate::card::Card;

/// Quick preflop hand strength (0.0..1.0) using tier lookup.
pub fn preflop_strength(c1: Card, c2: Card) -> f32 {
    let high = c1.rank.max(c2.rank);
    let low = c1.rank.min(c2.rank);
    let suited = c1.suit == c2.suit;
    let pair = c1.rank == c2.rank;

    let tier = if pair {
        match high {
            14 | 13 | 12 => 0,
            11 | 10 => 1,
            9 | 8 => 2,
            7 | 6 => 3,
            5 | 4 => 4,
            _ => 5,
        }
    } else if suited {
        match (high, low) {
            (14, 13) => 0,
            (14, 12) | (14, 11) => 1,
            (14, 10) | (13, 12) | (13, 11) | (12, 11) => 2,
            (14, _) | (13, 10) | (12, 10) | (11, 10) | (10, 9) => 3,
            (13, lo) if lo >= 7 => 4,
            (_, _) if high - low <= 2 && high >= 7 => 4,
            _ => 5 + (14 - high) as u8 / 3,
        }
    } else {
        match (high, low) {
            (14, 13) => 1,
            (14, 12) => 2,
            (14, 11) | (14, 10) => 3,
            (13, 12) | (13, 11) | (12, 11) => 4,
            (13, 10) | (12, 10) | (11, 10) => 5,
            _ => 6 + (14 - high) as u8 / 2,
        }
    };

    (0.95 - tier as f32 * 0.10).clamp(0.05, 0.95)
}

/// Board texture features (flush draws, straight draws, pairing).
pub fn board_texture(community: &[Card]) -> [f32; 6] {
    if community.is_empty() {
        return [0.0; 6];
    }

    let mut suit_counts = [0u8; 4];
    let mut rank_counts = [0u8; 15]; // index by rank (2-14)
    for c in community {
        suit_counts[c.suit as usize] += 1;
        rank_counts[c.rank as usize] += 1;
    }

    let max_suit = *suit_counts.iter().max().unwrap_or(&0);
    let flush_draw = if max_suit >= 4 {
        1.0
    } else if max_suit >= 3 {
        0.5
    } else {
        0.0
    };

    let paired = rank_counts.iter().filter(|&&c| c >= 2).count() as f32;
    let trips = rank_counts.iter().filter(|&&c| c >= 3).count() as f32;

    // Straight potential: count consecutive ranks
    let mut max_consecutive = 0u8;
    let mut current_run = 0u8;
    for rank in 2..=14 {
        if rank_counts[rank] > 0 {
            current_run += 1;
            max_consecutive = max_consecutive.max(current_run);
        } else {
            current_run = 0;
        }
    }
    // Ace can wrap: check A-2-3-4-5
    if rank_counts[14] > 0 {
        let mut low_run = 1u8;
        for rank in 2..=5 {
            if rank_counts[rank] > 0 {
                low_run += 1;
            } else {
                break;
            }
        }
        max_consecutive = max_consecutive.max(low_run);
    }

    let straight_draw = if max_consecutive >= 5 {
        1.0
    } else if max_consecutive >= 4 {
        0.7
    } else if max_consecutive >= 3 {
        0.3
    } else {
        0.0
    };

    let high_card = community.iter().map(|c| c.rank).max().unwrap_or(0) as f32 / 14.0;
    let num_cards = community.len() as f32 / 5.0;

    [
        flush_draw,
        straight_draw,
        paired / 5.0,
        trips / 5.0,
        high_card,
        num_cards,
    ]
}
