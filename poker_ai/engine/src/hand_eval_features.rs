use crate::card::Card;

/// Preflop hand equity vs. a single random opponent (heads-up equity).
///
/// Covers all 169 canonical hand types (13 pairs + 78 suited + 78 offsuit).
/// Values are from PokerStove/Equilab simulations expressed as win-probability
/// fractions in [0.337, 0.849].  This replaces the old 6-tier coarse system
/// (0.45–0.95) which collapsed 169 hands into only 6 buckets and caused
/// premium preflop hands to map numerically to the same range as strong
/// postflop made hands (flush, full house).
///
/// Using real equity values gives the network 169 distinct, ordered inputs
/// that reflect the true relative strength of every starting hand.
pub fn preflop_strength(c1: Card, c2: Card) -> f32 {
    let high = c1.rank.max(c2.rank);
    let low  = c1.rank.min(c2.rank);
    let suited = c1.suit == c2.suit;

    if high == low {
        // ── Pocket pairs ─────────────────────────────────────────────────
        match high {
            14 => 0.849, // AA
            13 => 0.821, // KK
            12 => 0.796, // QQ
            11 => 0.771, // JJ
            10 => 0.751, // TT
             9 => 0.717, // 99
             8 => 0.690, // 88
             7 => 0.663, // 77
             6 => 0.634, // 66
             5 => 0.603, // 55
             4 => 0.572, // 44
             3 => 0.536, // 33
             _ => 0.503, // 22
        }
    } else if suited {
        // ── Suited non-pairs ─────────────────────────────────────────────
        match (high, low) {
            // Ace-high
            (14, 13) => 0.662, // AKs
            (14, 12) => 0.654, // AQs
            (14, 11) => 0.649, // AJs
            (14, 10) => 0.643, // ATs
            (14,  9) => 0.627, // A9s
            (14,  8) => 0.623, // A8s
            (14,  7) => 0.621, // A7s
            (14,  6) => 0.617, // A6s
            (14,  5) => 0.625, // A5s  (wheel potential)
            (14,  4) => 0.621, // A4s
            (14,  3) => 0.616, // A3s
            (14,  2) => 0.610, // A2s
            // King-high
            (13, 12) => 0.634, // KQs
            (13, 11) => 0.626, // KJs
            (13, 10) => 0.619, // KTs
            (13,  9) => 0.601, // K9s
            (13,  8) => 0.588, // K8s
            (13,  7) => 0.582, // K7s
            (13,  6) => 0.576, // K6s
            (13,  5) => 0.570, // K5s
            (13,  4) => 0.565, // K4s
            (13,  3) => 0.560, // K3s
            (13,  2) => 0.555, // K2s
            // Queen-high
            (12, 11) => 0.603, // QJs
            (12, 10) => 0.595, // QTs
            (12,  9) => 0.579, // Q9s
            (12,  8) => 0.567, // Q8s
            (12,  7) => 0.553, // Q7s
            (12,  6) => 0.548, // Q6s
            (12,  5) => 0.542, // Q5s
            (12,  4) => 0.537, // Q4s
            (12,  3) => 0.531, // Q3s
            (12,  2) => 0.526, // Q2s
            // Jack-high
            (11, 10) => 0.579, // JTs
            (11,  9) => 0.562, // J9s
            (11,  8) => 0.548, // J8s
            (11,  7) => 0.533, // J7s
            (11,  6) => 0.520, // J6s
            (11,  5) => 0.513, // J5s
            (11,  4) => 0.507, // J4s
            (11,  3) => 0.501, // J3s
            (11,  2) => 0.495, // J2s
            // Ten-high
            (10,  9) => 0.548, // T9s
            (10,  8) => 0.533, // T8s
            (10,  7) => 0.518, // T7s
            (10,  6) => 0.503, // T6s
            (10,  5) => 0.488, // T5s
            (10,  4) => 0.482, // T4s
            (10,  3) => 0.476, // T3s
            (10,  2) => 0.469, // T2s
            // Nine-high
            ( 9,  8) => 0.517, // 98s
            ( 9,  7) => 0.502, // 97s
            ( 9,  6) => 0.488, // 96s
            ( 9,  5) => 0.472, // 95s
            ( 9,  4) => 0.464, // 94s
            ( 9,  3) => 0.458, // 93s
            ( 9,  2) => 0.452, // 92s
            // Eight-high
            ( 8,  7) => 0.486, // 87s
            ( 8,  6) => 0.471, // 86s
            ( 8,  5) => 0.455, // 85s
            ( 8,  4) => 0.447, // 84s
            ( 8,  3) => 0.441, // 83s
            ( 8,  2) => 0.434, // 82s
            // Seven-high
            ( 7,  6) => 0.455, // 76s
            ( 7,  5) => 0.440, // 75s
            ( 7,  4) => 0.424, // 74s
            ( 7,  3) => 0.416, // 73s
            ( 7,  2) => 0.410, // 72s
            // Six-high
            ( 6,  5) => 0.436, // 65s
            ( 6,  4) => 0.419, // 64s
            ( 6,  3) => 0.411, // 63s
            ( 6,  2) => 0.404, // 62s
            // Five-high
            ( 5,  4) => 0.420, // 54s
            ( 5,  3) => 0.403, // 53s
            ( 5,  2) => 0.394, // 52s
            // Four-high
            ( 4,  3) => 0.388, // 43s
            ( 4,  2) => 0.380, // 42s
            // Three-high
            ( 3,  2) => 0.365, // 32s
            _ => 0.40,
        }
    } else {
        // ── Offsuit non-pairs ─────────────────────────────────────────────
        match (high, low) {
            // Ace-high
            (14, 13) => 0.645, // AKo
            (14, 12) => 0.637, // AQo
            (14, 11) => 0.629, // AJo
            (14, 10) => 0.621, // ATo
            (14,  9) => 0.604, // A9o
            (14,  8) => 0.599, // A8o
            (14,  7) => 0.594, // A7o
            (14,  6) => 0.590, // A6o
            (14,  5) => 0.596, // A5o  (wheel potential)
            (14,  4) => 0.590, // A4o
            (14,  3) => 0.585, // A3o
            (14,  2) => 0.578, // A2o
            // King-high
            (13, 12) => 0.614, // KQo
            (13, 11) => 0.606, // KJo
            (13, 10) => 0.596, // KTo
            (13,  9) => 0.577, // K9o
            (13,  8) => 0.563, // K8o
            (13,  7) => 0.557, // K7o
            (13,  6) => 0.550, // K6o
            (13,  5) => 0.544, // K5o
            (13,  4) => 0.539, // K4o
            (13,  3) => 0.533, // K3o
            (13,  2) => 0.527, // K2o
            // Queen-high
            (12, 11) => 0.581, // QJo
            (12, 10) => 0.570, // QTo
            (12,  9) => 0.553, // Q9o
            (12,  8) => 0.540, // Q8o
            (12,  7) => 0.525, // Q7o
            (12,  6) => 0.519, // Q6o
            (12,  5) => 0.513, // Q5o
            (12,  4) => 0.507, // Q4o
            (12,  3) => 0.502, // Q3o
            (12,  2) => 0.497, // Q2o
            // Jack-high
            (11, 10) => 0.555, // JTo
            (11,  9) => 0.537, // J9o
            (11,  8) => 0.522, // J8o
            (11,  7) => 0.506, // J7o
            (11,  6) => 0.492, // J6o
            (11,  5) => 0.485, // J5o
            (11,  4) => 0.479, // J4o
            (11,  3) => 0.473, // J3o
            (11,  2) => 0.467, // J2o
            // Ten-high
            (10,  9) => 0.525, // T9o
            (10,  8) => 0.509, // T8o
            (10,  7) => 0.493, // T7o
            (10,  6) => 0.478, // T6o
            (10,  5) => 0.463, // T5o
            (10,  4) => 0.456, // T4o
            (10,  3) => 0.450, // T3o
            (10,  2) => 0.444, // T2o
            // Nine-high
            ( 9,  8) => 0.494, // 98o
            ( 9,  7) => 0.478, // 97o
            ( 9,  6) => 0.463, // 96o
            ( 9,  5) => 0.447, // 95o
            ( 9,  4) => 0.439, // 94o
            ( 9,  3) => 0.433, // 93o
            ( 9,  2) => 0.427, // 92o
            // Eight-high
            ( 8,  7) => 0.463, // 87o
            ( 8,  6) => 0.447, // 86o
            ( 8,  5) => 0.430, // 85o
            ( 8,  4) => 0.422, // 84o
            ( 8,  3) => 0.416, // 83o
            ( 8,  2) => 0.409, // 82o
            // Seven-high
            ( 7,  6) => 0.431, // 76o
            ( 7,  5) => 0.416, // 75o
            ( 7,  4) => 0.399, // 74o
            ( 7,  3) => 0.391, // 73o
            ( 7,  2) => 0.384, // 72o
            // Six-high
            ( 6,  5) => 0.410, // 65o
            ( 6,  4) => 0.392, // 64o
            ( 6,  3) => 0.384, // 63o
            ( 6,  2) => 0.377, // 62o
            // Five-high
            ( 5,  4) => 0.393, // 54o
            ( 5,  3) => 0.376, // 53o
            ( 5,  2) => 0.368, // 52o
            // Four-high
            ( 4,  3) => 0.360, // 43o
            ( 4,  2) => 0.353, // 42o
            // Three-high
            ( 3,  2) => 0.337, // 32o
            _ => 0.38,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;

    fn card(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    // ── Pocket pair ordering ──────────────────────────────────────────────

    #[test]
    fn test_pairs_descending() {
        let aa = preflop_strength(card(14, 0), card(14, 1));
        let kk = preflop_strength(card(13, 0), card(13, 1));
        let qq = preflop_strength(card(12, 0), card(12, 1));
        let jj = preflop_strength(card(11, 0), card(11, 1));
        let tt = preflop_strength(card(10, 0), card(10, 1));
        let twos = preflop_strength(card(2, 0), card(2, 1));

        assert!(aa > kk, "AA > KK");
        assert!(kk > qq, "KK > QQ");
        assert!(qq > jj, "QQ > JJ");
        assert!(jj > tt, "JJ > TT");
        assert!(tt > twos, "TT > 22");
    }

    // ── Suited > offsuit for the same ranks ───────────────────────────────

    #[test]
    fn test_suited_beats_offsuit_same_ranks() {
        // AK
        let aks = preflop_strength(card(14, 0), card(13, 0));
        let ako = preflop_strength(card(14, 0), card(13, 1));
        assert!(aks > ako, "AKs > AKo: {} vs {}", aks, ako);

        // JT
        let jts = preflop_strength(card(11, 2), card(10, 2));
        let jto = preflop_strength(card(11, 2), card(10, 3));
        assert!(jts > jto, "JTs > JTo: {} vs {}", jts, jto);

        // 72
        let s72 = preflop_strength(card(7, 1), card(2, 1));
        let o72 = preflop_strength(card(7, 1), card(2, 2));
        assert!(s72 > o72, "72s > 72o: {} vs {}", s72, o72);
    }

    // ── Premium hands >> trash ─────────────────────────────────────────────

    #[test]
    fn test_aa_much_stronger_than_72o() {
        let aa  = preflop_strength(card(14, 0), card(14, 1));
        let o72 = preflop_strength(card(7, 0), card(2, 1));
        assert!(aa > o72 + 0.40, "AA equity should be >0.40 above 72o: AA={} 72o={}", aa, o72);
    }

    // ── Value range ────────────────────────────────────────────────────────

    #[test]
    fn test_all_values_in_range() {
        let all_hands = [
            // Pairs
            (14, 14, 0), (13, 13, 0), (12, 12, 0), (11, 11, 0),
            (10, 10, 0), (9, 9, 0), (8, 8, 0), (7, 7, 0),
            (6, 6, 0), (5, 5, 0), (4, 4, 0), (3, 3, 0), (2, 2, 0),
            // Suited
            (14, 13, 0), (14, 2, 0), (13, 12, 0), (7, 6, 0), (3, 2, 0),
            // Offsuit
            (14, 13, 1), (14, 2, 1), (7, 2, 1), (3, 2, 1),
        ];
        for (h, l, suit_offset) in all_hands {
            let v = preflop_strength(card(h, 0), card(l, suit_offset));
            assert!(
                v >= 0.30 && v <= 0.90,
                "preflop_strength({},{},suited={}) = {} out of [0.30,0.90]",
                h, l, suit_offset == 0, v
            );
        }
    }

    // ── Card order symmetry ────────────────────────────────────────────────

    #[test]
    fn test_card_order_symmetric() {
        // Passing c1/c2 in either order should give the same result
        let v1 = preflop_strength(card(14, 0), card(13, 1));
        let v2 = preflop_strength(card(13, 1), card(14, 0));
        assert_eq!(v1, v2, "AKo should be symmetric: {} vs {}", v1, v2);

        let v3 = preflop_strength(card(9, 0), card(9, 1));
        let v4 = preflop_strength(card(9, 1), card(9, 0));
        assert_eq!(v3, v4, "99 should be symmetric");
    }

    // ── Specific known values ─────────────────────────────────────────────

    #[test]
    fn test_known_equity_values() {
        // AA
        assert_eq!(preflop_strength(card(14, 0), card(14, 1)), 0.849);
        // 72o (worst hand)
        assert_eq!(preflop_strength(card(7, 0), card(2, 1)), 0.384);
        // 32o (absolute worst by rank)
        assert_eq!(preflop_strength(card(3, 0), card(2, 1)), 0.337);
        // AKs
        assert_eq!(preflop_strength(card(14, 0), card(13, 0)), 0.662);
        // AKo
        assert_eq!(preflop_strength(card(14, 0), card(13, 1)), 0.645);
    }

    // ── A5s wheel bonus ────────────────────────────────────────────────────

    #[test]
    fn test_a5s_wheel_bonus() {
        // A5s (0.625) should beat A6s (0.617) due to wheel potential
        let a5s = preflop_strength(card(14, 0), card(5, 0));
        let a6s = preflop_strength(card(14, 0), card(6, 0));
        assert!(a5s > a6s, "A5s={} should > A6s={} (wheel potential)", a5s, a6s);
    }

    // ── Board texture: empty board ─────────────────────────────────────────

    #[test]
    fn test_board_texture_empty() {
        let t = board_texture(&[]);
        assert_eq!(t, [0.0; 6]);
    }

    // ── Board texture: flush draw ──────────────────────────────────────────

    #[test]
    fn test_board_texture_flush_draw() {
        let board = vec![
            card(2, 0), card(7, 0), card(11, 0), // 3 clubs
        ];
        let t = board_texture(&board);
        assert_eq!(t[0], 0.5, "3 suited cards = flush draw (0.5)");
    }
}
