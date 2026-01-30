//! Poker Variant Trait System
//!
//! This module defines the abstraction for different poker game variants.
//! By implementing the `PokerVariant` trait, new game types can be added
//! without modifying the core game engine.
//!
//! Supported/planned variants:
//! - Texas Hold'em (implemented)
//! - Omaha Hi (planned)
//! - Omaha Hi-Lo (planned)
//! - Short Deck Hold'em (planned)
//! - 5-Card Draw (planned)

// Allow dead code during development - these variants will be integrated incrementally
#![allow(dead_code)]

use super::deck::Card;
use super::hand::HandRank;

/// Betting structure types
#[derive(Debug, Clone, PartialEq)]
pub enum BettingStructure {
    NoLimit,
    PotLimit,
    FixedLimit { small_bet: i64, big_bet: i64 },
}

/// Defines how community cards are dealt across streets
#[derive(Debug, Clone)]
pub struct StreetConfig {
    pub name: &'static str,
    pub cards_to_deal: usize,
}

/// Requirements for making a valid hand
#[derive(Debug, Clone, Default)]
pub struct HandRequirements {
    /// Number of hole cards that MUST be used (e.g., 2 for Omaha)
    pub must_use_hole_cards: Option<usize>,
    /// Number of community cards that MUST be used (e.g., 3 for Omaha)
    pub must_use_community_cards: Option<usize>,
}

/// Core trait that defines a poker game variant
pub trait PokerVariant: Send + Sync + std::fmt::Debug {
    /// The display name of this variant
    fn name(&self) -> &'static str;

    /// Short identifier for serialization/configuration
    fn id(&self) -> &'static str;

    /// Number of hole cards dealt to each player
    fn hole_cards_count(&self) -> usize;

    /// Configuration for each street (flop, turn, river)
    fn streets(&self) -> Vec<StreetConfig>;

    /// The betting structure for this variant
    fn betting_structure(&self) -> BettingStructure;

    /// Requirements for forming a valid hand
    fn hand_requirements(&self) -> HandRequirements {
        HandRequirements::default()
    }

    /// Evaluate the best 5-card hand from hole cards and community cards
    /// This method can be overridden for variants with special hand requirements (e.g., Omaha)
    fn evaluate_hand(&self, hole_cards: &[Card], community_cards: &[Card]) -> HandRank {
        // Default implementation uses standard hand evaluation
        super::hand::evaluate_hand(hole_cards, community_cards)
    }

    /// Returns true if this variant supports hi-lo split pots
    fn is_hi_lo(&self) -> bool {
        false
    }

    /// Minimum number of players required
    fn min_players(&self) -> usize {
        2
    }

    /// Maximum number of players supported
    fn max_players(&self) -> usize {
        9
    }

    /// Clone the variant into a boxed trait object
    fn clone_box(&self) -> Box<dyn PokerVariant>;
}

/// Texas Hold'em - the most popular poker variant
#[derive(Debug, Clone, Default)]
pub struct TexasHoldem;

impl PokerVariant for TexasHoldem {
    fn name(&self) -> &'static str {
        "Texas Hold'em"
    }

    fn id(&self) -> &'static str {
        "holdem"
    }

    fn hole_cards_count(&self) -> usize {
        2
    }

    fn streets(&self) -> Vec<StreetConfig> {
        vec![
            StreetConfig { name: "Flop", cards_to_deal: 3 },
            StreetConfig { name: "Turn", cards_to_deal: 1 },
            StreetConfig { name: "River", cards_to_deal: 1 },
        ]
    }

    fn betting_structure(&self) -> BettingStructure {
        BettingStructure::NoLimit
    }

    fn max_players(&self) -> usize {
        10 // Holdem supports up to 10 players
    }

    fn clone_box(&self) -> Box<dyn PokerVariant> {
        Box::new(self.clone())
    }
}

/// Omaha - four hole cards, must use exactly 2
#[derive(Debug, Clone, Default)]
pub struct OmahaHi;

impl PokerVariant for OmahaHi {
    fn name(&self) -> &'static str {
        "Omaha"
    }

    fn id(&self) -> &'static str {
        "omaha"
    }

    fn hole_cards_count(&self) -> usize {
        4
    }

    fn streets(&self) -> Vec<StreetConfig> {
        vec![
            StreetConfig { name: "Flop", cards_to_deal: 3 },
            StreetConfig { name: "Turn", cards_to_deal: 1 },
            StreetConfig { name: "River", cards_to_deal: 1 },
        ]
    }

    fn betting_structure(&self) -> BettingStructure {
        BettingStructure::PotLimit
    }

    fn hand_requirements(&self) -> HandRequirements {
        HandRequirements {
            must_use_hole_cards: Some(2),
            must_use_community_cards: Some(3),
        }
    }

    fn evaluate_hand(&self, hole_cards: &[Card], community_cards: &[Card]) -> HandRank {
        // Omaha requires exactly 2 hole cards + 3 community cards
        super::hand::evaluate_omaha_hand(hole_cards, community_cards)
            .unwrap_or_else(|| super::hand::evaluate_hand(hole_cards, community_cards))
    }

    fn max_players(&self) -> usize {
        9 // Omaha with 4 cards per player supports fewer players
    }

    fn clone_box(&self) -> Box<dyn PokerVariant> {
        Box::new(self.clone())
    }
}

/// Omaha Hi-Lo - split pot variant
#[derive(Debug, Clone, Default)]
pub struct OmahaHiLo;

impl PokerVariant for OmahaHiLo {
    fn name(&self) -> &'static str {
        "Omaha Hi-Lo"
    }

    fn id(&self) -> &'static str {
        "omaha_hilo"
    }

    fn hole_cards_count(&self) -> usize {
        4
    }

    fn streets(&self) -> Vec<StreetConfig> {
        vec![
            StreetConfig { name: "Flop", cards_to_deal: 3 },
            StreetConfig { name: "Turn", cards_to_deal: 1 },
            StreetConfig { name: "River", cards_to_deal: 1 },
        ]
    }

    fn betting_structure(&self) -> BettingStructure {
        BettingStructure::PotLimit
    }

    fn hand_requirements(&self) -> HandRequirements {
        HandRequirements {
            must_use_hole_cards: Some(2),
            must_use_community_cards: Some(3),
        }
    }

    fn evaluate_hand(&self, hole_cards: &[Card], community_cards: &[Card]) -> HandRank {
        // Omaha Hi-Lo uses same evaluation for hi hand (lo hand evaluation would be separate)
        super::hand::evaluate_omaha_hand(hole_cards, community_cards)
            .unwrap_or_else(|| super::hand::evaluate_hand(hole_cards, community_cards))
    }

    fn is_hi_lo(&self) -> bool {
        true
    }

    fn max_players(&self) -> usize {
        9
    }

    fn clone_box(&self) -> Box<dyn PokerVariant> {
        Box::new(self.clone())
    }
}

/// Short Deck (6+) Hold'em - cards 2-5 removed
#[derive(Debug, Clone, Default)]
pub struct ShortDeckHoldem;

impl PokerVariant for ShortDeckHoldem {
    fn name(&self) -> &'static str {
        "Short Deck Hold'em"
    }

    fn id(&self) -> &'static str {
        "short_deck"
    }

    fn hole_cards_count(&self) -> usize {
        2
    }

    fn streets(&self) -> Vec<StreetConfig> {
        vec![
            StreetConfig { name: "Flop", cards_to_deal: 3 },
            StreetConfig { name: "Turn", cards_to_deal: 1 },
            StreetConfig { name: "River", cards_to_deal: 1 },
        ]
    }

    fn betting_structure(&self) -> BettingStructure {
        BettingStructure::NoLimit
    }

    fn max_players(&self) -> usize {
        9
    }

    fn clone_box(&self) -> Box<dyn PokerVariant> {
        Box::new(self.clone())
    }
}

/// Factory function to create a variant from its ID
pub fn variant_from_id(id: &str) -> Option<Box<dyn PokerVariant>> {
    match id {
        "holdem" => Some(Box::new(TexasHoldem)),
        "omaha" => Some(Box::new(OmahaHi)),
        "omaha_hilo" => Some(Box::new(OmahaHiLo)),
        "short_deck" => Some(Box::new(ShortDeckHoldem)),
        _ => None,
    }
}

/// Get all available variant IDs
pub fn available_variants() -> Vec<&'static str> {
    vec!["holdem", "omaha", "omaha_hilo", "short_deck"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texas_holdem_config() {
        let variant = TexasHoldem;
        assert_eq!(variant.name(), "Texas Hold'em");
        assert_eq!(variant.id(), "holdem");
        assert_eq!(variant.hole_cards_count(), 2);
        assert_eq!(variant.streets().len(), 3);
        assert_eq!(variant.streets()[0].cards_to_deal, 3); // Flop
        assert!(!variant.is_hi_lo());
    }

    #[test]
    fn test_omaha_config() {
        let variant = OmahaHi;
        assert_eq!(variant.name(), "Omaha");
        assert_eq!(variant.hole_cards_count(), 4);
        
        let reqs = variant.hand_requirements();
        assert_eq!(reqs.must_use_hole_cards, Some(2));
        assert_eq!(reqs.must_use_community_cards, Some(3));
    }

    #[test]
    fn test_omaha_hilo_is_split() {
        let variant = OmahaHiLo;
        assert!(variant.is_hi_lo());
    }

    #[test]
    fn test_variant_factory() {
        assert!(variant_from_id("holdem").is_some());
        assert!(variant_from_id("omaha").is_some());
        assert!(variant_from_id("invalid").is_none());
    }

    #[test]
    fn test_available_variants() {
        let variants = available_variants();
        assert!(variants.contains(&"holdem"));
        assert!(variants.contains(&"omaha"));
        assert!(variants.contains(&"omaha_hilo"));
    }

    #[test]
    fn test_omaha_evaluate_hand() {
        use super::super::deck::Card;
        
        let variant = OmahaHi;
        // Hole: AA23 (four cards)
        // Board: KKKQ7 (five community)
        // In Omaha, must use exactly 2 hole + 3 community
        // Best: Full House (AA + KKK)
        let hole_cards = vec![
            Card::new(14, 0), // Ace
            Card::new(14, 1), // Ace
            Card::new(2, 2),  // Two
            Card::new(3, 3),  // Three
        ];
        let community_cards = vec![
            Card::new(13, 0), // King
            Card::new(13, 1), // King
            Card::new(13, 2), // King
            Card::new(12, 3), // Queen
            Card::new(7, 0),  // Seven
        ];

        let rank = variant.evaluate_hand(&hole_cards, &community_cards);
        assert_eq!(rank.description, "Full House");
    }
}
