// Allow unused imports for public API re-exports during incremental development
#![allow(unused_imports)]

pub mod betting;
pub mod constants;
pub mod deck;
pub mod error;
pub mod format;
pub mod hand;
pub mod player;
pub mod pot;
pub mod table;
pub mod variant;

// Re-export commonly used items
// Note: Some items are not yet integrated into the main code but are exported
// for future use and to document the public API

// Table and game state
pub use table::{GamePhase, PokerTable, PublicTableState, PublicPlayerState};

// Player types
pub use player::{Player, PlayerAction, PlayerState};

// Card and deck types  
pub use deck::Card;

// Pot management
pub use pot::PotManager;

// Variant types
pub use variant::{
    PokerVariant, variant_from_id, available_variants,
    TexasHoldem, OmahaHi, OmahaHiLo, ShortDeckHoldem,
    BettingStructure, StreetConfig, HandRequirements,
};
