pub mod auth;
pub mod clubs;
pub mod profile;
pub mod tables;
pub mod tournaments;

pub use auth::{router as auth_router, AppState};
pub use clubs::router as clubs_router;
pub use profile::router as profile_router;
pub use tables::{router as tables_router, TableAppState};
pub use tournaments::{router as tournaments_router, TournamentAppState};
