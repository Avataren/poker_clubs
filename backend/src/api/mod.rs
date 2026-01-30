pub mod auth;
pub mod clubs;
pub mod tables;

pub use auth::{AppState, router as auth_router};
pub use clubs::router as clubs_router;
pub use tables::{router as tables_router, TableAppState};
