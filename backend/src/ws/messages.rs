use crate::game::{PlayerAction, PublicTableState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum ClientMessage {
    // Screen subscriptions
    ViewingClubsList,
    ViewingClub {
        club_id: String,
    },
    LeavingView,

    // Table actions
    JoinTable {
        table_id: String,
        buyin: i64,
    },
    TakeSeat {
        table_id: String,
        seat: usize,
        buyin: i64,
    },
    LeaveTable,
    StandUp,
    TopUp {
        amount: i64,
    },
    PlayerAction {
        action: PlayerAction,
    },
    GetTableState,
    Ping,

    // Bot management
    AddBot {
        table_id: String,
        name: Option<String>,
        strategy: Option<String>,
    },
    RemoveBot {
        table_id: String,
        bot_user_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
#[allow(clippy::large_enum_variant)] // TableState is the primary message, size difference is intentional
pub enum ServerMessage {
    Connected,
    TableState(PublicTableState),
    PlayerJoined {
        username: String,
        seat: usize,
    },
    PlayerLeft {
        username: String,
    },
    ActionRequired {
        your_turn: bool,
    },
    Error {
        message: String,
    },
    Pong,
    ClubUpdate,
    GlobalUpdate,

    // Tournament events
    TournamentStarted {
        tournament_id: String,
        tournament_name: String,
        table_id: Option<String>,
    },
    TournamentBlindLevelIncreased {
        tournament_id: String,
        level: i64,
        small_blind: i64,
        big_blind: i64,
        ante: i64,
    },
    TournamentPlayerEliminated {
        tournament_id: String,
        username: String,
        position: i64,
        prize: i64,
    },
    TournamentFinished {
        tournament_id: String,
        tournament_name: String,
        winners: Vec<TournamentWinner>,
    },
    TournamentCancelled {
        tournament_id: String,
        tournament_name: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentWinner {
    pub username: String,
    pub position: i64,
    pub prize: i64,
}
