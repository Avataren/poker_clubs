use crate::game::deck::Card;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlayerState {
    Active,         // Still in the hand
    Folded,         // Folded this hand
    AllIn,          // All chips in the pot
    SittingOut,     // Voluntarily sitting out (won't auto-activate)
    WaitingForHand, // Joined mid-hand, waiting for next hand to start
    Eliminated,     // Tournament only: player has no chips left
    Disconnected,   // WebSocket disconnected, waiting for reconnection (grace period)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub user_id: String,
    pub username: String,
    pub seat: usize,
    pub stack: i64,
    pub hole_cards: Vec<Card>,
    pub current_bet: i64,
    pub state: PlayerState,
    pub total_bet_this_hand: i64, // Track total bet for pot calculation
    pub has_acted_this_round: bool, // Track if player acted this betting round
    pub is_winner: bool,          // Whether this player won the last showdown
    pub last_action: Option<String>, // Last action taken (for display), cleared on new round
    pub pot_won: i64,             // Amount won from pot (for animation)
    pub shown_cards: Vec<bool>,   // Which cards the winner chose to show after fold-win
}

impl Player {
    pub fn new(user_id: String, username: String, seat: usize, stack: i64) -> Self {
        Self {
            user_id,
            username,
            seat,
            stack,
            hole_cards: vec![],
            current_bet: 0,
            state: PlayerState::Active,
            total_bet_this_hand: 0,
            has_acted_this_round: false,
            is_winner: false,
            last_action: None,
            pot_won: 0,
            shown_cards: vec![],
        }
    }

    pub fn deal_cards(&mut self, cards: Vec<Card>) {
        self.hole_cards.extend(cards);
    }

    pub fn place_bet(&mut self, amount: i64) -> i64 {
        let actual_bet = amount.min(self.stack);
        self.stack -= actual_bet;
        self.current_bet += actual_bet;
        self.total_bet_this_hand += actual_bet;

        if self.stack == 0 {
            self.state = PlayerState::AllIn;
        }

        actual_bet
    }

    pub fn fold(&mut self) {
        self.state = PlayerState::Folded;
    }

    pub fn reset_for_new_round(&mut self) {
        self.current_bet = 0;
        self.has_acted_this_round = false;
        self.last_action = None;
    }

    pub fn reset_for_new_hand(&mut self) {
        self.hole_cards.clear();
        self.current_bet = 0;
        self.total_bet_this_hand = 0;
        self.has_acted_this_round = false;
        self.is_winner = false;
        self.last_action = None;
        self.pot_won = 0;
        self.shown_cards.clear();

        // Activate players who have chips and aren't voluntarily sitting out, eliminated, or disconnected
        if self.stack > 0
            && self.state != PlayerState::SittingOut
            && self.state != PlayerState::Eliminated
            && self.state != PlayerState::Disconnected
        {
            self.state = PlayerState::Active;
        }
        // Players with 0 stack will be handled by check_eliminations in tournament mode
        // or set to SittingOut in cash games
        // If already Eliminated, stay Eliminated
        // If SittingOut voluntarily, stay SittingOut
    }

    pub fn add_chips(&mut self, amount: i64) {
        self.stack += amount;
        if self.state == PlayerState::SittingOut && self.stack > 0 {
            self.state = PlayerState::Active;
        }
    }

    pub fn can_act(&self) -> bool {
        matches!(self.state, PlayerState::Active)
    }

    pub fn is_active_in_hand(&self) -> bool {
        matches!(self.state, PlayerState::Active | PlayerState::AllIn)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", content = "amount")]
pub enum PlayerAction {
    Fold,
    Check,
    Call,
    Raise(i64),
    AllIn,
    ShowCards(Vec<usize>),
}
