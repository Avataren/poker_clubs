use crate::game::deck::Card;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlayerState {
    Active,         // Still in the hand
    Folded,         // Folded this hand
    AllIn,          // All chips in the pot
    SittingOut,     // Voluntarily sitting out (won't auto-activate)
    WaitingForHand, // Joined mid-hand, waiting for next hand to start
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
    pub is_winner: bool, // Whether this player won the last showdown
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
        }
    }

    pub fn deal_cards(&mut self, cards: Vec<Card>) {
        self.hole_cards = cards;
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
    }

    pub fn reset_for_new_hand(&mut self) {
        self.hole_cards.clear();
        self.current_bet = 0;
        self.total_bet_this_hand = 0;
        self.has_acted_this_round = false;
        self.is_winner = false;

        // Activate players who have chips and aren't voluntarily sitting out
        if self.stack > 0 && self.state != PlayerState::SittingOut {
            self.state = PlayerState::Active;
        } else if self.stack == 0 {
            self.state = PlayerState::SittingOut;
        }
        // If SittingOut, stay SittingOut (voluntary sit-out)
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
}
