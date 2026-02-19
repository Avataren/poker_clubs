//! In-memory hand history log accumulated during a hand, flushed to DB at hand end.

use super::GamePhase;
use crate::game::deck::Card;
use serde::{Deserialize, Serialize};

/// A single action recorded during a hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandAction {
    pub street: String,
    pub seq: i32,
    pub seat: usize,
    pub player_name: String,
    pub action_type: String,
    pub amount: i64,
}

/// Per-player data captured at hand start and updated at hand end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandPlayerRecord {
    pub seat: usize,
    pub user_id: String,
    pub username: String,
    pub hole_cards: Vec<Card>,
    pub starting_stack: i64,
    pub final_stack: i64,
    pub is_winner: bool,
    pub pot_won: i64,
    pub winning_hand_desc: Option<String>,
    pub showed_cards: bool,
    pub folded: bool,
}

/// Accumulated hand data, built up during play and flushed to DB at hand end.
#[derive(Debug, Clone, Default)]
pub struct HandHistoryLog {
    pub active: bool,
    pub table_id: String,
    pub tournament_id: Option<String>,
    pub hand_number: i64,
    pub small_blind: i64,
    pub big_blind: i64,
    pub ante: i64,
    pub dealer_seat: usize,
    pub community_cards: Vec<Card>,
    pub pot_total: i64,
    pub actions: Vec<HandAction>,
    pub players: Vec<HandPlayerRecord>,
    action_seq: i32,
}

impl HandHistoryLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start recording a new hand.
    pub fn begin_hand(
        &mut self,
        table_id: &str,
        tournament_id: Option<&str>,
        hand_number: i64,
        small_blind: i64,
        big_blind: i64,
        ante: i64,
        dealer_seat: usize,
        players: &[(usize, String, String, i64)], // (seat, user_id, username, stack)
    ) {
        self.active = true;
        self.table_id = table_id.to_string();
        self.tournament_id = tournament_id.map(|s| s.to_string());
        self.hand_number = hand_number;
        self.small_blind = small_blind;
        self.big_blind = big_blind;
        self.ante = ante;
        self.dealer_seat = dealer_seat;
        self.community_cards.clear();
        self.pot_total = 0;
        self.actions.clear();
        self.players.clear();
        self.action_seq = 0;

        for (seat, user_id, username, stack) in players {
            self.players.push(HandPlayerRecord {
                seat: *seat,
                user_id: user_id.clone(),
                username: username.clone(),
                hole_cards: Vec::new(),
                starting_stack: *stack,
                final_stack: *stack,
                is_winner: false,
                pot_won: 0,
                winning_hand_desc: None,
                showed_cards: false,
                folded: false,
            });
        }
    }

    /// Record hole cards for a player (call after dealing).
    pub fn set_hole_cards(&mut self, seat: usize, cards: Vec<Card>) {
        if let Some(p) = self.players.iter_mut().find(|p| p.seat == seat) {
            p.hole_cards = cards;
        }
    }

    /// Record a player action.
    pub fn record_action(
        &mut self,
        phase: &GamePhase,
        seat: usize,
        player_name: &str,
        action_type: &str,
        amount: i64,
    ) {
        if !self.active {
            return;
        }
        let street = match phase {
            GamePhase::PreFlop => "preflop",
            GamePhase::Flop => "flop",
            GamePhase::Turn => "turn",
            GamePhase::River => "river",
            _ => "preflop",
        };
        self.actions.push(HandAction {
            street: street.to_string(),
            seq: self.action_seq,
            seat,
            player_name: player_name.to_string(),
            action_type: action_type.to_string(),
            amount,
        });
        self.action_seq += 1;
    }

    /// Update community cards (call when new cards are dealt).
    pub fn set_community_cards(&mut self, cards: &[Card]) {
        self.community_cards = cards.to_vec();
    }

    /// Mark a player as folded.
    pub fn mark_folded(&mut self, seat: usize) {
        if let Some(p) = self.players.iter_mut().find(|p| p.seat == seat) {
            p.folded = true;
        }
    }

    /// Mark a player as having shown cards (showdown or voluntary show).
    pub fn mark_showed_cards(&mut self, seat: usize) {
        if let Some(p) = self.players.iter_mut().find(|p| p.seat == seat) {
            p.showed_cards = true;
        }
    }

    /// Finalize hand results after showdown/fold-win.
    pub fn finalize(
        &mut self,
        pot_total: i64,
        community_cards: &[Card],
        player_results: &[(usize, i64, i64, bool, Option<String>)], // (seat, final_stack, pot_won, is_winner, hand_desc)
    ) {
        self.pot_total = pot_total;
        self.community_cards = community_cards.to_vec();
        for (seat, final_stack, pot_won, is_winner, hand_desc) in player_results {
            if let Some(p) = self.players.iter_mut().find(|p| p.seat == *seat) {
                p.final_stack = *final_stack;
                p.pot_won = *pot_won;
                p.is_winner = *is_winner;
                p.winning_hand_desc = hand_desc.clone();
                // Players who went to showdown (didn't fold) showed cards
                if !p.folded {
                    p.showed_cards = true;
                }
            }
        }
        self.active = false;
    }

    /// Reset for next hand.
    pub fn clear(&mut self) {
        self.active = false;
        self.actions.clear();
        self.players.clear();
        self.community_cards.clear();
        self.action_seq = 0;
    }
}
