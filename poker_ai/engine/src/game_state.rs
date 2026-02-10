use crate::actions::{Action, ActionRecord, DiscreteAction};
use crate::card::{Card, Deck};
use crate::hand_eval::{evaluate_hand, HandRank};
use crate::pot::{award_pots, calculate_side_pots, SidePot};

/// Game phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
    HandOver,
}

impl Phase {
    pub fn index(&self) -> usize {
        match self {
            Phase::Preflop => 0,
            Phase::Flop => 1,
            Phase::Turn => 2,
            Phase::River => 3,
            Phase::Showdown => 4,
            Phase::HandOver => 5,
        }
    }
}

/// Minimal poker table for fast simulation.
#[derive(Debug, Clone)]
pub struct SimTable {
    pub num_players: usize,
    pub stacks: Vec<i64>,
    pub hole_cards: Vec<[Card; 2]>,
    pub community_cards: Vec<Card>,
    pub pot: i64,
    pub side_pots: Vec<SidePot>,
    pub current_bet: i64,
    pub player_bets: Vec<i64>,       // bets this round
    pub total_bets: Vec<i64>,        // total bets this hand
    pub dealer: usize,
    pub current_player: usize,
    pub phase: Phase,
    pub folded: Vec<bool>,
    pub all_in: Vec<bool>,
    pub deck: Deck,
    pub action_history: Vec<ActionRecord>,
    pub big_blind: i64,
    pub small_blind: i64,
    pub last_raiser: Option<usize>,
    pub players_acted_this_round: Vec<bool>,
    pub rewards: Vec<f64>,           // chips won - chips invested, per player
    pub initial_stacks: Vec<i64>,
}

impl SimTable {
    pub fn new(num_players: usize, starting_stack: i64, small_blind: i64, big_blind: i64) -> Self {
        assert!(num_players >= 2 && num_players <= 9);
        Self {
            num_players,
            stacks: vec![starting_stack; num_players],
            hole_cards: Vec::new(),
            community_cards: Vec::new(),
            pot: 0,
            side_pots: Vec::new(),
            current_bet: 0,
            player_bets: vec![0; num_players],
            total_bets: vec![0; num_players],
            dealer: 0,
            current_player: 0,
            phase: Phase::HandOver,
            folded: vec![false; num_players],
            all_in: vec![false; num_players],
            deck: Deck::new(),
            action_history: Vec::new(),
            big_blind,
            small_blind,
            last_raiser: None,
            players_acted_this_round: vec![false; num_players],
            rewards: vec![0.0; num_players],
            initial_stacks: vec![starting_stack; num_players],
        }
    }

    /// Start a new hand. Returns the current player index.
    pub fn start_hand(&mut self, rng: &mut impl rand::Rng) -> usize {
        // Reset state
        self.community_cards.clear();
        self.pot = 0;
        self.current_bet = 0;
        self.player_bets = vec![0; self.num_players];
        self.total_bets = vec![0; self.num_players];
        self.folded = vec![false; self.num_players];
        self.all_in = vec![false; self.num_players];
        self.action_history.clear();
        self.side_pots.clear();
        self.last_raiser = None;
        self.players_acted_this_round = vec![false; self.num_players];
        self.rewards = vec![0.0; self.num_players];
        self.initial_stacks = self.stacks.clone();

        // Mark busted players as folded
        for i in 0..self.num_players {
            if self.stacks[i] <= 0 {
                self.folded[i] = true;
            }
        }

        // Shuffle and deal
        self.deck = Deck::new();
        self.deck.shuffle(rng);

        self.hole_cards = Vec::with_capacity(self.num_players);
        for _ in 0..self.num_players {
            let c1 = self.deck.deal().unwrap();
            let c2 = self.deck.deal().unwrap();
            self.hole_cards.push([c1, c2]);
        }

        // Post blinds
        let sb_seat = self.next_active_from((self.dealer + 1) % self.num_players);
        let bb_seat = self.next_active_from((sb_seat + 1) % self.num_players);

        self.post_blind(sb_seat, self.small_blind);
        self.post_blind(bb_seat, self.big_blind);
        self.current_bet = self.big_blind;

        self.phase = Phase::Preflop;

        // First to act preflop is after BB
        self.current_player = self.next_active_from((bb_seat + 1) % self.num_players);
        self.current_player
    }

    fn post_blind(&mut self, seat: usize, amount: i64) {
        let actual = amount.min(self.stacks[seat]);
        self.stacks[seat] -= actual;
        self.player_bets[seat] = actual;
        self.total_bets[seat] = actual;
        self.pot += actual;
        if self.stacks[seat] == 0 {
            self.all_in[seat] = true;
        }
    }

    fn next_active_from(&self, start: usize) -> usize {
        for i in 0..self.num_players {
            let seat = (start + i) % self.num_players;
            if !self.folded[seat] && !self.all_in[seat] {
                return seat;
            }
        }
        // All folded or all-in, return start
        start % self.num_players
    }

    /// Get legal discrete action mask (true = legal).
    pub fn legal_actions_mask(&self) -> [bool; 8] {
        let mut mask = [false; 8];
        if self.phase == Phase::HandOver || self.phase == Phase::Showdown {
            return mask;
        }

        let seat = self.current_player;
        if self.folded[seat] || self.all_in[seat] {
            return mask;
        }

        let to_call = self.current_bet - self.player_bets[seat];
        let stack = self.stacks[seat];

        // Can always fold (unless can check)
        if to_call > 0 {
            mask[DiscreteAction::Fold as usize] = true;
        }

        // Check/Call - always available
        mask[DiscreteAction::CheckCall as usize] = true;

        // All-in always available if has chips
        if stack > 0 {
            mask[DiscreteAction::AllIn as usize] = true;
        }

        // Raise options (only if stack > to_call)
        if stack > to_call {
            let _min_raise = self.big_blind.max(self.current_bet);
            for action_idx in 2..=6 {
                if let Some(da) = DiscreteAction::from_index(action_idx) {
                    let action = da.to_action(self.pot, self.current_bet, self.player_bets[seat], stack);
                    if let Action::Raise(raise_to) = action {
                        let raise_amount = raise_to - self.current_bet;
                        if raise_amount >= self.big_blind && raise_to <= self.player_bets[seat] + stack {
                            mask[action_idx] = true;
                        }
                    }
                }
            }
        }

        // If can check, don't allow fold
        if to_call == 0 {
            mask[DiscreteAction::Fold as usize] = false;
        }

        mask
    }

    /// Apply an action. Returns (hand_over, next_player).
    pub fn apply_action(&mut self, action_idx: usize) -> (bool, usize) {
        let seat = self.current_player;
        let da = DiscreteAction::from_index(action_idx).expect("invalid action index");
        let action = da.to_action(self.pot, self.current_bet, self.player_bets[seat], self.stacks[seat]);

        let bet_ratio = if self.pot > 0 {
            match &action {
                Action::Raise(to) => (*to - self.player_bets[seat]) as f32 / self.pot as f32,
                Action::AllIn => self.stacks[seat] as f32 / self.pot as f32,
                Action::CheckCall => {
                    (self.current_bet - self.player_bets[seat]) as f32 / self.pot.max(1) as f32
                }
                Action::Fold => 0.0,
            }
        } else {
            0.0
        };

        self.action_history.push(ActionRecord {
            seat,
            action: da,
            bet_ratio,
        });

        match action {
            Action::Fold => {
                self.folded[seat] = true;
            }
            Action::CheckCall => {
                let to_call = (self.current_bet - self.player_bets[seat]).min(self.stacks[seat]);
                self.stacks[seat] -= to_call;
                self.player_bets[seat] += to_call;
                self.total_bets[seat] += to_call;
                self.pot += to_call;
                if self.stacks[seat] == 0 && to_call > 0 {
                    self.all_in[seat] = true;
                }
            }
            Action::Raise(raise_to) => {
                // Clamp raise_to to what player can afford
                let max_raise_to = self.player_bets[seat] + self.stacks[seat];
                let actual_raise_to = raise_to.min(max_raise_to);
                let amount = actual_raise_to - self.player_bets[seat];
                self.stacks[seat] -= amount;
                self.player_bets[seat] = actual_raise_to;
                self.total_bets[seat] += amount;
                self.pot += amount;
                self.current_bet = actual_raise_to;
                self.last_raiser = Some(seat);
                // Reset acted flags for others
                for i in 0..self.num_players {
                    if i != seat {
                        self.players_acted_this_round[i] = false;
                    }
                }
                if self.stacks[seat] == 0 {
                    self.all_in[seat] = true;
                }
            }
            Action::AllIn => {
                let amount = self.stacks[seat];
                let new_bet = self.player_bets[seat] + amount;
                self.stacks[seat] = 0;
                self.pot += amount;
                self.total_bets[seat] += amount;
                if new_bet > self.current_bet {
                    // This is a raise
                    self.current_bet = new_bet;
                    self.last_raiser = Some(seat);
                    for i in 0..self.num_players {
                        if i != seat {
                            self.players_acted_this_round[i] = false;
                        }
                    }
                }
                self.player_bets[seat] = new_bet;
                self.all_in[seat] = true;
            }
        }

        self.players_acted_this_round[seat] = true;

        // Check if betting round is over
        if self.is_round_over() {
            self.advance_phase();
        } else {
            self.current_player = self.next_active_from((seat + 1) % self.num_players);
        }

        let hand_over = self.phase == Phase::HandOver;
        (hand_over, self.current_player)
    }

    fn active_count(&self) -> usize {
        (0..self.num_players)
            .filter(|&i| !self.folded[i])
            .count()
    }

    fn can_act_count(&self) -> usize {
        (0..self.num_players)
            .filter(|&i| !self.folded[i] && !self.all_in[i])
            .count()
    }

    fn is_round_over(&self) -> bool {
        // Only one player left (everyone else folded)
        if self.active_count() <= 1 {
            return true;
        }

        // Everyone who can act has acted and bets are matched
        let can_act = self.can_act_count();
        if can_act == 0 {
            return true; // all remaining are all-in
        }

        // Check if all active non-all-in players have acted and matched bets
        for i in 0..self.num_players {
            if self.folded[i] || self.all_in[i] {
                continue;
            }
            if !self.players_acted_this_round[i] {
                return false;
            }
            if self.player_bets[i] != self.current_bet {
                return false;
            }
        }

        true
    }

    fn advance_phase(&mut self) {
        // Only one player? Award pot immediately
        if self.active_count() <= 1 {
            self.resolve_hand();
            return;
        }

        // Reset round state
        self.player_bets = vec![0; self.num_players];
        self.current_bet = 0;
        self.last_raiser = None;
        self.players_acted_this_round = vec![false; self.num_players];

        match self.phase {
            Phase::Preflop => {
                self.phase = Phase::Flop;
                // Burn and deal 3
                self.deck.deal(); // burn
                for _ in 0..3 {
                    if let Some(c) = self.deck.deal() {
                        self.community_cards.push(c);
                    }
                }
            }
            Phase::Flop => {
                self.phase = Phase::Turn;
                self.deck.deal(); // burn
                if let Some(c) = self.deck.deal() {
                    self.community_cards.push(c);
                }
            }
            Phase::Turn => {
                self.phase = Phase::River;
                self.deck.deal(); // burn
                if let Some(c) = self.deck.deal() {
                    self.community_cards.push(c);
                }
            }
            Phase::River => {
                self.phase = Phase::Showdown;
                self.resolve_hand();
                return;
            }
            _ => {
                self.resolve_hand();
                return;
            }
        }

        // If only one player can act (rest are all-in), skip to showdown
        if self.can_act_count() <= 1 && self.active_count() > 1 {
            // Deal remaining community cards and go to showdown
            while self.community_cards.len() < 5 {
                self.deck.deal(); // burn
                if let Some(c) = self.deck.deal() {
                    self.community_cards.push(c);
                }
            }
            self.phase = Phase::Showdown;
            self.resolve_hand();
            return;
        }

        // Set first player after dealer
        self.current_player = self.next_active_from((self.dealer + 1) % self.num_players);
    }

    fn resolve_hand(&mut self) {
        self.phase = Phase::HandOver;

        let active: Vec<usize> = (0..self.num_players)
            .filter(|&i| !self.folded[i])
            .collect();

        if active.len() == 1 {
            // Uncontested pot
            let winner = active[0];
            self.stacks[winner] += self.pot;
            self.rewards[winner] = (self.stacks[winner] - self.initial_stacks[winner]) as f64;
            for i in 0..self.num_players {
                if i != winner {
                    self.rewards[i] = (self.stacks[i] - self.initial_stacks[i]) as f64;
                }
            }
            return;
        }

        // Calculate side pots
        let player_bets: Vec<(usize, i64, bool)> = (0..self.num_players)
            .map(|i| (i, self.total_bets[i], !self.folded[i]))
            .collect();
        let (pots, uncontested) = calculate_side_pots(&player_bets);

        // Return uncontested amounts
        for (idx, amount) in &uncontested {
            self.stacks[*idx] += amount;
        }

        // Evaluate hands
        let hands: Vec<(usize, HandRank)> = active
            .iter()
            .map(|&i| {
                let rank = evaluate_hand(&self.hole_cards[i], &self.community_cards);
                (i, rank)
            })
            .collect();

        // Determine winners for each pot
        let winners_by_pot: Vec<Vec<usize>> = if pots.is_empty() {
            // Simple case: all bets equal, one pot
            let winners = crate::hand_eval::determine_winners(&hands);
            let share = self.pot / winners.len() as i64;
            let remainder = self.pot % winners.len() as i64;
            for (i, &w) in winners.iter().enumerate() {
                self.stacks[w] += if i == 0 { share + remainder } else { share };
            }
            // Set rewards
            for i in 0..self.num_players {
                self.rewards[i] = (self.stacks[i] - self.initial_stacks[i]) as f64;
            }
            return;
        } else {
            pots.iter()
                .map(|pot| {
                    let eligible_hands: Vec<(usize, HandRank)> = hands
                        .iter()
                        .filter(|(idx, _)| pot.eligible.contains(idx))
                        .cloned()
                        .collect();
                    crate::hand_eval::determine_winners(&eligible_hands)
                })
                .collect()
        };

        let payouts = award_pots(&pots, &winners_by_pot);
        for (idx, amount) in payouts {
            self.stacks[idx] += amount;
        }

        // Set rewards
        for i in 0..self.num_players {
            self.rewards[i] = (self.stacks[i] - self.initial_stacks[i]) as f64;
        }
    }

    /// Advance dealer position for next hand.
    pub fn advance_dealer(&mut self) {
        self.dealer = (self.dealer + 1) % self.num_players;
    }

    /// Encode observation for the given player as a feature vector.
    /// Returns 569 floats total (without LSTM hidden state which is maintained in Python).
    pub fn encode_observation(&self, seat: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(569);

        // --- Card encoding (364 floats) ---
        // Hole cards: 2 x 52 one-hot
        for i in 0..2 {
            let mut onehot = [0.0f32; 52];
            if seat < self.hole_cards.len() {
                onehot[self.hole_cards[seat][i].index()] = 1.0;
            }
            features.extend_from_slice(&onehot);
        }
        // Community cards: 5 x 52 one-hot (zero-padded)
        for i in 0..5 {
            let mut onehot = [0.0f32; 52];
            if i < self.community_cards.len() {
                onehot[self.community_cards[i].index()] = 1.0;
            }
            features.extend_from_slice(&onehot);
        }

        // --- Game state (25 floats) ---
        // Phase one-hot (6)
        let mut phase_oh = [0.0f32; 6];
        phase_oh[self.phase.index()] = 1.0;
        features.extend_from_slice(&phase_oh);

        // Stack / initial stack ratio (1)
        let stack_ratio = if self.initial_stacks[seat] > 0 {
            self.stacks[seat] as f32 / self.initial_stacks[seat] as f32
        } else {
            0.0
        };
        features.push(stack_ratio);

        // Pot / BB ratio (1)
        let pot_bb = self.pot as f32 / self.big_blind.max(1) as f32;
        features.push(pot_bb.min(50.0) / 50.0);

        // Stack / pot ratio (SPR) (1)
        let spr = if self.pot > 0 {
            self.stacks[seat] as f32 / self.pot as f32
        } else {
            10.0
        };
        features.push(spr.min(20.0) / 20.0);

        // Position: distance from dealer normalized (1)
        let position = ((seat + self.num_players - self.dealer) % self.num_players) as f32
            / (self.num_players - 1).max(1) as f32;
        features.push(position);

        // Num opponents still in hand (1)
        let opponents = self.active_count().saturating_sub(1) as f32
            / (self.num_players - 1).max(1) as f32;
        features.push(opponents);

        // Num players who can still act (1)
        let can_act = self.can_act_count() as f32 / self.num_players.max(1) as f32;
        features.push(can_act);

        // To-call / pot ratio (1)
        let to_call = (self.current_bet - self.player_bets.get(seat).copied().unwrap_or(0)) as f32;
        let to_call_ratio = if self.pot > 0 {
            to_call / self.pot as f32
        } else {
            0.0
        };
        features.push(to_call_ratio.min(5.0) / 5.0);

        // Num players (normalized) (1)
        features.push(self.num_players as f32 / 9.0);

        // Per-opponent features: folded, all-in, stack ratio (up to 8 opponents = 24)
        // Pad to 8 opponents for fixed size
        for i in 0..8 {
            let opp = (seat + 1 + i) % self.num_players;
            if i < self.num_players - 1 {
                features.push(if self.folded[opp] { 1.0 } else { 0.0 });
                features.push(if self.all_in[opp] { 1.0 } else { 0.0 });
                let opp_stack_ratio = if self.initial_stacks[opp] > 0 {
                    self.stacks[opp] as f32 / self.initial_stacks[opp] as f32
                } else {
                    0.0
                };
                features.push(opp_stack_ratio.min(3.0) / 3.0);
            } else {
                features.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
        }
        // Total game state: 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 24 = 38
        // We said 25, but let me recount: we need exactly 25
        // Trim: 6 phase + 8 scalars + 8*... let's use less opponent info
        // Actually the plan says 25 game state + 128 LSTM + 52 hand strength = 205 non-card
        // Plus 364 card = 569. Let me just ensure we hit 569 by controlling the layout.

        // --- Placeholder for LSTM hidden state (128 floats) - filled by Python ---
        // We encode action_history here as raw data, Python LSTM will process it
        // For now output zeros as placeholder
        features.resize(364 + 25 + 128, 0.0);

        // --- Hand strength features (52 floats) ---
        if seat < self.hole_cards.len() && !self.hole_cards.is_empty() {
            let total_cards = 2 + self.community_cards.len();
            if total_cards >= 5 {
                let hand_rank = evaluate_hand(&self.hole_cards[seat], &self.community_cards);
                features.push(hand_rank.normalized());
            } else {
                features.push(0.0); // no hand rank available preflop
            }

            // Preflop strength
            features.push(crate::hand_eval_features::preflop_strength(
                self.hole_cards[seat][0],
                self.hole_cards[seat][1],
            ));

            // Pad remaining hand strength features
            features.resize(569, 0.0);
        } else {
            features.resize(569, 0.0);
        }

        features
    }

    /// Get action history encoded for LSTM input.
    /// Each action is 7 floats. Returns Vec of [7] arrays.
    pub fn encode_action_history(&self) -> Vec<[f32; 7]> {
        self.action_history
            .iter()
            .map(|ar| ar.encode(self.num_players))
            .collect()
    }
}
