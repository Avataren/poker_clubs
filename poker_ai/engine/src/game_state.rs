use crate::actions::{Action, ActionRecord, DiscreteAction};
use crate::card::{Card, Deck};
use crate::hand_eval::{evaluate_hand, HandRank};
use crate::pot::{award_pots, calculate_side_pots, SidePot};

/// Per-opponent running statistics using exponential moving average.
/// EMA adapts to shifting opponent policy (~50 hand effective window).
const EMA_ALPHA: f64 = 0.02;

/// Number of floats in the encoded per-opponent stats vector.
pub const STATS_PER_OPPONENT: usize = 15;

#[derive(Debug, Clone)]
pub struct PlayerStats {
    // --- EMA running stats (persist across hands) ---
    pub vpip: f64,
    pub pfr: f64,
    pub aggression: f64,          // overall raise/(raise+call)
    pub fold_to_bet: f64,
    pub hands_played: f64,
    // Street-specific aggression
    pub flop_aggression: f64,
    pub turn_aggression: f64,
    pub river_aggression: f64,
    // Showdown stats
    pub wtsd: f64,                // went to showdown % (when saw flop)
    pub wsd: f64,                 // won at showdown %
    // C-bet frequency
    pub cbet: f64,
    // Bet sizing tendencies
    pub avg_bet_size: f64,        // avg bet/raise as fraction of pot
    pub preflop_raise_size: f64,  // avg preflop raise size / pot
    // Position-aware VPIP
    pub ep_vpip: f64,             // VPIP from early position
    pub lp_vpip: f64,             // VPIP from late position

    // --- Per-hand accumulators (reset each hand) ---
    hand_vpip: bool,
    hand_pfr: bool,
    hand_raises: u32,
    hand_calls: u32,
    hand_folds_to_bet: u32,
    hand_faced_bets: u32,
    // Street-specific raise/call counts
    hand_flop_raises: u32,
    hand_flop_actions: u32,       // raises + calls + checks on flop
    hand_turn_raises: u32,
    hand_turn_actions: u32,
    hand_river_raises: u32,
    hand_river_actions: u32,
    // Showdown tracking
    hand_saw_flop: bool,
    // C-bet tracking
    hand_cbet_opportunity: bool,  // raised preflop and saw flop
    hand_cbet_taken: bool,        // bet/raised on flop after preflop raise
    hand_acted_on_flop: bool,     // already acted on flop (only first action counts for cbet)
    // Bet sizing accumulators
    hand_bet_size_sum: f64,       // sum of bet/raise sizes as pot fractions
    hand_bet_count: u32,
    hand_preflop_raise_size_sum: f64,
    hand_preflop_raise_count: u32,
    // Position tracking
    hand_is_early_position: bool,
    hand_is_late_position: bool,
}

impl PlayerStats {
    pub fn new() -> Self {
        Self {
            vpip: 0.5, pfr: 0.5, aggression: 0.5, fold_to_bet: 0.5,
            hands_played: 0.0,
            flop_aggression: 0.5, turn_aggression: 0.5, river_aggression: 0.5,
            wtsd: 0.5, wsd: 0.5,
            cbet: 0.5,
            avg_bet_size: 0.5, preflop_raise_size: 0.5,
            ep_vpip: 0.5, lp_vpip: 0.5,
            hand_vpip: false, hand_pfr: false,
            hand_raises: 0, hand_calls: 0,
            hand_folds_to_bet: 0, hand_faced_bets: 0,
            hand_flop_raises: 0, hand_flop_actions: 0,
            hand_turn_raises: 0, hand_turn_actions: 0,
            hand_river_raises: 0, hand_river_actions: 0,
            hand_saw_flop: false,
            hand_cbet_opportunity: false, hand_cbet_taken: false,
            hand_acted_on_flop: false,
            hand_bet_size_sum: 0.0, hand_bet_count: 0,
            hand_preflop_raise_size_sum: 0.0, hand_preflop_raise_count: 0,
            hand_is_early_position: false, hand_is_late_position: false,
        }
    }

    pub fn start_hand(&mut self) {
        self.hand_vpip = false;
        self.hand_pfr = false;
        self.hand_raises = 0;
        self.hand_calls = 0;
        self.hand_folds_to_bet = 0;
        self.hand_faced_bets = 0;
        self.hand_flop_raises = 0;
        self.hand_flop_actions = 0;
        self.hand_turn_raises = 0;
        self.hand_turn_actions = 0;
        self.hand_river_raises = 0;
        self.hand_river_actions = 0;
        self.hand_saw_flop = false;
        self.hand_cbet_opportunity = false;
        self.hand_cbet_taken = false;
        self.hand_acted_on_flop = false;
        self.hand_bet_size_sum = 0.0;
        self.hand_bet_count = 0;
        self.hand_preflop_raise_size_sum = 0.0;
        self.hand_preflop_raise_count = 0;
        self.hand_is_early_position = false;
        self.hand_is_late_position = false;
    }

    /// Set position category for this hand (called during start_hand setup).
    /// `seat_offset`: seats from dealer (1=SB, 2=BB, 3..=EP, etc.)
    /// `num_players`: total players at table.
    pub fn set_position(&mut self, seat_offset: usize, num_players: usize) {
        // Early position: first third of seats after blinds (or UTG in HU/3-handed)
        // Late position: dealer (BTN) and cutoff
        if num_players <= 3 {
            // HU/3-max: dealer=LP, others=EP
            self.hand_is_late_position = seat_offset == 0 || seat_offset == num_players;
            self.hand_is_early_position = !self.hand_is_late_position;
        } else {
            // 4+ players: BTN(offset 0)=LP, CO(offset n-1 from dealer... let's think)
            // seat_offset = (seat - dealer) % num_players
            // 0 = dealer/BTN, 1 = SB, 2 = BB, 3 = UTG, ...
            // LP = BTN (0) and CO (num_players-1)
            // EP = UTG positions (3, 4 for 6-max)
            self.hand_is_late_position = seat_offset == 0 || seat_offset == num_players - 1;
            self.hand_is_early_position = seat_offset >= 3 && seat_offset <= num_players / 2 + 1;
        }
    }

    /// Notify that this player survived to see the flop.
    pub fn mark_saw_flop(&mut self) {
        self.hand_saw_flop = true;
        if self.hand_pfr {
            self.hand_cbet_opportunity = true;
        }
    }

    pub fn end_hand(&mut self) {
        self.hands_played += 1.0;
        let a = EMA_ALPHA;

        // Core stats
        self.vpip = self.vpip * (1.0 - a) + if self.hand_vpip { a } else { 0.0 };
        self.pfr = self.pfr * (1.0 - a) + if self.hand_pfr { a } else { 0.0 };
        let total_rc = (self.hand_raises + self.hand_calls) as f64;
        if total_rc > 0.0 {
            let hand_agg = self.hand_raises as f64 / total_rc;
            self.aggression = self.aggression * (1.0 - a) + hand_agg * a;
        }
        if self.hand_faced_bets > 0 {
            let hand_ftb = self.hand_folds_to_bet as f64 / self.hand_faced_bets as f64;
            self.fold_to_bet = self.fold_to_bet * (1.0 - a) + hand_ftb * a;
        }

        // Street-specific aggression (only update if player acted on that street)
        if self.hand_flop_actions > 0 {
            let agg = self.hand_flop_raises as f64 / self.hand_flop_actions as f64;
            self.flop_aggression = self.flop_aggression * (1.0 - a) + agg * a;
        }
        if self.hand_turn_actions > 0 {
            let agg = self.hand_turn_raises as f64 / self.hand_turn_actions as f64;
            self.turn_aggression = self.turn_aggression * (1.0 - a) + agg * a;
        }
        if self.hand_river_actions > 0 {
            let agg = self.hand_river_raises as f64 / self.hand_river_actions as f64;
            self.river_aggression = self.river_aggression * (1.0 - a) + agg * a;
        }

        // C-bet: only update when there was an opportunity
        if self.hand_cbet_opportunity {
            let cbet_val = if self.hand_cbet_taken { 1.0 } else { 0.0 };
            self.cbet = self.cbet * (1.0 - a) + cbet_val * a;
        }

        // Bet sizing
        if self.hand_bet_count > 0 {
            let avg = self.hand_bet_size_sum / self.hand_bet_count as f64;
            self.avg_bet_size = self.avg_bet_size * (1.0 - a) + avg * a;
        }
        if self.hand_preflop_raise_count > 0 {
            let avg = self.hand_preflop_raise_size_sum / self.hand_preflop_raise_count as f64;
            self.preflop_raise_size = self.preflop_raise_size * (1.0 - a) + avg * a;
        }

        // Position-aware VPIP (only update for the position category played)
        if self.hand_is_early_position {
            self.ep_vpip = self.ep_vpip * (1.0 - a) + if self.hand_vpip { a } else { 0.0 };
        }
        if self.hand_is_late_position {
            self.lp_vpip = self.lp_vpip * (1.0 - a) + if self.hand_vpip { a } else { 0.0 };
        }
    }

    /// Record showdown result. Called from resolve_hand().
    pub fn record_showdown(&mut self, won: bool) {
        let a = EMA_ALPHA;
        // WTSD: updated when player saw flop (regardless of showdown)
        // — but we only call this when they ARE at showdown, so wtsd always gets 1.0
        // The "didn't reach showdown" case is handled in end_hand_no_showdown()
        if self.hand_saw_flop {
            self.wtsd = self.wtsd * (1.0 - a) + a; // went to showdown = 1.0
        }
        // WSD: won at showdown
        self.wsd = self.wsd * (1.0 - a) + if won { a } else { 0.0 };
    }

    /// Record that player saw flop but did NOT reach showdown (folded post-flop).
    pub fn record_no_showdown(&mut self) {
        if self.hand_saw_flop {
            let a = EMA_ALPHA;
            self.wtsd = self.wtsd * (1.0 - a); // went to showdown = 0.0
        }
    }

    pub fn record_action(&mut self, action: &Action, phase: Phase, facing_bet: bool, bet_ratio: f64) {
        let is_preflop = phase == Phase::Preflop;

        match action {
            Action::Fold => {
                if facing_bet {
                    self.hand_folds_to_bet += 1;
                    self.hand_faced_bets += 1;
                }
                // Count as non-raise action on the current street
                match phase {
                    Phase::Flop => { self.hand_flop_actions += 1; }
                    Phase::Turn => { self.hand_turn_actions += 1; }
                    Phase::River => { self.hand_river_actions += 1; }
                    _ => {}
                }
            }
            Action::CheckCall => {
                if facing_bet {
                    self.hand_calls += 1;
                    self.hand_faced_bets += 1;
                    if is_preflop { self.hand_vpip = true; }
                }
                match phase {
                    Phase::Flop => { self.hand_flop_actions += 1; }
                    Phase::Turn => { self.hand_turn_actions += 1; }
                    Phase::River => { self.hand_river_actions += 1; }
                    _ => {}
                }
            }
            Action::Raise(_) | Action::AllIn => {
                self.hand_raises += 1;
                if facing_bet { self.hand_faced_bets += 1; }
                if is_preflop {
                    self.hand_vpip = true;
                    self.hand_pfr = true;
                    // Track preflop raise sizing
                    if bet_ratio > 0.0 {
                        self.hand_preflop_raise_size_sum += bet_ratio;
                        self.hand_preflop_raise_count += 1;
                    }
                }
                // Street-specific raise tracking
                match phase {
                    Phase::Flop => {
                        self.hand_flop_raises += 1;
                        self.hand_flop_actions += 1;
                        // C-bet: first aggressive action on flop after preflop raise
                        if self.hand_cbet_opportunity && !self.hand_acted_on_flop {
                            self.hand_cbet_taken = true;
                        }
                    }
                    Phase::Turn => {
                        self.hand_turn_raises += 1;
                        self.hand_turn_actions += 1;
                    }
                    Phase::River => {
                        self.hand_river_raises += 1;
                        self.hand_river_actions += 1;
                    }
                    _ => {}
                }
                // Track bet sizing (all streets)
                if bet_ratio > 0.0 {
                    self.hand_bet_size_sum += bet_ratio;
                    self.hand_bet_count += 1;
                }
            }
        }

        // Track first flop action for c-bet detection
        if phase == Phase::Flop && !self.hand_acted_on_flop {
            self.hand_acted_on_flop = true;
        }
    }

    /// Sample size indicator (saturates at 100 hands).
    pub fn sample_size(&self) -> f32 {
        (self.hands_played as f32 / 100.0).min(1.0)
    }

    /// Encode as 15 floats for observation vector.
    pub fn encode(&self) -> [f32; STATS_PER_OPPONENT] {
        [
            self.vpip as f32,
            self.pfr as f32,
            self.aggression as f32,
            self.fold_to_bet as f32,
            self.sample_size(),
            self.flop_aggression as f32,
            self.turn_aggression as f32,
            self.river_aggression as f32,
            self.wtsd as f32,
            self.wsd as f32,
            self.cbet as f32,
            self.avg_bet_size as f32,
            self.preflop_raise_size as f32,
            self.ep_vpip as f32,
            self.lp_vpip as f32,
        ]
    }
}

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
    pub starting_stack: i64,
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
    pub street_action_count: usize,
    pub street_raise_count: usize,
    pub player_stats: Vec<PlayerStats>,
}

impl SimTable {
    pub fn new(num_players: usize, starting_stack: i64, small_blind: i64, big_blind: i64) -> Self {
        assert!(num_players >= 2 && num_players <= 9);
        Self {
            num_players,
            starting_stack,
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
            street_action_count: 0,
            street_raise_count: 0,
            player_stats: (0..num_players).map(|_| PlayerStats::new()).collect(),
        }
    }

    /// Rebuy busted players to the configured starting stack.
    pub fn rebuy_busted_players(&mut self) {
        for i in 0..self.num_players {
            if self.stacks[i] <= 0 {
                self.stacks[i] = self.starting_stack;
            }
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
        self.street_action_count = 0;
        self.street_raise_count = 0;

        // Reset per-hand stat flags and set position
        for i in 0..self.num_players {
            self.player_stats[i].start_hand();
            let seat_offset = (i + self.num_players - self.dealer) % self.num_players;
            self.player_stats[i].set_position(seat_offset, self.num_players);
        }

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
    pub fn legal_actions_mask(&self) -> [bool; 9] {
        let mut mask = [false; 9];
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
            for action_idx in 2..=7 {
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
        self.street_action_count += 1;

        // Record stats for cross-hand opponent modeling
        let facing_bet = self.current_bet > self.player_bets[seat];
        self.player_stats[seat].record_action(&action, self.phase, facing_bet, bet_ratio as f64);

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
                self.street_raise_count += 1;
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
                    self.street_raise_count += 1;
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

    fn count_street_actions(&self) -> usize {
        self.street_action_count
    }

    fn count_street_raises(&self) -> usize {
        self.street_raise_count
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
        self.street_action_count = 0;
        self.street_raise_count = 0;

        match self.phase {
            Phase::Preflop => {
                self.phase = Phase::Flop;
                // Mark non-folded players as having seen the flop
                for i in 0..self.num_players {
                    if !self.folded[i] {
                        self.player_stats[i].mark_saw_flop();
                    }
                }
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

        let is_showdown = active.len() > 1;

        if active.len() == 1 {
            // Uncontested pot — no showdown
            let winner = active[0];
            self.stacks[winner] += self.pot;

            // Record no-showdown for folded players who saw flop
            for i in 0..self.num_players {
                if self.folded[i] {
                    self.player_stats[i].record_no_showdown();
                }
                // Winner also didn't go to showdown (won by fold)
                if i == winner {
                    self.player_stats[i].record_no_showdown();
                }
            }

            // Commit per-hand stats
            for s in &mut self.player_stats {
                s.end_hand();
            }

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

        // Ensure all community cards are dealt before evaluating
        while self.community_cards.len() < 5 {
            self.deck.deal(); // burn
            if let Some(c) = self.deck.deal() {
                self.community_cards.push(c);
            }
        }

        // Evaluate hands
        let hands: Vec<(usize, HandRank)> = active
            .iter()
            .map(|&i| {
                let rank = evaluate_hand(&self.hole_cards[i], &self.community_cards);
                (i, rank)
            })
            .collect();

        // Collect all winners across all pots for showdown stats
        let mut all_winners: Vec<usize> = Vec::new();

        // Determine winners for each pot
        let winners_by_pot: Vec<Vec<usize>> = if pots.is_empty() {
            // Simple case: all bets equal, one pot
            let winners = crate::hand_eval::determine_winners(&hands);
            if !winners.is_empty() {
                let share = self.pot / winners.len() as i64;
                let remainder = self.pot % winners.len() as i64;
                for (i, &w) in winners.iter().enumerate() {
                    self.stacks[w] += if i == 0 { share + remainder } else { share };
                }
                all_winners.extend_from_slice(&winners);
            } else if let Some(&first_active) = active.first() {
                self.stacks[first_active] += self.pot;
                all_winners.push(first_active);
            }

            // Record showdown stats for all active players
            if is_showdown {
                for &seat in &active {
                    self.player_stats[seat].record_showdown(all_winners.contains(&seat));
                }
            }
            // Record no-showdown for folded players
            for i in 0..self.num_players {
                if self.folded[i] {
                    self.player_stats[i].record_no_showdown();
                }
            }

            // Commit per-hand stats
            for s in &mut self.player_stats {
                s.end_hand();
            }

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
                    let w = crate::hand_eval::determine_winners(&eligible_hands);
                    all_winners.extend_from_slice(&w);
                    w
                })
                .collect()
        };

        let payouts = award_pots(&pots, &winners_by_pot);
        for (idx, amount) in payouts {
            self.stacks[idx] += amount;
        }

        // Record showdown stats for all active players
        if is_showdown {
            for &seat in &active {
                self.player_stats[seat].record_showdown(all_winners.contains(&seat));
            }
        }
        // Record no-showdown for folded players
        for i in 0..self.num_players {
            if self.folded[i] {
                self.player_stats[i].record_no_showdown();
            }
        }

        // Commit per-hand stats for all players
        for s in &mut self.player_stats {
            s.end_hand();
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
    /// Returns 710 floats total:
    ///   364 (cards) + 166 (game state) + 128 (history placeholder) + 52 (hand strength)
    pub fn encode_observation(&self, seat: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(710);

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

        // --- Game state (35 floats) ---
        let to_call_i64 = self.current_bet - self.player_bets.get(seat).copied().unwrap_or(0);
        let to_call = to_call_i64.max(0) as f32;
        let pot_f = self.pot.max(0) as f32;
        let stack = self.stacks[seat].max(0) as f32;

        // Phase one-hot (6)
        let mut phase_oh = [0.0f32; 6];
        phase_oh[self.phase.index()] = 1.0;
        features.extend_from_slice(&phase_oh);

        // Stack / initial stack ratio (1)
        let stack_ratio = if self.initial_stacks[seat] > 0 {
            stack / self.initial_stacks[seat] as f32
        } else {
            0.0
        };
        features.push(stack_ratio);

        // Pot / BB ratio (1)
        let pot_bb = pot_f / self.big_blind.max(1) as f32;
        features.push(pot_bb.min(50.0) / 50.0);

        // Stack / pot ratio (SPR) (1)
        let spr = if self.pot > 0 {
            stack / pot_f
        } else {
            10.0
        };
        features.push(spr.min(20.0) / 20.0);

        // Position: distance from dealer normalized (1)
        let position = ((seat + self.num_players - self.dealer) % self.num_players) as f32
            / (self.num_players - 1).max(1) as f32;
        features.push(position);

        // Num opponents still in hand (1)
        let active = self.active_count();
        let opponents = active.saturating_sub(1) as f32
            / (self.num_players - 1).max(1) as f32;
        features.push(opponents);

        // Num players who can still act (1)
        let can_act = self.can_act_count() as f32 / self.num_players.max(1) as f32;
        features.push(can_act);

        // To-call / pot ratio (1)
        let to_call_ratio = if self.pot > 0 {
            to_call / pot_f
        } else {
            0.0
        };
        features.push(to_call_ratio.min(5.0) / 5.0);

        // Num players (normalized) (1)
        features.push(self.num_players as f32 / 9.0);

        // --- NEW features (10 floats) ---

        // Pot odds: to_call / (pot + to_call) (1)
        let pot_odds = if pot_f + to_call > 0.0 {
            to_call / (pot_f + to_call)
        } else {
            0.0
        };
        features.push(pot_odds);

        // Effective stack / pot: min(hero_stack, max_opp_stack) / pot (1)
        let max_opp_stack = (0..self.num_players)
            .filter(|&i| i != seat && !self.folded[i])
            .map(|i| self.stacks[i].max(0) as f32)
            .fold(0.0f32, f32::max);
        let eff_stack = stack.min(max_opp_stack);
        let eff_stack_pot = if pot_f > 0.0 {
            eff_stack / pot_f
        } else {
            10.0
        };
        features.push(eff_stack_pot.min(20.0) / 20.0);

        // Street action count: num actions this street / 10 (1)
        let street_action_count = self.count_street_actions();
        features.push((street_action_count as f32 / 10.0).min(1.0));

        // Total action count: total actions this hand / 30 (1)
        features.push((self.action_history.len() as f32 / 30.0).min(1.0));

        // Num raises this street / 4 (1)
        let raises_this_street = self.count_street_raises();
        features.push((raises_this_street as f32 / 4.0).min(1.0));

        // Last aggressor is hero (1)
        let last_agg_hero = if self.last_raiser == Some(seat) { 1.0 } else { 0.0 };
        features.push(last_agg_hero);

        // Hero's current bet / pot (1)
        let hero_bet = self.player_bets.get(seat).copied().unwrap_or(0).max(0) as f32;
        let hero_bet_pot = if pot_f > 0.0 { hero_bet / pot_f } else { 0.0 };
        features.push(hero_bet_pot.min(5.0) / 5.0);

        // Hero invested / starting stack (1)
        let total_invested = self.total_bets.get(seat).copied().unwrap_or(0).max(0) as f32;
        let hero_invested = if self.initial_stacks[seat] > 0 {
            total_invested / self.initial_stacks[seat] as f32
        } else {
            0.0
        };
        features.push(hero_invested.min(1.0));

        // Per-opponent features: folded, all-in, stack ratio (up to 8 opponents = 24)
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

        // Per-opponent running stats (15 per opponent, up to 8 = 120)
        for i in 0..8 {
            let opp = (seat + 1 + i) % self.num_players;
            if i < self.num_players - 1 {
                let stats = &self.player_stats[opp];
                features.extend_from_slice(&stats.encode());
            } else {
                features.extend_from_slice(&[0.0f32; STATS_PER_OPPONENT]);
            }
        }
        // Game state: 6 phase + 8 scalars + 8 new + 24 opponents + 120 opp_stats = 166
        debug_assert_eq!(features.len(), 364 + 166,
            "game state mismatch: got {} floats, expected 166", features.len() - 364);

        // --- Placeholder for history hidden state (128 floats) - filled by Python ---
        features.resize(364 + 166 + 128, 0.0);

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
            features.resize(710, 0.0);
        } else {
            features.resize(710, 0.0);
        }

        features
    }

    /// Get action history encoded for history MLP input.
    /// Each action is 11 floats. Returns Vec of [11] arrays.
    pub fn encode_action_history(&self) -> Vec<[f32; 11]> {
        self.action_history
            .iter()
            .map(|ar| ar.encode(self.num_players))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::SimTable;

    #[test]
    fn test_rebuy_busted_players() {
        let mut table = SimTable::new(2, 10000, 50, 100);
        table.stacks[0] = 0;
        table.stacks[1] = 25000;

        table.rebuy_busted_players();

        assert_eq!(table.stacks[0], 10000);
        assert_eq!(table.stacks[1], 25000);
    }
}
