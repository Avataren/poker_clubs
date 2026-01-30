use super::{
    constants::{
        DEFAULT_MAX_SEATS, DEFAULT_SHOWDOWN_DELAY_MS, DEFAULT_STREET_DELAY_MS,
        MIN_PLAYERS_TO_START,
    },
    deck::{Card, Deck},
    error::{GameError, GameResult},
    hand::{determine_winners, evaluate_hand, HandRank},
    player::{Player, PlayerAction, PlayerState},
    pot::PotManager,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GamePhase {
    Waiting,    // Waiting for players
    PreFlop,    // Hole cards dealt, pre-flop betting
    Flop,       // 3 community cards, betting
    Turn,       // 4th community card, betting
    River,      // 5th community card, betting
    Showdown,   // Reveal and determine winner
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PokerTable {
    pub table_id: String,
    pub name: String,
    pub small_blind: i64,
    pub big_blind: i64,
    pub players: Vec<Player>,
    pub max_seats: usize,
    pub last_winner_message: Option<String>,
    pub winning_hand: Option<String>,  // Description of the winning hand
    pub dealer_seat: usize,
    pub current_player: usize,
    pub phase: GamePhase,
    pub community_cards: Vec<Card>,
    pub deck: Deck,
    pub pot: PotManager,
    pub current_bet: i64,
    pub min_raise: i64,
    pub last_phase_change_time: Option<u64>,
    pub street_delay_ms: u64,  // Delay between flop/turn/river
    pub showdown_delay_ms: u64, // Delay to show results
}

impl PokerTable {
    pub fn new(table_id: String, name: String, small_blind: i64, big_blind: i64) -> Self {
        Self::with_max_seats(table_id, name, small_blind, big_blind, DEFAULT_MAX_SEATS)
    }

    pub fn with_max_seats(table_id: String, name: String, small_blind: i64, big_blind: i64, max_seats: usize) -> Self {
        Self {
            table_id,
            name,
            small_blind,
            big_blind,
            players: Vec::new(),
            max_seats,
            last_winner_message: None,
            winning_hand: None,
            dealer_seat: 0,
            current_player: 0,
            phase: GamePhase::Waiting,
            community_cards: Vec::new(),
            deck: Deck::new(),
            pot: PotManager::new(),
            current_bet: 0,
            min_raise: big_blind,
            last_phase_change_time: None,
            street_delay_ms: DEFAULT_STREET_DELAY_MS,
            showdown_delay_ms: DEFAULT_SHOWDOWN_DELAY_MS,
        }
    }

    pub fn add_player(&mut self, user_id: String, username: String, buyin: i64) -> GameResult<usize> {
        // Check if player is already at the table and not broke
        if let Some(existing) = self.players.iter().find(|p| p.user_id == user_id) {
            // If player is sitting out with no chips (broke), remove them so they can rebuy
            if existing.state == PlayerState::SittingOut && existing.stack == 0 {
                self.remove_player(&user_id);
            } else {
                return Err(GameError::PlayerAlreadySeated);
            }
        }

        if self.players.len() >= self.max_seats {
            return Err(GameError::TableFull);
        }

        let seat = self.players.len();
        let mut player = Player::new(user_id, username, seat, buyin);

        // If a hand is in progress, make player wait until next hand
        if self.phase != GamePhase::Waiting {
            player.state = PlayerState::WaitingForHand;
            tracing::debug!("Player {} joining mid-hand, setting to WaitingForHand", player.username);
        }

        self.players.push(player);

        // Start game if we have enough players
        if self.players.len() >= MIN_PLAYERS_TO_START && self.phase == GamePhase::Waiting {
            self.start_new_hand();
        }

        Ok(seat)
    }

    pub fn remove_player(&mut self, user_id: &str) {
        self.players.retain(|p| p.user_id != user_id);
        // Recalculate seat numbers
        for (idx, player) in self.players.iter_mut().enumerate() {
            player.seat = idx;
        }
    }
    pub fn take_seat(&mut self, user_id: String, username: String, seat: usize, buyin: i64) -> GameResult<usize> {
        // Check if player is already at the table
        if let Some(existing) = self.players.iter().find(|p| p.user_id == user_id) {
            // If player is sitting out with no chips (broke), remove them so they can rebuy
            if existing.state == PlayerState::SittingOut && existing.stack == 0 {
                self.players.retain(|p| p.user_id != user_id);
            } else {
                return Err(GameError::PlayerAlreadySeated);
            }
        }

        // Validate seat number
        if seat >= self.max_seats {
            return Err(GameError::InvalidSeat { seat, max_seats: self.max_seats });
        }

        // Check if seat is occupied
        if self.players.iter().any(|p| p.seat == seat) {
            return Err(GameError::SeatOccupied { seat });
        }

        let mut player = Player::new(user_id, username, seat, buyin);

        // If a hand is in progress, make player wait until next hand
        if self.phase != GamePhase::Waiting {
            player.state = PlayerState::WaitingForHand;
            tracing::debug!("Player {} joining mid-hand, setting to WaitingForHand", player.username);
        }

        self.players.push(player);

        // Start game if we have enough active players
        if self.active_players_count() >= MIN_PLAYERS_TO_START && self.phase == GamePhase::Waiting {
            self.start_new_hand();
        }

        Ok(seat)
    }

    pub fn top_up(&mut self, user_id: &str, amount: i64) -> GameResult<()> {
        // Find the player
        let player = self.players.iter_mut()
            .find(|p| p.user_id == user_id)
            .ok_or(GameError::PlayerNotAtTable)?;

        // Validate top-up amount
        if amount <= 0 {
            return Err(GameError::InvalidAction { reason: "Top-up amount must be positive".to_string() });
        }

        // Check if player can top up during a hand
        if self.phase != GamePhase::Waiting && player.state != PlayerState::SittingOut {
            return Err(GameError::GameInProgress);
        }

        // Add chips to player's stack
        player.stack += amount;
        
        // If player was sitting out due to being broke, set them to WaitingForHand
        if player.state == PlayerState::SittingOut && player.stack > 0 {
            player.state = PlayerState::WaitingForHand;
            tracing::info!("Player {} topped up ${} and is now waiting for next hand", player.username, amount);
        }

        Ok(())
    }

    pub fn stand_up(&mut self, user_id: &str) -> GameResult<()> {
        let player = self.players.iter_mut()
            .find(|p| p.user_id == user_id)
            .ok_or(GameError::PlayerNotAtTable)?;

        // If player is in an active hand, mark them to stand up after hand concludes
        if self.phase != GamePhase::Waiting && 
           (player.state == PlayerState::Active || player.state == PlayerState::AllIn) {
            player.state = PlayerState::SittingOut;
            tracing::debug!("Player {} will stand up after current hand", player.username);
            Ok(())
        } else {
            // Remove player immediately
            let username = player.username.clone();
            self.players.retain(|p| p.user_id != user_id);
            tracing::debug!("Player {} stood up immediately", username);
            Ok(())
        }
    }

    fn active_players_count(&self) -> usize {
        self.players.iter()
            .filter(|p| p.state == PlayerState::Active || p.state == PlayerState::WaitingForHand)
            .count()
    }

    pub fn start_new_hand(&mut self) {
        self.last_winner_message = None;
        self.winning_hand = None;

        // Reset all players and check for broke players
        for player in &mut self.players {
            player.reset_for_new_hand();
            // reset_for_new_hand already sets broke players (stack=0) to SittingOut
        }
        
        // Count players who can actually play (have chips)
        let playable_count = self.players.iter()
            .filter(|p| p.stack > 0)
            .count();
        
        if playable_count < MIN_PLAYERS_TO_START {
            self.phase = GamePhase::Waiting;
            // Clear community cards when going to waiting
            self.community_cards.clear();
            tracing::info!("Not enough players with chips to start hand, going to Waiting phase");
            return;
        }

        // Clear community cards and reset game state
        self.community_cards.clear();
        self.pot.reset();
        self.deck.reset_and_shuffle();
        self.current_bet = 0;
        self.min_raise = self.big_blind;

        // Move dealer button
        self.dealer_seat = (self.dealer_seat + 1) % self.players.len();

        // Post blinds
        self.post_blinds();

        // Deal hole cards
        self.deal_hole_cards();

        // Set phase and current player
        self.phase = GamePhase::PreFlop;
        self.current_player = self.next_active_player(self.dealer_seat);
    }

    fn post_blinds(&mut self) {
        let num_players = self.players.len();

        // Small blind (next player after dealer)
        let sb_seat = (self.dealer_seat + 1) % num_players;
        let sb_amount = self.players[sb_seat].place_bet(self.small_blind);
        self.pot.add_bet(sb_seat, sb_amount);
        // Posting blind does NOT count as acting - player can still raise

        // Big blind
        let bb_seat = (self.dealer_seat + 2) % num_players;
        let bb_amount = self.players[bb_seat].place_bet(self.big_blind);
        self.pot.add_bet(bb_seat, bb_amount);
        // Posting blind does NOT count as acting - BB has option to raise

        self.current_bet = self.big_blind;

        // First to act is after big blind
        self.current_player = (bb_seat + 1) % num_players;
        
        tracing::info!("Blinds posted: SB=${} at seat {}, BB=${} at seat {}. SB state={:?}, BB state={:?}",
                      sb_amount, sb_seat, bb_amount, bb_seat,
                      self.players[sb_seat].state, self.players[bb_seat].state);
    }

    fn deal_hole_cards(&mut self) {
        for player in &mut self.players {
            if player.can_act() {
                let cards = self.deck.deal_multiple(2);
                player.deal_cards(cards);
            }
        }
    }

    pub fn handle_action(&mut self, user_id: &str, action: PlayerAction) -> GameResult<()> {
        tracing::debug!("handle_action: user_id={}, action={:?}, current_player={}", user_id, action, self.current_player);

        // Verify it's the player's turn
        let current = &self.players[self.current_player];
        if current.user_id != user_id {
            tracing::debug!("Not your turn: current player is {}", current.user_id);
            return Err(GameError::NotYourTurn);
        }

        if !current.can_act() {
            tracing::debug!("Cannot act: state={:?}", current.state);
            return Err(GameError::CannotAct);
        }

        tracing::debug!("Processing action: {:?}", action);
        // Process action
        match action {
            PlayerAction::Fold => {
                self.players[self.current_player].fold();
            }
            PlayerAction::Check => {
                if self.current_bet > self.players[self.current_player].current_bet {
                    return Err(GameError::CannotCheck { current_bet: self.current_bet });
                }
            }
            PlayerAction::Call => {
                let to_call = self.current_bet - self.players[self.current_player].current_bet;
                let actual = self.players[self.current_player].place_bet(to_call);
                self.pot.add_bet(self.current_player, actual);
            }
            PlayerAction::Raise(amount) => {
                let current_bet = self.players[self.current_player].current_bet;
                let raise_to = self.current_bet + amount;

                if amount < self.min_raise {
                    return Err(GameError::RaiseTooSmall { min_raise: self.min_raise, attempted: amount });
                }

                let total_to_bet = raise_to - current_bet;
                let actual = self.players[self.current_player].place_bet(total_to_bet);
                self.pot.add_bet(self.current_player, actual);

                self.current_bet = current_bet + actual;
                self.min_raise = amount;

                // Reset has_acted for all other players since there's a new bet
                for (idx, player) in self.players.iter_mut().enumerate() {
                    if idx != self.current_player {
                        player.has_acted_this_round = false;
                    }
                }
            }
            PlayerAction::AllIn => {
                let stack = self.players[self.current_player].stack;
                let actual = self.players[self.current_player].place_bet(stack);
                self.pot.add_bet(self.current_player, actual);

                let new_total = self.players[self.current_player].current_bet;
                if new_total > self.current_bet {
                    self.min_raise = new_total - self.current_bet;
                    self.current_bet = new_total;

                    // Reset has_acted for all other players since there's a new bet
                    for (idx, player) in self.players.iter_mut().enumerate() {
                        if idx != self.current_player {
                            player.has_acted_this_round = false;
                        }
                    }
                }
            }
        }

        // Mark player as having acted
        self.players[self.current_player].has_acted_this_round = true;

        tracing::info!("Player {} acted, advancing. Current player before: {}", self.players[self.current_player].username, self.current_player);
        
        // Move to next player or next phase
        self.advance_action();
        
        tracing::info!("After advance: current_player={}, state={:?}", self.current_player, 
                       if self.current_player < self.players.len() { 
                           format!("{:?}", self.players[self.current_player].state) 
                       } else { 
                           "INVALID INDEX".to_string() 
                       });

        Ok(())
    }

    fn advance_action(&mut self) {
        // Check if betting round is complete
        if self.is_betting_round_complete() {
            self.advance_phase();
            // Don't auto-advance anymore - delays are handled by check_auto_advance
        } else {
            self.current_player = self.next_active_player(self.current_player);
        }
    }

    fn is_betting_round_complete(&self) -> bool {
        let active_players: Vec<&Player> = self.players.iter()
            .filter(|p| p.can_act())
            .collect();

        if active_players.is_empty() {
            return true;
        }

        // Need at least 2 checks:
        // 1. All active players have acted this round
        // 2. All active players have matched the current bet

        let all_acted = active_players.iter().all(|p| p.has_acted_this_round);
        let all_matched = active_players.iter().all(|p| p.current_bet == self.current_bet);

        // Debug logging
        tracing::debug!(
            "Betting round check: phase={:?}, current_bet={}, active_players={}, all_acted={}, all_matched={}",
            self.phase, self.current_bet, active_players.len(), all_acted, all_matched
        );
        for (i, p) in self.players.iter().enumerate() {
            tracing::debug!(
                "  Player {}: {}, bet={}, acted={}, state={:?}",
                i, p.username, p.current_bet, p.has_acted_this_round, p.state
            );
        }

        all_acted && all_matched
    }

    fn advance_phase(&mut self) {
        // Record when phase changed
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_phase_change_time = Some(now);

        // Reset current bets for new betting round
        for player in &mut self.players {
            player.reset_for_new_round();
        }
        self.pot.end_betting_round();
        self.current_bet = 0;

        match self.phase {
            GamePhase::PreFlop => {
                // Deal flop
                self.community_cards = self.deck.deal_multiple(3);
                self.phase = GamePhase::Flop;
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::Flop => {
                // Deal turn
                self.community_cards.push(self.deck.deal().unwrap());
                self.phase = GamePhase::Turn;
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::Turn => {
                // Deal river
                self.community_cards.push(self.deck.deal().unwrap());
                self.phase = GamePhase::River;
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::River => {
                // Go to showdown
                self.phase = GamePhase::Showdown;
                self.resolve_showdown();
            }
            _ => {}
        }
    }

    fn resolve_showdown(&mut self) {
        // Reset all highlighted cards and winner flags
        for card in &mut self.community_cards {
            card.highlighted = false;
        }
        for player in &mut self.players {
            for card in &mut player.hole_cards {
                card.highlighted = false;
            }
            player.is_winner = false;
        }

        // Evaluate hands for active players
        let mut hands: Vec<(usize, HandRank)> = Vec::new();

        for (idx, player) in self.players.iter().enumerate() {
            if player.is_active_in_hand() {
                let hand_rank = evaluate_hand(&player.hole_cards, &self.community_cards);
                hands.push((idx, hand_rank));
            }
        }

        // Determine winners
        let winner_indices = determine_winners(hands.clone());

        // Get the winning hand description and mark cards as highlighted for all winners
        if let Some(&first_winner_idx) = winner_indices.first() {
            // Find the winning hand_rank (all winners have the same rank)
            if let Some((_, winning_hand_rank)) = hands.iter().find(|(idx, _)| *idx == first_winner_idx) {
                self.winning_hand = Some(winning_hand_rank.description.clone());
                
                // Process each winner
                for &winner_idx in &winner_indices {
                    // Mark player as winner
                    if let Some(winner) = self.players.get_mut(winner_idx) {
                        winner.is_winner = true;
                    }
                    
                    // Get the best cards for this winner
                    if let Some((_, hand_rank)) = hands.iter().find(|(idx, _)| *idx == winner_idx) {
                        let best_cards = &hand_rank.best_cards;
                        
                        tracing::info!("Winner {} best cards: {:?}", winner_idx, best_cards);
                        
                        // Highlight community cards that are in the best hand
                        for community_card in &mut self.community_cards {
                            if best_cards.iter().any(|c| c.rank == community_card.rank && c.suit == community_card.suit) {
                                community_card.highlighted = true;
                                tracing::info!("Highlighted community card: rank={} suit={}", community_card.rank, community_card.suit);
                            }
                        }
                        
                        // Highlight hole cards of the winner that are in the best hand
                        if let Some(winner) = self.players.get_mut(winner_idx) {
                            for hole_card in &mut winner.hole_cards {
                                if best_cards.iter().any(|c| c.rank == hole_card.rank && c.suit == hole_card.suit) {
                                    hole_card.highlighted = true;
                                    tracing::info!("Highlighted hole card for {}: rank={} suit={}", winner.username, hole_card.rank, hole_card.suit);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Award pot
        let payouts = self.pot.award_pots(vec![winner_indices]);

        // Build winner message
        let mut winner_names = Vec::new();
        for (player_idx, amount) in &payouts {
            self.players[*player_idx].add_chips(*amount);
            winner_names.push(format!("{} wins ${}", self.players[*player_idx].username, amount));
        }
        
        self.last_winner_message = Some(winner_names.join(", "));

        // Record showdown time for delay before new hand
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_phase_change_time = Some(now);
        
        tracing::info!("Showdown complete. Winner: {}. Will auto-advance after delay.", winner_names.join(", "));
    }

    fn next_active_player(&self, after: usize) -> usize {
        let mut idx = (after + 1) % self.players.len();
        let start = idx;

        tracing::debug!("next_active_player: after={}, num_players={}", after, self.players.len());
        
        loop {
            tracing::debug!("  Checking idx={}: username={}, can_act={}, state={:?}", 
                           idx, self.players[idx].username, self.players[idx].can_act(), self.players[idx].state);
            if self.players[idx].can_act() {
                tracing::info!("next_active_player: returning idx={} ({})", idx, self.players[idx].username);
                return idx;
            }
            idx = (idx + 1) % self.players.len();
            if idx == start {
                tracing::warn!("next_active_player: No active players found! Returning fallback {}", after);
                break; // No active players found
            }
        }

        after // Fallback
    }

    // Check if enough time has passed to auto-advance to next phase
    pub fn check_auto_advance(&mut self) -> bool {
        // Only auto-advance if no players can act (all all-in or folded)
        if self.players.iter().any(|p| p.can_act()) {
            return false;
        }

        // Check if we're in a phase that needs delay
        let (needs_delay, delay_ms) = match self.phase {
            GamePhase::Flop | GamePhase::Turn | GamePhase::River => (true, self.street_delay_ms),
            GamePhase::Showdown => (true, self.showdown_delay_ms),
            _ => (false, 0),
        };

        if !needs_delay {
            return false;
        }

        // Check if enough time has passed
        if let Some(last_change) = self.last_phase_change_time {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            
            let elapsed = now - last_change;
            
            if elapsed >= delay_ms {
                tracing::info!("Auto-advancing from {:?} after {}ms delay", self.phase, elapsed);
                
                if self.phase == GamePhase::Showdown {
                    // After showdown, start new hand if we have enough players
                    // Check total players (not just active, since they might be in AllIn state)
                    if self.players.len() >= MIN_PLAYERS_TO_START {
                        tracing::info!("Starting new hand after showdown delay");
                        self.start_new_hand();
                    } else {
                        tracing::info!("Not enough players, going to Waiting phase");
                        self.phase = GamePhase::Waiting;
                    }
                } else {
                    // Advance to next street
                    self.advance_phase();
                }
                
                return true;
            }
        }

        false
    }

    pub fn get_public_state(&self, for_user_id: Option<&str>) -> PublicTableState {
        // Get the actual seat number of the current player (not array index)
        // During Waiting phase or invalid states, use the first player's seat if available
        let current_player_seat = if self.phase == GamePhase::Waiting {
            // In waiting phase, no one's turn - use first player's seat as placeholder
            if !self.players.is_empty() {
                self.players[0].seat
            } else {
                0
            }
        } else if self.current_player < self.players.len() {
            self.players[self.current_player].seat
        } else {
            // Fallback: try to find any active player's seat
            self.players.iter()
                .find(|p| p.can_act())
                .map(|p| p.seat)
                .unwrap_or(0)
        };
        
        PublicTableState {
            table_id: self.table_id.clone(),
            name: self.name.clone(),
            phase: self.phase.clone(),
            community_cards: self.community_cards.clone(),
            pot_total: self.pot.total(),
            current_bet: self.current_bet,
            current_player_seat,
            players: self.players.iter().map(|p| {
                PublicPlayerState {
                    user_id: p.user_id.clone(),
                    username: p.username.clone(),
                    seat: p.seat,
                    stack: p.stack,
                    current_bet: p.current_bet,
                    state: p.state.clone(),
                    hole_cards: if Some(p.user_id.as_str()) == for_user_id 
                        || (self.phase == GamePhase::Showdown && p.is_active_in_hand()) {
                        Some(p.hole_cards.clone())
                    } else {
                        None
                    },
                    is_winner: p.is_winner,
                }
            }).collect(),
            max_seats: self.max_seats,
            last_winner_message: self.last_winner_message.clone(),
            winning_hand: self.winning_hand.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicTableState {
    pub table_id: String,
    pub name: String,
    pub phase: GamePhase,
    pub community_cards: Vec<Card>,
    pub pot_total: i64,
    pub current_bet: i64,
    pub current_player_seat: usize,
    pub players: Vec<PublicPlayerState>,
    pub max_seats: usize,
    pub last_winner_message: Option<String>,
    pub winning_hand: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPlayerState {
    pub user_id: String,
    pub username: String,
    pub seat: usize,
    pub stack: i64,
    pub current_bet: i64,
    pub state: PlayerState,
    pub hole_cards: Option<Vec<Card>>, // Only visible to the player themselves
    pub is_winner: bool,
}
