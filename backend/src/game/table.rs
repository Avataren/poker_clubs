use super::{
    constants::{
        DEFAULT_MAX_SEATS, DEFAULT_SHOWDOWN_DELAY_MS, DEFAULT_STREET_DELAY_MS,
        MIN_PLAYERS_TO_START,
    },
    deck::{Card, Deck},
    error::{GameError, GameResult},
    format::{CashGame, GameFormat},
    hand::{determine_winners, evaluate_hand, HandRank},
    player::{Player, PlayerAction, PlayerState},
    pot::PotManager,
    variant::{PokerVariant, TexasHoldem},
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

#[derive(Debug, Serialize, Deserialize)]
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
    #[serde(skip, default = "default_variant")]
    variant: Box<dyn PokerVariant>,
    #[serde(skip, default = "default_format")]
    format: Box<dyn GameFormat>,
}

fn default_variant() -> Box<dyn PokerVariant> {
    Box::new(TexasHoldem)
}

fn default_format() -> Box<dyn GameFormat> {
    Box::new(CashGame::new(50, 100, DEFAULT_MAX_SEATS))
}

impl Clone for PokerTable {
    fn clone(&self) -> Self {
        Self {
            table_id: self.table_id.clone(),
            name: self.name.clone(),
            small_blind: self.small_blind,
            big_blind: self.big_blind,
            players: self.players.clone(),
            max_seats: self.max_seats,
            last_winner_message: self.last_winner_message.clone(),
            winning_hand: self.winning_hand.clone(),
            dealer_seat: self.dealer_seat,
            current_player: self.current_player,
            phase: self.phase.clone(),
            community_cards: self.community_cards.clone(),
            deck: self.deck.clone(),
            pot: self.pot.clone(),
            current_bet: self.current_bet,
            min_raise: self.min_raise,
            last_phase_change_time: self.last_phase_change_time,
            street_delay_ms: self.street_delay_ms,
            showdown_delay_ms: self.showdown_delay_ms,
            variant: self.variant.clone_box(),
            format: self.format.clone_box(),
        }
    }
}

impl PokerTable {
    pub fn new(table_id: String, name: String, small_blind: i64, big_blind: i64) -> Self {
        Self::with_max_seats(table_id, name, small_blind, big_blind, DEFAULT_MAX_SEATS)
    }

    pub fn with_max_seats(table_id: String, name: String, small_blind: i64, big_blind: i64, max_seats: usize) -> Self {
        Self::with_variant(table_id, name, small_blind, big_blind, max_seats, Box::new(TexasHoldem))
    }

    /// Create a table with a specific poker variant
    pub fn with_variant(
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        max_seats: usize,
        variant: Box<dyn PokerVariant>,
    ) -> Self {
        let format = Box::new(CashGame::new(small_blind, big_blind, max_seats));
        Self::with_variant_and_format(table_id, name, small_blind, big_blind, max_seats, variant, format)
    }

    /// Create a table with a specific variant and game format
    pub fn with_variant_and_format(
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        max_seats: usize,
        variant: Box<dyn PokerVariant>,
        format: Box<dyn GameFormat>,
    ) -> Self {
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
            variant,
            format,
        }
    }

    /// Get the variant ID of this table
    #[allow(dead_code)]
    pub fn variant_id(&self) -> &'static str {
        self.variant.id()
    }

    /// Get the variant name of this table
    #[allow(dead_code)]
    pub fn variant_name(&self) -> &'static str {
        self.variant.name()
    }

    /// Get the format ID of this table
    #[allow(dead_code)]
    pub fn format_id(&self) -> &'static str {
        self.format.format_id()
    }

    /// Get the format name of this table
    #[allow(dead_code)]
    pub fn format_name(&self) -> &str {
        self.format.name()
    }

    /// Check if the format allows players to cash out
    #[allow(dead_code)]
    pub fn can_cash_out(&self) -> bool {
        self.format.can_cash_out()
    }

    /// Check if the format allows top-ups
    #[allow(dead_code)]
    pub fn can_top_up(&self) -> bool {
        self.format.can_top_up()
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

        // If we're in Waiting phase, check if we can now start a new hand
        if self.phase == GamePhase::Waiting {
            let playable_count = self.players.iter()
                .filter(|p| p.stack > 0)
                .count();
            
            if playable_count >= MIN_PLAYERS_TO_START {
                tracing::info!("Enough players with chips after top-up, starting new hand");
                self.start_new_hand();
            }
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

        // Move dealer button to next eligible player (skip sitting out/broke players)
        self.dealer_seat = self.next_eligible_player_for_button(self.dealer_seat);

        // Post blinds
        self.post_blinds();

        // Deal hole cards
        self.deal_hole_cards();

        // Set phase and current player
        self.phase = GamePhase::PreFlop;
        self.current_player = self.next_active_player(self.dealer_seat);
    }

    fn post_blinds(&mut self) {
        // Small blind is the next eligible player after dealer
        let sb_seat = self.next_eligible_player_for_button(self.dealer_seat);
        let sb_amount = self.players[sb_seat].place_bet(self.small_blind);
        self.pot.add_bet(sb_seat, sb_amount);
        // Posting blind does NOT count as acting - player can still raise

        // Big blind is the next eligible player after small blind
        let bb_seat = self.next_eligible_player_for_button(sb_seat);
        let bb_amount = self.players[bb_seat].place_bet(self.big_blind);
        self.pot.add_bet(bb_seat, bb_amount);
        // Posting blind does NOT count as acting - BB has option to raise

        self.current_bet = self.big_blind;

        // First to act is after big blind
        self.current_player = self.next_active_player(bb_seat);
        
        tracing::info!("Blinds posted: SB=${} at seat {}, BB=${} at seat {}. SB state={:?}, BB state={:?}",
                      sb_amount, sb_seat, bb_amount, bb_seat,
                      self.players[sb_seat].state, self.players[bb_seat].state);
    }

    fn deal_hole_cards(&mut self) {
        let hole_cards_count = self.variant.hole_cards_count();
        
        // Calculate where dealing starts: small blind (first player after dealer)
        let sb_seat = self.next_eligible_player_for_button(self.dealer_seat);
        
        // Deal one card at a time to each player, just like real poker
        // Start from small blind and go around the table for each round
        for round in 0..hole_cards_count {
            let mut current_seat = sb_seat;
            for _ in 0..self.players.len() {
                if self.players[current_seat].can_act() {
                    if let Some(card) = self.deck.deal() {
                        self.players[current_seat].deal_cards(vec![card]);
                    }
                }
                current_seat = (current_seat + 1) % self.players.len();
            }
            tracing::debug!("Dealt card round {} of {}, starting from seat {}", round + 1, hole_cards_count, sb_seat);
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

        // Record last action for display
        self.players[self.current_player].last_action = Some(match &action {
            PlayerAction::Fold => "Fold".to_string(),
            PlayerAction::Check => "Check".to_string(),
            PlayerAction::Call => {
                let bet = self.players[self.current_player].current_bet;
                format!("Call ${}", bet)
            }
            PlayerAction::Raise(amt) => format!("Raise ${}", amt),
            PlayerAction::AllIn => {
                let bet = self.players[self.current_player].current_bet;
                format!("All In ${}", bet)
            }
        });

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

        // If only one player is active in hand, they win immediately (no showdown)
        let active_in_hand: Vec<usize> = self.players.iter().enumerate()
            .filter(|(_, p)| p.is_active_in_hand())
            .map(|(idx, _)| idx)
            .collect();

        if active_in_hand.len() == 1 {
            let winner_idx = active_in_hand[0];
            let total = self.pot.total();
            self.players[winner_idx].add_chips(total);
            self.players[winner_idx].is_winner = true;
            self.players[winner_idx].pot_won = total;
            self.last_winner_message = Some(format!(
                "{} wins ${}",
                self.players[winner_idx].username, total
            ));
            // Pot reset happens in start_new_hand(), not here
            self.phase = GamePhase::Showdown;
            tracing::info!(
                "Hand over: {} wins ${} (all others folded)",
                self.players[winner_idx].username, total
            );
            return;
        }

        match self.phase {
            GamePhase::PreFlop => {
                // Burn a card before dealing the flop
                self.deck.deal();
                // Deal flop
                self.community_cards = self.deck.deal_multiple(3);
                self.phase = GamePhase::Flop;
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::Flop => {
                // Burn a card before dealing the turn
                self.deck.deal();
                // Deal turn
                self.community_cards.push(self.deck.deal().unwrap());
                self.phase = GamePhase::Turn;
                self.current_player = self.next_active_player(self.dealer_seat);
            }
            GamePhase::Turn => {
                // Burn a card before dealing the river
                self.deck.deal();
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
                // Use variant-specific hand evaluation (e.g., Omaha requires using exactly 2 hole cards)
                let hand_rank = self.variant.evaluate_hand(&player.hole_cards, &self.community_cards);
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
                
                // Process each winner for card highlighting
                for &winner_idx in &winner_indices {
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

        // Calculate side pots based on each player's total contribution
        let player_bets: Vec<(usize, i64, bool)> = self.players.iter().enumerate()
            .filter(|(_, p)| p.total_bet_this_hand > 0)
            .map(|(idx, p)| (idx, p.total_bet_this_hand, p.is_active_in_hand()))
            .collect();
        self.pot.calculate_side_pots(&player_bets);

        // Determine winners for each pot based on eligible players
        let mut winners_by_pot = Vec::new();
        for pot in &self.pot.pots {
            let eligible_hands: Vec<(usize, HandRank)> = hands.iter()
                .filter(|(idx, _)| pot.eligible_players.contains(idx))
                .cloned()
                .collect();
            let pot_winners = determine_winners(eligible_hands);
            winners_by_pot.push(pot_winners);
        }

        // Award pots
        let payouts = self.pot.award_pots(winners_by_pot);

        // Mark all payout recipients as winners and build message
        let mut winner_names = Vec::new();
        for (player_idx, amount) in &payouts {
            self.players[*player_idx].add_chips(*amount);
            self.players[*player_idx].is_winner = true;
            self.players[*player_idx].pot_won = *amount;
            winner_names.push(format!("{} wins ${}", self.players[*player_idx].username, amount));
        }

        // DON'T reset pot here - keep it visible for animation
        // It will be reset when start_new_hand() is called
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

    /// Find the next player eligible for dealer/blind positions
    /// (must have chips and not be sitting out voluntarily)
    fn next_eligible_player_for_button(&self, after: usize) -> usize {
        let mut idx = (after + 1) % self.players.len();
        let start = idx;

        tracing::debug!("next_eligible_player_for_button: after={}, num_players={}", after, self.players.len());

        loop {
            let player = &self.players[idx];
            // Player is eligible if they have chips and aren't voluntarily sitting out
            if player.stack > 0 && player.state != PlayerState::SittingOut {
                tracing::info!("next_eligible_player_for_button: returning idx={} ({})", idx, player.username);
                return idx;
            }
            tracing::debug!("  Skipping idx={}: username={}, stack={}, state={:?}",
                           idx, player.username, player.stack, player.state);
            idx = (idx + 1) % self.players.len();
            if idx == start {
                tracing::warn!("next_eligible_player_for_button: No eligible players found! Returning fallback {}", after);
                break; // No eligible players found
            }
        }

        after // Fallback
    }

    // Check if enough time has passed to auto-advance to next phase
    pub fn check_auto_advance(&mut self) -> bool {
        // During Showdown, always allow auto-advance (start new hand after delay).
        // For other phases, auto-advance only if fewer than 2 players can act
        // (everyone all-in or folded, no meaningful betting possible).
        if self.phase != GamePhase::Showdown {
            let can_act_count = self.players.iter().filter(|p| p.can_act()).count();
            if can_act_count >= 2 {
                return false;
            }
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
            variant_id: self.variant.id().to_string(),
            variant_name: self.variant.name().to_string(),
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
                    hole_cards: if Some(p.user_id.as_str()) == for_user_id {
                        // Show own cards face-up
                        Some(p.hole_cards.clone())
                    } else if self.phase == GamePhase::Showdown && p.is_active_in_hand() {
                        // During showdown, show all active players' cards face-up
                        Some(p.hole_cards.clone())
                    } else if p.is_active_in_hand() && !p.hole_cards.is_empty() {
                        // For other players still in the hand, send placeholder face-down cards
                        // WITHOUT revealing the actual rank/suit (security: prevent cheating via network inspection)
                        let num_cards = p.hole_cards.len();
                        Some(vec![
                            Card { rank: 0, suit: 0, highlighted: false, face_up: false };
                            num_cards
                        ])
                    } else {
                        // No cards for folded/sitting out players
                        None
                    },
                    is_winner: p.is_winner,
                    last_action: p.last_action.clone(),
                    pot_won: p.pot_won,
                }
            }).collect(),
            max_seats: self.max_seats,
            last_winner_message: self.last_winner_message.clone(),
            winning_hand: self.winning_hand.clone(),
            format_id: self.format.format_id().to_string(),
            format_name: self.format.name().to_string(),
            can_cash_out: self.format.can_cash_out(),
            can_top_up: self.format.can_top_up(),
            dealer_seat: if self.phase != GamePhase::Waiting && !self.players.is_empty() {
                Some(self.players[self.dealer_seat].seat)
            } else {
                None
            },
            small_blind_seat: if self.phase != GamePhase::Waiting && self.players.len() >= 2 {
                let sb_idx = self.next_eligible_player_for_button(self.dealer_seat);
                Some(self.players[sb_idx].seat)
            } else {
                None
            },
            big_blind_seat: if self.phase != GamePhase::Waiting && self.players.len() >= 2 {
                let sb_idx = self.next_eligible_player_for_button(self.dealer_seat);
                let bb_idx = self.next_eligible_player_for_button(sb_idx);
                Some(self.players[bb_idx].seat)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicTableState {
    pub table_id: String,
    pub name: String,
    pub variant_id: String,
    pub variant_name: String,
    pub format_id: String,
    pub format_name: String,
    pub can_cash_out: bool,
    pub can_top_up: bool,
    pub phase: GamePhase,
    pub community_cards: Vec<Card>,
    pub pot_total: i64,
    pub current_bet: i64,
    pub current_player_seat: usize,
    pub players: Vec<PublicPlayerState>,
    pub max_seats: usize,
    pub last_winner_message: Option<String>,
    pub winning_hand: Option<String>,
    pub dealer_seat: Option<usize>,
    pub small_blind_seat: Option<usize>,
    pub big_blind_seat: Option<usize>,
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
    pub last_action: Option<String>,
    pub pot_won: i64, // Amount won from pot (for animation)
}

impl PokerTable {
    // ==================== Tournament-Specific Methods ====================

    /// Update blinds and minimum raise (called when tournament blind level advances)
    pub fn update_blinds(&mut self, small_blind: i64, big_blind: i64) {
        self.small_blind = small_blind;
        self.big_blind = big_blind;
        self.min_raise = big_blind;
        tracing::info!(
            "Table {} blinds updated to {}/{}",
            self.table_id,
            small_blind,
            big_blind
        );
    }

    /// Apply antes from all active players (called at start of hand in tournament mode)
    pub fn apply_antes(&mut self, ante: i64) -> GameResult<()> {
        if ante == 0 {
            return Ok(());
        }

        for (seat, player) in self.players.iter_mut().enumerate() {
            if player.can_act() || player.state == PlayerState::WaitingForHand {
                let actual_ante = player.place_bet(ante);
                self.pot.add_bet(seat, actual_ante);
                tracing::debug!(
                    "Player {} posted ante: ${} (requested: ${})",
                    player.username,
                    actual_ante,
                    ante
                );
            }
        }

        Ok(())
    }

    /// Check for eliminated players (stack = 0 in tournament mode)
    /// Returns list of user_ids who are eliminated
    pub fn check_eliminations(&mut self) -> Vec<String> {
        if !self.format.eliminates_players() {
            return vec![];
        }

        let mut eliminated = vec![];
        for player in &mut self.players {
            if player.stack == 0 
                && player.state != PlayerState::Eliminated 
                && player.state != PlayerState::SittingOut
            {
                let user_id = player.user_id.clone();
                let username = player.username.clone();
                player.state = PlayerState::Eliminated;
                eliminated.push(user_id);
                tracing::info!("Player {} eliminated from tournament", username);
            }
        }
        eliminated
    }

    /// Check if tournament is finished (1 or fewer players remaining)
    pub fn tournament_finished(&self) -> bool {
        if !self.format.eliminates_players() {
            return false;
        }

        let active_count = self.players.iter()
            .filter(|p| p.state != PlayerState::Eliminated && p.stack > 0)
            .count();

        active_count <= 1
    }

    /// Get remaining tournament players (not eliminated)
    pub fn get_remaining_players(&self) -> Vec<String> {
        self.players
            .iter()
            .filter(|p| p.state != PlayerState::Eliminated && p.stack > 0)
            .map(|p| p.user_id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::variant::{OmahaHi, variant_from_id};

    #[test]
    fn test_table_default_variant() {
        let table = PokerTable::new(
            "t1".to_string(),
            "Test Table".to_string(),
            5,
            10,
        );
        assert_eq!(table.variant_id(), "holdem");
        assert_eq!(table.variant_name(), "Texas Hold'em");
    }

    #[test]
    fn test_table_with_omaha_variant() {
        let table = PokerTable::with_variant(
            "t2".to_string(),
            "Omaha Table".to_string(),
            5,
            10,
            9,
            Box::new(OmahaHi),
        );
        assert_eq!(table.variant_id(), "omaha");
        assert_eq!(table.variant_name(), "Omaha");
    }

    #[test]
    fn test_table_with_variant_factory() {
        let variant = variant_from_id("omaha_hilo").expect("Should find variant");
        let table = PokerTable::with_variant(
            "t3".to_string(),
            "Omaha Hi-Lo Table".to_string(),
            5,
            10,
            9,
            variant,
        );
        assert_eq!(table.variant_id(), "omaha_hilo");
    }

    #[test]
    fn test_public_state_includes_variant() {
        let table = PokerTable::with_variant(
            "t4".to_string(),
            "Test".to_string(),
            5,
            10,
            6,
            Box::new(OmahaHi),
        );
        let state = table.get_public_state(None);
        assert_eq!(state.variant_id, "omaha");
        assert_eq!(state.variant_name, "Omaha");
    }

    #[test]
    fn test_table_clone_preserves_variant() {
        let table = PokerTable::with_variant(
            "t5".to_string(),
            "Clone Test".to_string(),
            5,
            10,
            9,
            Box::new(OmahaHi),
        );
        let cloned = table.clone();
        assert_eq!(cloned.variant_id(), "omaha");
        assert_eq!(cloned.variant_name(), "Omaha");
    }

    #[test]
    fn test_table_default_format() {
        let table = PokerTable::new(
            "t6".to_string(),
            "Test".to_string(),
            5,
            10,
        );
        assert_eq!(table.format_id(), "cash");
        assert!(table.can_cash_out());
        assert!(table.can_top_up());
    }

    #[test]
    fn test_public_state_includes_format() {
        let table = PokerTable::new(
            "t7".to_string(),
            "Test".to_string(),
            25,
            50,
        );
        let state = table.get_public_state(None);
        assert_eq!(state.format_id, "cash");
        assert!(state.can_cash_out);
        assert!(state.can_top_up);
    }

    #[test]
    fn test_table_with_sng_format() {
        use crate::game::format::SitAndGo;
        
        let sng = SitAndGo::new(100, 1500, 6, 300);
        let table = PokerTable::with_variant_and_format(
            "t8".to_string(),
            "SNG Test".to_string(),
            25,
            50,
            6,
            Box::new(TexasHoldem),
            Box::new(sng),
        );
        
        assert_eq!(table.format_id(), "sng");
        assert!(!table.can_cash_out());
        assert!(!table.can_top_up());
    }
}