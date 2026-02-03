use super::{
    constants::{
        DEFAULT_MAX_SEATS, DEFAULT_SHOWDOWN_DELAY_MS, DEFAULT_STREET_DELAY_MS, MIN_PLAYERS_TO_START,
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
    Waiting,  // Waiting for players
    PreFlop,  // Hole cards dealt, pre-flop betting
    Flop,     // 3 community cards, betting
    Turn,     // 4th community card, betting
    River,    // 5th community card, betting
    Showdown, // Reveal and determine winner
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
    pub winning_hand: Option<String>, // Description of the winning hand
    pub dealer_seat: usize,
    pub current_player: usize,
    pub phase: GamePhase,
    pub community_cards: Vec<Card>,
    pub deck: Deck,
    pub pot: PotManager,
    pub current_bet: i64,
    pub min_raise: i64,
    pub last_phase_change_time: Option<u64>,
    pub street_delay_ms: u64,          // Delay between flop/turn/river
    pub showdown_delay_ms: u64,        // Delay to show results
    pub tournament_id: Option<String>, // If this is a tournament table
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
            tournament_id: self.tournament_id.clone(),
            variant: self.variant.clone_box(),
            format: self.format.clone_box(),
        }
    }
}

impl PokerTable {
    pub fn new(table_id: String, name: String, small_blind: i64, big_blind: i64) -> Self {
        Self::with_max_seats(table_id, name, small_blind, big_blind, DEFAULT_MAX_SEATS)
    }

    pub fn with_max_seats(
        table_id: String,
        name: String,
        small_blind: i64,
        big_blind: i64,
        max_seats: usize,
    ) -> Self {
        Self::with_variant(
            table_id,
            name,
            small_blind,
            big_blind,
            max_seats,
            Box::new(TexasHoldem),
        )
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
        Self::with_variant_and_format(
            table_id,
            name,
            small_blind,
            big_blind,
            max_seats,
            variant,
            format,
        )
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
            tournament_id: None,
            variant,
            format,
        }
    }

    /// Set the tournament ID for this table
    pub fn set_tournament_id(&mut self, tournament_id: Option<String>) {
        self.tournament_id = tournament_id;
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

    /// Force start a new hand (used for tournaments after all players are seated)
    pub fn force_start_hand(&mut self) {
        if self.phase == GamePhase::Waiting {
            self.start_new_hand();
        }
    }

    pub fn add_player(
        &mut self,
        user_id: String,
        username: String,
        buyin: i64,
    ) -> GameResult<usize> {
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
            tracing::debug!(
                "Player {} joining mid-hand, setting to WaitingForHand",
                player.username
            );
        }

        self.players.push(player);

        // Start game if we have enough players AND the format allows auto-start
        // Cash games: auto-start with 2+ players
        // Tournaments (SNG/MTT): do NOT auto-start, wait for explicit start
        if self.players.len() >= MIN_PLAYERS_TO_START 
            && self.phase == GamePhase::Waiting 
            && self.format.should_auto_start() {
            self.start_new_hand();
        }

        Ok(seat)
    }

    pub fn remove_player(&mut self, user_id: &str) {
        // Find the index of the player being removed
        let removed_idx = self.players.iter().position(|p| p.user_id == user_id);
        
        self.players.retain(|p| p.user_id != user_id);
        
        // Recalculate seat numbers
        for (idx, player) in self.players.iter_mut().enumerate() {
            player.seat = idx;
        }
        
        // Adjust dealer_seat and current_player if needed
        if let Some(removed_idx) = removed_idx {
            if !self.players.is_empty() {
                // If dealer was removed or is now out of bounds, move it to a valid position
                if self.dealer_seat >= self.players.len() || self.dealer_seat == removed_idx {
                    self.dealer_seat = if self.dealer_seat > 0 {
                        (self.dealer_seat - 1).min(self.players.len() - 1)
                    } else {
                        0
                    };
                } else if removed_idx < self.dealer_seat {
                    // Player before dealer was removed, adjust dealer index
                    self.dealer_seat -= 1;
                }
                
                // Same for current_player
                if self.current_player >= self.players.len() || self.current_player == removed_idx {
                    self.current_player = if self.current_player > 0 {
                        (self.current_player - 1).min(self.players.len() - 1)
                    } else {
                        0
                    };
                } else if removed_idx < self.current_player {
                    // Player before current was removed, adjust current index
                    self.current_player -= 1;
                }
            }
        }
    }
    pub fn take_seat(
        &mut self,
        user_id: String,
        username: String,
        seat: usize,
        buyin: i64,
    ) -> GameResult<usize> {
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
            return Err(GameError::InvalidSeat {
                seat,
                max_seats: self.max_seats,
            });
        }

        // Check if seat is occupied
        if self.players.iter().any(|p| p.seat == seat) {
            return Err(GameError::SeatOccupied { seat });
        }

        let mut player = Player::new(user_id, username, seat, buyin);

        // If a hand is in progress, make player wait until next hand
        if self.phase != GamePhase::Waiting {
            player.state = PlayerState::WaitingForHand;
            tracing::debug!(
                "Player {} joining mid-hand, setting to WaitingForHand",
                player.username
            );
        }

        self.players.push(player);

        // Start game if we have enough active players AND the format allows auto-start
        // Cash games: auto-start with 2+ players
        // Tournaments (SNG/MTT): do NOT auto-start, wait for explicit start
        if self.active_players_count() >= MIN_PLAYERS_TO_START 
            && self.phase == GamePhase::Waiting 
            && self.format.should_auto_start() {
            self.start_new_hand();
        }

        Ok(seat)
    }

    pub fn top_up(&mut self, user_id: &str, amount: i64) -> GameResult<()> {
        // Find the player
        let player = self
            .players
            .iter_mut()
            .find(|p| p.user_id == user_id)
            .ok_or(GameError::PlayerNotAtTable)?;

        // Validate top-up amount
        if amount <= 0 {
            return Err(GameError::InvalidAction {
                reason: "Top-up amount must be positive".to_string(),
            });
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
            tracing::info!(
                "Player {} topped up ${} and is now waiting for next hand",
                player.username,
                amount
            );
        }

        // If we're in Waiting phase, check if we can now start a new hand
        if self.phase == GamePhase::Waiting {
            let playable_count = self.players.iter().filter(|p| p.stack > 0).count();

            if playable_count >= MIN_PLAYERS_TO_START {
                tracing::info!("Enough players with chips after top-up, starting new hand");
                self.start_new_hand();
            }
        }

        Ok(())
    }

    pub fn stand_up(&mut self, user_id: &str) -> GameResult<()> {
        let player = self
            .players
            .iter_mut()
            .find(|p| p.user_id == user_id)
            .ok_or(GameError::PlayerNotAtTable)?;

        // If player is in an active hand, mark them to stand up after hand concludes
        if self.phase != GamePhase::Waiting
            && (player.state == PlayerState::Active || player.state == PlayerState::AllIn)
        {
            player.state = PlayerState::SittingOut;
            tracing::debug!(
                "Player {} will stand up after current hand",
                player.username
            );
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
        self.players
            .iter()
            .filter(|p| p.state == PlayerState::Active || p.state == PlayerState::WaitingForHand)
            .count()
    }

    pub fn start_new_hand(&mut self) {
        self.last_winner_message = None;
        self.winning_hand = None;

        // Reset all players for new hand
        for player in &mut self.players {
            player.reset_for_new_hand();
        }

        // Count players who can actually play (have chips and not eliminated)
        let playable_count = self.players.iter().filter(|p| p.stack > 0).count();

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
        // For the first hand from Waiting phase, find the first eligible player (not the next one)
        if self.phase == GamePhase::Waiting {
            self.dealer_seat = self.first_eligible_player_for_button();
        } else {
            self.dealer_seat = self.next_eligible_player_for_button(self.dealer_seat);
        }

        // Post blinds (also sets current_player)
        self.post_blinds();

        // Deal hole cards
        self.deal_hole_cards();

        // Set phase - current_player was already set by post_blinds
        self.phase = GamePhase::PreFlop;
    }

    fn post_blinds(&mut self) {
        let num_players = self.players.len();
        
        if num_players == 2 {
            // Heads-up special case: dealer posts SB, other player posts BB
            // Dealer is at self.dealer_seat
            let dealer_idx = self.dealer_seat;
            let other_idx = (dealer_idx + 1) % 2;
            
            // Dealer posts SB
            let sb_amount = self.players[dealer_idx].place_bet(self.small_blind);
            self.pot.add_bet(dealer_idx, sb_amount);
            
            // Other player posts BB
            let bb_amount = self.players[other_idx].place_bet(self.big_blind);
            self.pot.add_bet(other_idx, bb_amount);
            
            self.current_bet = self.big_blind;
            
            // In heads-up, dealer acts first pre-flop (after posting SB)
            self.current_player = dealer_idx;
            
            tracing::info!(
                "Heads-up blinds posted: Dealer (SB) at idx {} (seat {}, {}) posted ${}, BB at idx {} (seat {}, {}) posted ${}",
                dealer_idx,
                self.players[dealer_idx].seat,
                self.players[dealer_idx].username,
                sb_amount,
                other_idx,
                self.players[other_idx].seat,
                self.players[other_idx].username,
                bb_amount
            );
        } else {
            // 3+ players: normal blind posting
            // Small blind is the next player after dealer (includes sitting out in tournaments)
            let sb_seat = self.next_player_for_blind(self.dealer_seat);
            tracing::info!("post_blinds: dealer_seat={}, sb_seat={}", self.dealer_seat, sb_seat);
            let sb_amount = self.players[sb_seat].place_bet(self.small_blind);
            self.pot.add_bet(sb_seat, sb_amount);
            // Posting blind does NOT count as acting - player can still raise

            // Big blind is the next player after small blind
            let bb_seat = self.next_player_for_blind(sb_seat);
            tracing::info!("post_blinds: bb_seat={}", bb_seat);
            let bb_amount = self.players[bb_seat].place_bet(self.big_blind);
            self.pot.add_bet(bb_seat, bb_amount);
            // Posting blind does NOT count as acting - BB has option to raise

            self.current_bet = self.big_blind;

            // First to act is after big blind
            self.current_player = self.next_active_player(bb_seat);

            tracing::info!(
                "Blinds posted: Dealer at idx {} (seat {}, {}), SB=${} at idx {} (seat {}, {}), BB=${} at idx {} (seat {}, {})",
                self.dealer_seat,
                self.players[self.dealer_seat].seat,
                self.players[self.dealer_seat].username,
                sb_amount,
                sb_seat,
                self.players[sb_seat].seat,
                self.players[sb_seat].username,
                bb_amount,
                bb_seat,
                self.players[bb_seat].seat,
                self.players[bb_seat].username
            );
        }
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
            tracing::debug!(
                "Dealt card round {} of {}, starting from seat {}",
                round + 1,
                hole_cards_count,
                sb_seat
            );
        }
    }

    pub fn handle_action(&mut self, user_id: &str, action: PlayerAction) -> GameResult<()> {
        tracing::debug!(
            "handle_action: user_id={}, action={:?}, current_player={}",
            user_id,
            action,
            self.current_player
        );

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
                    return Err(GameError::CannotCheck {
                        current_bet: self.current_bet,
                    });
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
                    return Err(GameError::RaiseTooSmall {
                        min_raise: self.min_raise,
                        attempted: amount,
                    });
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

        tracing::info!(
            "Player {} acted, advancing. Current player before: {}",
            self.players[self.current_player].username,
            self.current_player
        );

        // Move to next player or next phase
        self.advance_action();

        tracing::info!(
            "After advance: current_player={}, state={:?}",
            self.current_player,
            if self.current_player < self.players.len() {
                format!("{:?}", self.players[self.current_player].state)
            } else {
                "INVALID INDEX".to_string()
            }
        );

        Ok(())
    }

    fn advance_action(&mut self) {
        // Check if betting round is complete
        if self.is_betting_round_complete() {
            self.advance_phase();
            // Don't auto-advance anymore - delays are handled by check_auto_advance
        } else {
            self.current_player = self.next_active_player(self.current_player);
            
            // Auto-fold sitting out players in tournaments
            if self.format.eliminates_players() {
                let current = &self.players[self.current_player];
                if current.state == PlayerState::SittingOut {
                    tracing::info!("Auto-folding sitting out player: {}", current.username);
                    self.players[self.current_player].fold();
                    self.players[self.current_player].has_acted_this_round = true;
                    self.players[self.current_player].last_action = Some("Auto-Fold (Sitting Out)".to_string());
                    // Recursively advance to next player
                    self.advance_action();
                }
            }
        }
    }

    fn is_betting_round_complete(&self) -> bool {
        let active_players: Vec<&Player> = self.players.iter().filter(|p| p.can_act()).collect();

        if active_players.is_empty() {
            return true;
        }

        // Need at least 2 checks:
        // 1. All active players have acted this round
        // 2. All active players have matched the current bet

        let all_acted = active_players.iter().all(|p| p.has_acted_this_round);
        let all_matched = active_players
            .iter()
            .all(|p| p.current_bet == self.current_bet);

        // Debug logging
        tracing::debug!(
            "Betting round check: phase={:?}, current_bet={}, active_players={}, all_acted={}, all_matched={}",
            self.phase, self.current_bet, active_players.len(), all_acted, all_matched
        );
        for (i, p) in self.players.iter().enumerate() {
            tracing::debug!(
                "  Player {}: {}, bet={}, acted={}, state={:?}",
                i,
                p.username,
                p.current_bet,
                p.has_acted_this_round,
                p.state
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
        let active_in_hand: Vec<usize> = self
            .players
            .iter()
            .enumerate()
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
                self.players[winner_idx].username,
                total
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
                let hand_rank = self
                    .variant
                    .evaluate_hand(&player.hole_cards, &self.community_cards);
                hands.push((idx, hand_rank));
            }
        }

        // Determine winners
        let winner_indices = determine_winners(hands.clone());

        // Get the winning hand description and mark cards as highlighted for all winners
        if let Some(&first_winner_idx) = winner_indices.first() {
            // Find the winning hand_rank (all winners have the same rank)
            if let Some((_, winning_hand_rank)) =
                hands.iter().find(|(idx, _)| *idx == first_winner_idx)
            {
                self.winning_hand = Some(winning_hand_rank.description.clone());

                // Process each winner for card highlighting
                for &winner_idx in &winner_indices {
                    // Get the best cards for this winner
                    if let Some((_, hand_rank)) = hands.iter().find(|(idx, _)| *idx == winner_idx) {
                        let best_cards = &hand_rank.best_cards;

                        tracing::info!("Winner {} best cards: {:?}", winner_idx, best_cards);

                        // Highlight community cards that are in the best hand
                        for community_card in &mut self.community_cards {
                            if best_cards.iter().any(|c| {
                                c.rank == community_card.rank && c.suit == community_card.suit
                            }) {
                                community_card.highlighted = true;
                                tracing::info!(
                                    "Highlighted community card: rank={} suit={}",
                                    community_card.rank,
                                    community_card.suit
                                );
                            }
                        }

                        // Highlight hole cards of the winner that are in the best hand
                        if let Some(winner) = self.players.get_mut(winner_idx) {
                            for hole_card in &mut winner.hole_cards {
                                if best_cards
                                    .iter()
                                    .any(|c| c.rank == hole_card.rank && c.suit == hole_card.suit)
                                {
                                    hole_card.highlighted = true;
                                    tracing::info!(
                                        "Highlighted hole card for {}: rank={} suit={}",
                                        winner.username,
                                        hole_card.rank,
                                        hole_card.suit
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Calculate side pots based on each player's total contribution
        let player_bets: Vec<(usize, i64, bool)> = self
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.total_bet_this_hand > 0)
            .map(|(idx, p)| (idx, p.total_bet_this_hand, p.is_active_in_hand()))
            .collect();
        self.pot.calculate_side_pots(&player_bets);

        // Determine winners for each pot based on eligible players
        let mut winners_by_pot = Vec::new();
        for pot in &self.pot.pots {
            let eligible_hands: Vec<(usize, HandRank)> = hands
                .iter()
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
            winner_names.push(format!(
                "{} wins ${}",
                self.players[*player_idx].username, amount
            ));
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

        tracing::info!(
            "Showdown complete. Winner: {}. Will auto-advance after delay.",
            winner_names.join(", ")
        );
    }

    fn next_active_player(&self, after: usize) -> usize {
        let mut idx = (after + 1) % self.players.len();
        let start = idx;

        tracing::debug!(
            "next_active_player: after={}, num_players={}",
            after,
            self.players.len()
        );

        loop {
            let player = &self.players[idx];
            tracing::debug!(
                "  Checking idx={}: username={}, can_act={}, state={:?}",
                idx,
                player.username,
                player.can_act(),
                player.state
            );
            
            // In tournaments, include sitting out players so they can be auto-folded
            // In cash games, only include active players
            let is_eligible = if self.format.eliminates_players() {
                player.can_act() || player.state == PlayerState::SittingOut
            } else {
                player.can_act()
            };
            
            if is_eligible {
                tracing::info!(
                    "next_active_player: returning idx={} ({})",
                    idx,
                    player.username
                );
                return idx;
            }
            idx = (idx + 1) % self.players.len();
            if idx == start {
                tracing::warn!(
                    "next_active_player: No active players found! Returning fallback {}",
                    after
                );
                break; // No active players found
            }
        }

        after // Fallback
    }

    /// Find the first player eligible for dealer button starting from seat 0
    /// Dealer button: must have chips and not be sitting out or eliminated
    fn first_eligible_player_for_button(&self) -> usize {
        for (idx, player) in self.players.iter().enumerate() {
            // Player is eligible for dealer if they have chips and aren't sitting out or eliminated
            if player.stack > 0 
                && player.state != PlayerState::SittingOut 
                && player.state != PlayerState::Eliminated 
            {
                tracing::info!(
                    "first_eligible_player_for_button: returning idx={} (seat {}, {})",
                    idx,
                    player.seat,
                    player.username
                );
                return idx;
            }
        }
        tracing::warn!("first_eligible_player_for_button: No eligible players found! Returning 0");
        0 // Fallback
    }

    /// Find the next player eligible for dealer button
    /// Dealer button: must have chips and not be sitting out or eliminated
    fn next_eligible_player_for_button(&self, after: usize) -> usize {
        let mut idx = (after + 1) % self.players.len();
        let start = idx;

        tracing::debug!(
            "next_eligible_player_for_button: after={}, num_players={}",
            after,
            self.players.len()
        );

        loop {
            let player = &self.players[idx];
            // Player is eligible for dealer if they have chips and aren't sitting out or eliminated
            if player.stack > 0 
                && player.state != PlayerState::SittingOut 
                && player.state != PlayerState::Eliminated 
            {
                tracing::info!(
                    "next_eligible_player_for_button: returning idx={} (seat {}, {})",
                    idx,
                    player.seat,
                    player.username
                );
                return idx;
            }
            tracing::debug!(
                "  Skipping idx={}: username={}, stack={}, state={:?}",
                idx,
                player.username,
                player.stack,
                player.state
            );
            idx = (idx + 1) % self.players.len();
            if idx == start {
                tracing::warn!("next_eligible_player_for_button: No eligible players found! Returning fallback {}", after);
                break; // No eligible players found
            }
        }

        after // Fallback
    }

    /// Find the next player for blind posting (includes sitting out players in tournaments)
    /// In tournaments: sitting out players still pay blinds
    /// In cash games: sitting out players don't pay blinds
    fn next_player_for_blind(&self, after: usize) -> usize {
        let mut idx = (after + 1) % self.players.len();
        let start = idx;

        loop {
            let player = &self.players[idx];
            // In tournaments, sitting out players still pay blinds
            // Only skip eliminated players
            let is_eligible = if self.format.eliminates_players() {
                player.stack > 0 && player.state != PlayerState::Eliminated
            } else {
                // In cash games, skip sitting out and eliminated
                player.stack > 0 
                    && player.state != PlayerState::SittingOut 
                    && player.state != PlayerState::Eliminated
            };

            if is_eligible {
                return idx;
            }
            
            idx = (idx + 1) % self.players.len();
            if idx == start {
                break;
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
                tracing::info!(
                    "Auto-advancing from {:?} after {}ms delay",
                    self.phase,
                    elapsed
                );

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
        self.get_public_state_with_tournament(for_user_id, None)
    }

    pub fn get_public_state_with_tournament(
        &self,
        for_user_id: Option<&str>,
        tournament_info: Option<TournamentInfo>,
    ) -> PublicTableState {
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
            self.players
                .iter()
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
            players: self
                .players
                .iter()
                .map(|p| {
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
                                Card {
                                    rank: 0,
                                    suit: 0,
                                    highlighted: false,
                                    face_up: false
                                };
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
                })
                .collect(),
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
            tournament_id: tournament_info.as_ref().map(|t| t.tournament_id.clone()),
            tournament_blind_level: tournament_info.as_ref().map(|t| t.blind_level),
            // Show current tournament level blinds (always show what the tournament is at now)
            tournament_small_blind: tournament_info.as_ref().map(|_| self.small_blind),
            tournament_big_blind: tournament_info.as_ref().map(|_| self.big_blind),
            tournament_level_start_time: tournament_info.as_ref().and_then(|t| t.level_start_time.clone()),
            tournament_level_duration_secs: tournament_info.as_ref().map(|t| t.level_duration_secs),
            tournament_next_small_blind: tournament_info.as_ref().and_then(|t| t.next_small_blind),
            tournament_next_big_blind: tournament_info.as_ref().and_then(|t| t.next_big_blind),
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
    // Tournament info (only present for tournament tables)
    pub tournament_id: Option<String>,
    pub tournament_blind_level: Option<i64>,
    pub tournament_small_blind: Option<i64>,
    pub tournament_big_blind: Option<i64>,
    pub tournament_level_start_time: Option<String>,
    pub tournament_level_duration_secs: Option<i64>,
    pub tournament_next_small_blind: Option<i64>,
    pub tournament_next_big_blind: Option<i64>,
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

/// Tournament info to include in table state
#[derive(Debug, Clone)]
pub struct TournamentInfo {
    pub tournament_id: String,
    pub blind_level: i64,
    pub level_start_time: Option<String>,
    pub level_duration_secs: i64,
    pub next_small_blind: Option<i64>,
    pub next_big_blind: Option<i64>,
}

impl PokerTable {
    /// Set tournament blinds for this hand (called at start of hand)
    /// Takes a snapshot of the current tournament level blinds
    // ==================== Tournament-Specific Methods ====================

    /// Update blinds and minimum raise (called when tournament blind level advances)
    /// Blinds are updated immediately but take effect at the start of the next hand
    pub fn update_blinds(&mut self, small_blind: i64, big_blind: i64) {
        // Simple snapshot approach: just update the blinds
        // They'll be used at the start of the next hand
        self.small_blind = small_blind;
        self.big_blind = big_blind;
        self.min_raise = big_blind;
        tracing::info!(
            "Table {} blinds updated to {}/{} (will take effect at next hand)",
            self.table_id,
            small_blind,
            big_blind
        );
    }

    /// Apply pending blinds if any (called at start of new hand)
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
    /// Eliminated players are removed from the table
    pub fn check_eliminations(&mut self) -> Vec<String> {
        if !self.format.eliminates_players() {
            return vec![];
        }

        let mut eliminated = vec![];
        
        // Find all players with 0 stack
        let mut to_remove = vec![];
        for (idx, player) in self.players.iter().enumerate() {
            if player.stack == 0 && player.state != PlayerState::Eliminated {
                eliminated.push(player.user_id.clone());
                to_remove.push(idx);
                tracing::info!(
                    "Player {} ({}) eliminated from tournament",
                    player.username,
                    player.user_id
                );
            }
        }

        // Remove eliminated players from the table (in reverse order to preserve indices)
        for &idx in to_remove.iter().rev() {
            self.players.remove(idx);
        }

        // Adjust current_player index if needed
        if !to_remove.is_empty() && !self.players.is_empty() {
            // Ensure current_player is still valid
            if self.current_player >= self.players.len() {
                self.current_player = 0;
            }
        }

        eliminated
    }

    /// Check if tournament is finished (1 or fewer players remaining)
    pub fn tournament_finished(&self) -> bool {
        if !self.format.eliminates_players() {
            return false;
        }

        // Since eliminated players are removed, just count remaining players with chips
        let active_count = self.players.iter().filter(|p| p.stack > 0).count();

        active_count <= 1
    }

    /// Get remaining tournament players (not eliminated)
    /// Note: eliminated players are removed from the table, so this returns all players
    pub fn get_remaining_players(&self) -> Vec<String> {
        self.players
            .iter()
            .filter(|p| p.stack > 0)
            .map(|p| p.user_id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::variant::{variant_from_id, OmahaHi};

    #[test]
    fn test_table_default_variant() {
        let table = PokerTable::new("t1".to_string(), "Test Table".to_string(), 5, 10);
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
        let table = PokerTable::new("t6".to_string(), "Test".to_string(), 5, 10);
        assert_eq!(table.format_id(), "cash");
        assert!(table.can_cash_out());
        assert!(table.can_top_up());
    }

    #[test]
    fn test_public_state_includes_format() {
        let table = PokerTable::new("t7".to_string(), "Test".to_string(), 25, 50);
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

    #[test]
    fn test_first_hand_dealer_position() {
        // Test that the first hand assigns dealer to the first eligible player
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);
        
        // Add 3 players
        table.take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000).unwrap();
        table.take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000).unwrap();
        table.take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000).unwrap();
        
        // Table should have started the hand automatically
        assert_eq!(table.phase, GamePhase::PreFlop);
        
        // Dealer should be at position 0 (first eligible player)
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(table.players[0].username, "Player 1");
    }

    #[test]
    fn test_blinds_posted_correctly() {
        // Test that SB and BB are posted by the correct players in a 3-player game
        use crate::game::format::SitAndGo;
        
        // Create table with SNG format to prevent auto-start
        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );
        
        // Add 3 players at seats 0, 1, 2
        println!("Adding players...");
        table.take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000).unwrap();
        println!("After adding p1, phase={:?}, players.len()={}", table.phase, table.players.len());
        
        table.take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000).unwrap();
        println!("After adding p2, phase={:?}, players.len()={}", table.phase, table.players.len());
        
        table.take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000).unwrap();
        println!("After adding p3, phase={:?}, players.len()={}", table.phase, table.players.len());
        
        // Game should NOT have auto-started (SNG format)
        assert_eq!(table.phase, GamePhase::Waiting);
        
        // Force start the hand
        table.force_start_hand();
        
        // Verify game started
        assert_eq!(table.phase, GamePhase::PreFlop);
        
        // Debug: print all players and their bets
        println!("Dealer at array index: {}", table.dealer_seat);
        for (idx, player) in table.players.iter().enumerate() {
            println!("Player[{}]: name={}, seat={}, bet={}, stack={}", 
                idx, player.username, player.seat, player.current_bet, player.stack);
        }
        
        // Dealer at array position 0, so:
        // - SB should be at array position 1 (Player 2)
        // - BB should be at array position 2 (Player 3)
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(table.players[1].current_bet, 50, "Player at index 1 should have posted SB");
        assert_eq!(table.players[2].current_bet, 100, "Player at index 2 should have posted BB");
        assert_eq!(table.players[0].current_bet, 0, "Dealer at index 0 should not have posted");
        
        // Verify stacks are reduced correctly
        assert_eq!(table.players[0].stack, 1000, "Dealer stack should be unchanged");
        assert_eq!(table.players[1].stack, 950, "SB stack should be reduced by 50");
        assert_eq!(table.players[2].stack, 900, "BB stack should be reduced by 100");
    }

    #[test]
    fn test_first_to_act_after_blinds() {
        // Test that the first player to act is the one after BB
        use crate::game::format::SitAndGo;
        
        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );
        
        // Add 4 players
        table.take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000).unwrap();
        table.take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000).unwrap();
        table.take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000).unwrap();
        table.take_seat("p4".to_string(), "Player 4".to_string(), 3, 1000).unwrap();
        
        // Force start
        table.force_start_hand();
        
        // Dealer at position 0, SB at 1, BB at 2
        // First to act should be position 3 (after BB)
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(table.current_player, 3, "First to act should be position 3");
        assert_eq!(table.players[table.current_player].username, "Player 4");
    }

    #[test]
    fn test_heads_up_blind_positions_simple() {
        // Test heads-up setup in isolation
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);
        
        // Add exactly 2 players
        println!("Adding first player...");
        table.take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000).unwrap();
        println!("Phase after p1: {:?}, players.len()={}", table.phase, table.players.len());
        
        println!("Adding second player...");
        table.take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000).unwrap();
        println!("Phase after p2: {:?}, players.len()={}", table.phase, table.players.len());
        
        // Game should have started
        assert_eq!(table.phase, GamePhase::PreFlop);
        assert_eq!(table.players.len(), 2);
        
        println!("Dealer at index: {}", table.dealer_seat);
        for (idx, p) in table.players.iter().enumerate() {
            println!("Player[{}]: seat={}, name={}, bet={}, stack={}", 
                idx, p.seat, p.username, p.current_bet, p.stack);
        }
        
        // In heads-up: dealer posts SB, other player posts BB
        // dealer_seat should be 0
        assert_eq!(table.dealer_seat, 0, "Dealer should be at index 0");
        
        // Dealer (index 0) should post SB (50)
        // Non-dealer (index 1) should post BB (100)
        assert_eq!(table.players[0].current_bet, 50, "Dealer should post SB in heads-up");
        assert_eq!(table.players[1].current_bet, 100, "Non-dealer should post BB in heads-up");
    }

    #[test]
    fn test_heads_up_blind_positions() {
        // In heads-up (2 players), dealer posts SB and acts first pre-flop
        let mut table = PokerTable::new("test".to_string(), "Test Table".to_string(), 50, 100);
        
        // Add 2 players
        table.take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000).unwrap();
        table.take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000).unwrap();
        
        // Dealer at position 0
        assert_eq!(table.dealer_seat, 0);
        
        // In heads-up: dealer (pos 0) posts SB, other player (pos 1) posts BB
        assert_eq!(table.players[0].current_bet, 50, "Dealer should post SB in heads-up");
        assert_eq!(table.players[1].current_bet, 100, "Non-dealer should post BB in heads-up");
        
        // In heads-up, dealer acts first pre-flop (after posting SB)
        assert_eq!(table.current_player, 0, "Dealer should act first in heads-up");
    }

    #[test]
    fn test_nine_player_sng_blinds() {
        // Test a 9-player SNG starting positions
        use crate::game::format::SitAndGo;
        
        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );
        
        // Add 9 players
        for i in 0..9 {
            table.take_seat(
                format!("p{}", i),
                format!("Player {}", i + 1),
                i,
                1000
            ).unwrap();
        }
        
        // Force start
        table.force_start_hand();
        
        // Dealer should be at position 0
        assert_eq!(table.dealer_seat, 0);
        
        // SB at position 1, BB at position 2
        assert_eq!(table.players[1].current_bet, 50, "Position 1 should post SB");
        assert_eq!(table.players[2].current_bet, 100, "Position 2 should post BB");
        
        // Verify only SB and BB have posted
        assert_eq!(table.players[0].current_bet, 0);
        for i in 3..9 {
            assert_eq!(table.players[i].current_bet, 0, "Position {} should not have posted", i);
        }
        
        // First to act should be position 3
        assert_eq!(table.current_player, 3);
        
        // All players should have cards
        for i in 0..9 {
            assert_eq!(table.players[i].hole_cards.len(), 2, "Player {} should have 2 cards", i);
        }
    }

    #[test]
    fn test_dealer_advances_between_hands() {
        // Test that dealer button moves correctly between hands
        use crate::game::format::SitAndGo;
        
        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );
        
        // Add 3 players
        table.take_seat("p1".to_string(), "Player 1".to_string(), 0, 1000).unwrap();
        table.take_seat("p2".to_string(), "Player 2".to_string(), 1, 1000).unwrap();
        table.take_seat("p3".to_string(), "Player 3".to_string(), 2, 1000).unwrap();
        
        // Force start first hand
        table.force_start_hand();
        
        // First hand - dealer at position 0
        assert_eq!(table.dealer_seat, 0);
        assert_eq!(table.players[1].current_bet, 50, "First hand: Position 1 should post SB");
        assert_eq!(table.players[2].current_bet, 100, "First hand: Position 2 should post BB");
        
        // Fast-forward to showdown and start new hand
        table.phase = GamePhase::Showdown;
        table.start_new_hand();
        
        // Second hand - dealer should move to position 1
        assert_eq!(table.dealer_seat, 1);
        assert_eq!(table.players[2].current_bet, 50, "Second hand: Position 2 should now post SB");
        assert_eq!(table.players[0].current_bet, 100, "Second hand: Position 0 should now post BB");
    }

    #[test]
    fn test_all_players_receive_hole_cards() {
        // Test that all players receive the correct number of hole cards
        use crate::game::format::SitAndGo;
        
        let sng_format = Box::new(SitAndGo::new(100, 1000, 9, 300));
        let mut table = PokerTable::with_variant_and_format(
            "test".to_string(),
            "Test Table".to_string(),
            50,
            100,
            9,
            Box::new(TexasHoldem),
            sng_format,
        );
        
        // Add 5 players
        for i in 0..5 {
            table.take_seat(
                format!("p{}", i),
                format!("Player {}", i + 1),
                i,
                1000
            ).unwrap();
        }
        
        // Force start
        table.force_start_hand();
        
        // All 5 players should have 2 hole cards each
        for i in 0..5 {
            assert_eq!(
                table.players[i].hole_cards.len(),
                2,
                "Player {} should have 2 hole cards",
                i + 1
            );
        }
        
        // Verify no duplicate cards between players
        let mut all_cards = Vec::new();
        for player in &table.players {
            for card in &player.hole_cards {
                assert!(!all_cards.contains(card), "Duplicate card dealt: {:?}", card);
                all_cards.push(card.clone());
            }
        }
    }
}

