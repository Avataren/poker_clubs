/// Actions a player can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    CheckCall,
    Raise(i64), // absolute raise-to amount
    AllIn,
}

/// Discrete action IDs for the neural network (9 actions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DiscreteAction {
    Fold = 0,
    CheckCall = 1,
    RaiseQuarterPot = 2,
    RaiseFortyPot = 3,
    RaiseSixtyPot = 4,
    RaiseEightyPot = 5,
    RaisePot = 6,
    RaiseOneAndHalfPot = 7,
    AllIn = 8,
}

impl DiscreteAction {
    pub const NUM_ACTIONS: usize = 9;

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Fold),
            1 => Some(Self::CheckCall),
            2 => Some(Self::RaiseQuarterPot),
            3 => Some(Self::RaiseFortyPot),
            4 => Some(Self::RaiseSixtyPot),
            5 => Some(Self::RaiseEightyPot),
            6 => Some(Self::RaisePot),
            7 => Some(Self::RaiseOneAndHalfPot),
            8 => Some(Self::AllIn),
            _ => None,
        }
    }

    /// Convert to a concrete Action given the current game state.
    pub fn to_action(self, pot: i64, current_bet: i64, player_bet: i64, _stack: i64) -> Action {
        let to_call = current_bet - player_bet;
        let effective_pot = pot + to_call; // pot after calling

        match self {
            Self::Fold => Action::Fold,
            Self::CheckCall => Action::CheckCall,
            Self::RaiseQuarterPot => {
                let raise_amount = effective_pot / 4;
                Action::Raise(current_bet + raise_amount)
            }
            Self::RaiseFortyPot => {
                let raise_amount = effective_pot * 2 / 5;
                Action::Raise(current_bet + raise_amount)
            }
            Self::RaiseSixtyPot => {
                let raise_amount = effective_pot * 3 / 5;
                Action::Raise(current_bet + raise_amount)
            }
            Self::RaiseEightyPot => {
                let raise_amount = effective_pot * 4 / 5;
                Action::Raise(current_bet + raise_amount)
            }
            Self::RaisePot => {
                let raise_amount = effective_pot;
                Action::Raise(current_bet + raise_amount)
            }
            Self::RaiseOneAndHalfPot => {
                let raise_amount = effective_pot * 3 / 2;
                Action::Raise(current_bet + raise_amount)
            }
            Self::AllIn => Action::AllIn,
        }
    }
}

/// Record of an action taken, for history encoding.
#[derive(Debug, Clone, Copy)]
pub struct ActionRecord {
    pub seat: usize,
    pub action: DiscreteAction,
    pub bet_ratio: f32, // bet amount / pot at time of action
}

impl ActionRecord {
    /// Encode as 11 floats: [seat_normalized, 9× action one-hot, bet_size/pot]
    pub fn encode(&self, num_players: usize) -> [f32; 11] {
        let seat_norm = if num_players > 1 {
            self.seat as f32 / (num_players - 1) as f32
        } else {
            0.0
        };
        let mut out = [0.0f32; 11];
        out[0] = seat_norm;
        out[1 + self.action as usize] = 1.0;
        out[10] = self.bet_ratio;
        out
    }
}
