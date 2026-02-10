/// Actions a player can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    CheckCall,
    Raise(i64), // absolute raise-to amount
    AllIn,
}

/// Discrete action IDs for the neural network (8 actions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DiscreteAction {
    Fold = 0,
    CheckCall = 1,
    RaiseHalfPot = 2,
    RaiseThreeQuarterPot = 3,
    RaisePot = 4,
    RaiseOneAndHalfPot = 5,
    RaiseTwoPot = 6,
    AllIn = 7,
}

impl DiscreteAction {
    pub const NUM_ACTIONS: usize = 8;

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Fold),
            1 => Some(Self::CheckCall),
            2 => Some(Self::RaiseHalfPot),
            3 => Some(Self::RaiseThreeQuarterPot),
            4 => Some(Self::RaisePot),
            5 => Some(Self::RaiseOneAndHalfPot),
            6 => Some(Self::RaiseTwoPot),
            7 => Some(Self::AllIn),
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
            Self::RaiseHalfPot => {
                let raise_amount = effective_pot / 2;
                Action::Raise(current_bet + raise_amount)
            }
            Self::RaiseThreeQuarterPot => {
                let raise_amount = effective_pot * 3 / 4;
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
            Self::RaiseTwoPot => {
                let raise_amount = effective_pot * 2;
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
    /// Encode as 7 floats: [seat_normalized, fold, check_call, raise_small, raise_med, raise_large, bet_ratio]
    pub fn encode(&self, num_players: usize) -> [f32; 7] {
        let seat_norm = if num_players > 1 {
            self.seat as f32 / (num_players - 1) as f32
        } else {
            0.0
        };
        let mut action_onehot = [0.0f32; 5];
        match self.action {
            DiscreteAction::Fold => action_onehot[0] = 1.0,
            DiscreteAction::CheckCall => action_onehot[1] = 1.0,
            DiscreteAction::RaiseHalfPot | DiscreteAction::RaiseThreeQuarterPot => {
                action_onehot[2] = 1.0
            }
            DiscreteAction::RaisePot | DiscreteAction::RaiseOneAndHalfPot => {
                action_onehot[3] = 1.0
            }
            DiscreteAction::RaiseTwoPot | DiscreteAction::AllIn => action_onehot[4] = 1.0,
        }
        [
            seat_norm,
            action_onehot[0],
            action_onehot[1],
            action_onehot[2],
            action_onehot[3],
            action_onehot[4],
            self.bet_ratio,
        ]
    }
}
