/// A pot with an amount and eligible player indices.
#[derive(Debug, Clone)]
pub struct SidePot {
    pub amount: i64,
    pub eligible: Vec<usize>,
}

/// Calculate side pots from player bets.
/// `player_bets`: (player_idx, total_bet, is_active)
/// Returns (pots, uncontested amounts per player).
pub fn calculate_side_pots(
    player_bets: &[(usize, i64, bool)],
) -> (Vec<SidePot>, Vec<(usize, i64)>) {
    if player_bets.is_empty() {
        return (vec![], vec![]);
    }

    let mut sorted: Vec<(usize, i64, bool)> = player_bets.to_vec();
    sorted.sort_by_key(|(_, bet, _)| *bet);

    let mut raw_pots = Vec::new();
    let mut uncontested = Vec::new();
    let mut prev_level = 0i64;

    for i in 0..sorted.len() {
        let (_, bet_level, _) = sorted[i];
        if bet_level <= prev_level {
            continue;
        }

        let level_contribution = bet_level - prev_level;
        let contributors = sorted.iter().filter(|(_, bet, _)| *bet > prev_level).count();
        let pot_amount = level_contribution * contributors as i64;

        let eligible: Vec<usize> = sorted
            .iter()
            .filter(|(_, bet, active)| *bet >= bet_level && *active)
            .map(|(idx, _, _)| *idx)
            .collect();

        if pot_amount > 0 {
            if eligible.len() <= 1 {
                if let Some(&player_idx) = eligible.first() {
                    uncontested.push((player_idx, pot_amount));
                }
            } else {
                raw_pots.push(SidePot {
                    amount: pot_amount,
                    eligible,
                });
            }
        }

        prev_level = bet_level;
    }

    // Merge consecutive pots with same eligible players
    let mut merged = Vec::new();
    for pot in raw_pots {
        if let Some(last) = merged.last_mut() {
            let last: &mut SidePot = last;
            if last.eligible == pot.eligible {
                last.amount += pot.amount;
                continue;
            }
        }
        merged.push(pot);
    }

    (merged, uncontested)
}

/// Award pots to winners. Returns (player_idx, amount_won) pairs.
pub fn award_pots(pots: &[SidePot], winners_by_pot: &[Vec<usize>]) -> Vec<(usize, i64)> {
    let mut payouts: Vec<(usize, i64)> = Vec::new();

    for (pot_idx, pot) in pots.iter().enumerate() {
        let winners = match winners_by_pot.get(pot_idx) {
            Some(w) if !w.is_empty() => w,
            _ => continue,
        };

        let share = pot.amount / winners.len() as i64;
        let remainder = pot.amount % winners.len() as i64;

        for (i, &winner_idx) in winners.iter().enumerate() {
            let amount = if i == 0 { share + remainder } else { share };
            if let Some(entry) = payouts.iter_mut().find(|(idx, _)| *idx == winner_idx) {
                entry.1 += amount;
            } else {
                payouts.push((winner_idx, amount));
            }
        }
    }

    payouts
}
