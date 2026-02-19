-- Hand history tables: stores complete hand data for replay/review.
-- All cards stored face-up in DB. Visibility filtering happens at read time.

CREATE TABLE IF NOT EXISTS hand_histories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id TEXT NOT NULL,
    tournament_id TEXT,
    hand_number INTEGER NOT NULL DEFAULT 0,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    ante INTEGER NOT NULL DEFAULT 0,
    dealer_seat INTEGER NOT NULL,
    community_cards TEXT NOT NULL DEFAULT '[]',
    pot_total INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_hand_histories_table_id ON hand_histories(table_id);
CREATE INDEX IF NOT EXISTS idx_hand_histories_started_at ON hand_histories(started_at);

CREATE TABLE IF NOT EXISTS hand_players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hand_histories(id),
    seat INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    username TEXT NOT NULL,
    hole_cards TEXT NOT NULL DEFAULT '[]',
    starting_stack INTEGER NOT NULL,
    final_stack INTEGER NOT NULL DEFAULT 0,
    is_winner INTEGER NOT NULL DEFAULT 0,
    pot_won INTEGER NOT NULL DEFAULT 0,
    winning_hand_desc TEXT,
    showed_cards INTEGER NOT NULL DEFAULT 0,
    folded INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hand_players_hand_id ON hand_players(hand_id);
CREATE INDEX IF NOT EXISTS idx_hand_players_user_id ON hand_players(user_id);

CREATE TABLE IF NOT EXISTS hand_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hand_histories(id),
    street TEXT NOT NULL,
    seq INTEGER NOT NULL,
    seat INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    action_type TEXT NOT NULL,
    amount INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hand_actions_hand_id ON hand_actions(hand_id);
