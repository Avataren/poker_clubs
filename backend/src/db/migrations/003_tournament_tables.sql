-- Migration 3: Tournament support (Sit & Go and Multi-Table Tournaments)
-- This migration adds comprehensive tournament functionality

-- Tournaments table (stores SNG and MTT configurations)
CREATE TABLE IF NOT EXISTS tournaments (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL,
    name TEXT NOT NULL,
    format_id TEXT NOT NULL CHECK (format_id IN ('sng', 'mtt')),
    variant_id TEXT NOT NULL,
    
    -- Buy-in and prizes
    buy_in INTEGER NOT NULL,
    starting_stack INTEGER NOT NULL,
    prize_pool INTEGER NOT NULL DEFAULT 0,
    
    -- Structure
    max_players INTEGER NOT NULL,
    min_players INTEGER NOT NULL DEFAULT 2,
    registered_players INTEGER NOT NULL DEFAULT 0,
    remaining_players INTEGER NOT NULL DEFAULT 0,
    
    -- Blind structure
    current_blind_level INTEGER NOT NULL DEFAULT 0,
    level_duration_secs INTEGER NOT NULL,
    level_start_time TEXT,
    
    -- Status: registering, seating, running, paused, finished, cancelled
    status TEXT NOT NULL CHECK (status IN ('registering', 'seating', 'running', 'paused', 'finished', 'cancelled')),
    
    -- Timing
    scheduled_start TEXT,
    pre_seat_secs INTEGER NOT NULL DEFAULT 0,
    actual_start TEXT,
    finished_at TEXT,
    cancel_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (club_id) REFERENCES clubs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tournaments_club ON tournaments(club_id);
CREATE INDEX IF NOT EXISTS idx_tournaments_status ON tournaments(status);
CREATE INDEX IF NOT EXISTS idx_tournaments_created ON tournaments(created_at);

-- Tournament registrations (players who signed up)
CREATE TABLE IF NOT EXISTS tournament_registrations (
    tournament_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    registered_at TEXT NOT NULL DEFAULT (datetime('now')),
    starting_table_id TEXT,
    eliminated_at TEXT,
    finish_position INTEGER,
    prize_amount INTEGER DEFAULT 0,
    
    PRIMARY KEY (tournament_id, user_id),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tournament_registrations_user ON tournament_registrations(user_id);
CREATE INDEX IF NOT EXISTS idx_tournament_registrations_tournament ON tournament_registrations(tournament_id);

-- Tournament blind levels (predefined structure for each tournament)
CREATE TABLE IF NOT EXISTS tournament_blind_levels (
    tournament_id TEXT NOT NULL,
    level_number INTEGER NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    ante INTEGER NOT NULL DEFAULT 0,
    
    PRIMARY KEY (tournament_id, level_number),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tournament_blind_levels ON tournament_blind_levels(tournament_id, level_number);

-- Tournament tables (for MTTs with multiple tables, links tournaments to tables)
CREATE TABLE IF NOT EXISTS tournament_tables (
    tournament_id TEXT NOT NULL,
    table_id TEXT NOT NULL,
    table_number INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    
    PRIMARY KEY (tournament_id, table_id),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE,
    FOREIGN KEY (table_id) REFERENCES tables(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tournament_tables_tournament ON tournament_tables(tournament_id);
CREATE INDEX IF NOT EXISTS idx_tournament_tables_table ON tournament_tables(table_id);
