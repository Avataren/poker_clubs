-- Migration 4: Add tournament pre-seat window and seating status
-- This migration adds pre_seat_secs and updates the status constraint to include seating.

PRAGMA foreign_keys = OFF;

ALTER TABLE tournaments RENAME TO tournaments_old;

CREATE TABLE tournaments (
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

INSERT INTO tournaments (
    id,
    club_id,
    name,
    format_id,
    variant_id,
    buy_in,
    starting_stack,
    prize_pool,
    max_players,
    min_players,
    registered_players,
    remaining_players,
    current_blind_level,
    level_duration_secs,
    level_start_time,
    status,
    scheduled_start,
    pre_seat_secs,
    actual_start,
    finished_at,
    cancel_reason,
    created_at
)
SELECT
    id,
    club_id,
    name,
    format_id,
    variant_id,
    buy_in,
    starting_stack,
    prize_pool,
    max_players,
    2,
    registered_players,
    remaining_players,
    current_blind_level,
    level_duration_secs,
    level_start_time,
    status,
    scheduled_start,
    0,
    actual_start,
    finished_at,
    NULL,
    created_at
FROM tournaments_old;

-- Also need to recreate tournament_tables to reference the new tournaments table
ALTER TABLE tournament_tables RENAME TO tournament_tables_old;

CREATE TABLE tournament_tables (
    tournament_id TEXT NOT NULL,
    table_id TEXT NOT NULL,
    table_number INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    
    PRIMARY KEY (tournament_id, table_id),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE
);

INSERT INTO tournament_tables SELECT * FROM tournament_tables_old;

CREATE INDEX IF NOT EXISTS idx_tournament_tables_tournament ON tournament_tables(tournament_id);
CREATE INDEX IF NOT EXISTS idx_tournament_tables_table ON tournament_tables(table_id);

DROP TABLE tournament_tables_old;
DROP TABLE tournaments_old;

CREATE INDEX IF NOT EXISTS idx_tournaments_club ON tournaments(club_id);
CREATE INDEX IF NOT EXISTS idx_tournaments_status ON tournaments(status);
CREATE INDEX IF NOT EXISTS idx_tournaments_created ON tournaments(created_at);

PRAGMA foreign_keys = ON;
