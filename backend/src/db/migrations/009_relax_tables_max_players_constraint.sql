-- Relax `tables.max_players` upper bound so tournament-created SNG tables
-- can exceed 9 seats when configured.
--
-- This migration is written to recover cleanly even if a previous attempt
-- partially completed and left `tables_new` behind.

DROP TABLE IF EXISTS tables_migrated;

CREATE TABLE IF NOT EXISTS tables_migrated (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL,
    name TEXT NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    min_buyin INTEGER NOT NULL,
    max_buyin INTEGER NOT NULL,
    max_players INTEGER NOT NULL CHECK (max_players >= 2),
    variant_id TEXT NOT NULL DEFAULT 'holdem',
    format_id TEXT NOT NULL DEFAULT 'cash',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (club_id) REFERENCES clubs(id) ON DELETE CASCADE
);

-- Ensure both potential source tables exist so copy statements are always valid.
CREATE TABLE IF NOT EXISTS tables (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL,
    name TEXT NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    min_buyin INTEGER NOT NULL,
    max_buyin INTEGER NOT NULL,
    max_players INTEGER NOT NULL CHECK (max_players >= 2),
    variant_id TEXT NOT NULL DEFAULT 'holdem',
    format_id TEXT NOT NULL DEFAULT 'cash',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (club_id) REFERENCES clubs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tables_new (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL,
    name TEXT NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    min_buyin INTEGER NOT NULL,
    max_buyin INTEGER NOT NULL,
    max_players INTEGER NOT NULL CHECK (max_players >= 2),
    variant_id TEXT NOT NULL DEFAULT 'holdem',
    format_id TEXT NOT NULL DEFAULT 'cash',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (club_id) REFERENCES clubs(id) ON DELETE CASCADE
);

INSERT OR IGNORE INTO tables_migrated (
    id,
    club_id,
    name,
    small_blind,
    big_blind,
    min_buyin,
    max_buyin,
    max_players,
    variant_id,
    format_id,
    created_at
)
SELECT
    id,
    club_id,
    name,
    small_blind,
    big_blind,
    min_buyin,
    max_buyin,
    max_players,
    COALESCE(variant_id, 'holdem'),
    COALESCE(format_id, 'cash'),
    created_at
FROM tables;

INSERT OR IGNORE INTO tables_migrated (
    id,
    club_id,
    name,
    small_blind,
    big_blind,
    min_buyin,
    max_buyin,
    max_players,
    variant_id,
    format_id,
    created_at
)
SELECT
    id,
    club_id,
    name,
    small_blind,
    big_blind,
    min_buyin,
    max_buyin,
    max_players,
    COALESCE(variant_id, 'holdem'),
    COALESCE(format_id, 'cash'),
    created_at
FROM tables_new;

DROP TABLE IF EXISTS tables;
DROP TABLE IF EXISTS tables_new;
ALTER TABLE tables_migrated RENAME TO tables;

CREATE INDEX IF NOT EXISTS idx_tables_club ON tables(club_id);
