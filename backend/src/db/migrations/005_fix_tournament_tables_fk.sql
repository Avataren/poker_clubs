-- Remove foreign key constraint from tournament_tables.table_id
-- Tournament tables are in-memory only and don't exist in the tables table

-- SQLite doesn't support dropping foreign keys directly, so we need to recreate the table

-- Create new table without the table_id foreign key
CREATE TABLE tournament_tables_new (
    tournament_id TEXT NOT NULL,
    table_id TEXT NOT NULL,
    table_number INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (tournament_id, table_id),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE
);

-- Copy existing data
INSERT INTO tournament_tables_new SELECT * FROM tournament_tables;

-- Drop old table
DROP TABLE tournament_tables;

-- Rename new table
ALTER TABLE tournament_tables_new RENAME TO tournament_tables;

-- Recreate indexes
CREATE INDEX idx_tournament_tables_tournament ON tournament_tables(tournament_id);
CREATE INDEX idx_tournament_tables_table ON tournament_tables(table_id);
