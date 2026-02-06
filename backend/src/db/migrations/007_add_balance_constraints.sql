-- Application-level balance checks prevent negative balances.
-- SQLite CHECK constraint on existing column requires table recreation,
-- so we rely on application logic for balance validation.

-- Add an index for faster balance lookups
CREATE INDEX IF NOT EXISTS idx_club_members_balance ON club_members(club_id, user_id, balance)
