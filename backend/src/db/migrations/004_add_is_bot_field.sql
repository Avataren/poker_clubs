-- Migration 4: Add is_bot field for proper bot user identification
-- This replaces the insecure username-based bot detection

ALTER TABLE users ADD COLUMN is_bot INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_users_is_bot ON users(is_bot);
