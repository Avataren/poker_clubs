-- Migration 10: add persisted user settings for avatar and deck style

ALTER TABLE users ADD COLUMN avatar_index INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN deck_style TEXT NOT NULL DEFAULT 'classic';

