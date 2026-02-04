-- Migration 6: Add tournament rebuys, addons, and late registration support

ALTER TABLE tournaments ADD COLUMN allow_rebuys INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN max_rebuys INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN rebuy_amount INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN rebuy_stack INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN allow_addons INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN max_addons INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN addon_amount INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN addon_stack INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournaments ADD COLUMN late_registration_secs INTEGER NOT NULL DEFAULT 0;

ALTER TABLE tournament_registrations ADD COLUMN rebuys INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tournament_registrations ADD COLUMN addons INTEGER NOT NULL DEFAULT 0;
