-- Remove foreign key constraint from tournament_tables.table_id
-- Tournament tables are in-memory only and don't exist in the tables table

-- NOTE: Migration 003 already creates tournament_tables without the FK constraint,
-- so this migration is a no-op for new databases. It only exists for databases
-- that were created with an older version of migration 003 that had the FK.

-- This migration is now idempotent - the table already has the correct schema
-- from migration 003, so we don't need to do anything.
