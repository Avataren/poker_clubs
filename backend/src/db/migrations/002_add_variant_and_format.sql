-- Migration 2: Add variant_id and format_id columns to tables
-- This migration adds support for storing poker variant and format information

-- Add variant_id column if it doesn't exist
ALTER TABLE tables ADD COLUMN variant_id TEXT NOT NULL DEFAULT 'holdem';

-- Add format_id column if it doesn't exist
ALTER TABLE tables ADD COLUMN format_id TEXT NOT NULL DEFAULT 'cash';
