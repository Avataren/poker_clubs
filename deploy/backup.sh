#!/bin/bash
# Poker Database Backup Script
# Backs up the SQLite database from the Docker volume
set -e

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="poker-$DATE.db"
MAX_BACKUPS=30

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "Poker Database Backup"
echo "=========================================="
echo "Time: $(date)"
echo "Backup file: $BACKUP_FILE"
echo ""

# Check if the volume exists
if ! docker volume inspect poker_data >/dev/null 2>&1; then
    echo "Error: Docker volume 'poker_data' not found!"
    echo "Make sure the poker container has been started at least once."
    exit 1
fi

# Create backup
echo "Creating backup from poker_data volume..."
docker run --rm \
  -v poker_data:/data:ro \
  -v "$(pwd)/backups:/backup" \
  debian:bookworm-slim \
  cp /data/poker.db "/backup/$BACKUP_FILE"

if [ -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_FILE" | cut -f1)
    echo "✓ Backup created successfully ($BACKUP_SIZE)"
else
    echo "✗ Backup failed!"
    exit 1
fi

# Clean up old backups
echo ""
echo "Cleaning up old backups (keeping last $MAX_BACKUPS)..."
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/poker-*.db 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    REMOVED=$(ls -t "$BACKUP_DIR"/poker-*.db | tail -n +$((MAX_BACKUPS + 1)) | wc -l)
    ls -t "$BACKUP_DIR"/poker-*.db | tail -n +$((MAX_BACKUPS + 1)) | xargs rm
    echo "✓ Removed $REMOVED old backup(s)"
else
    echo "✓ No cleanup needed ($BACKUP_COUNT backups total)"
fi

echo ""
echo "=========================================="
echo "Backup complete!"
echo "=========================================="
echo "Location: $BACKUP_DIR/$BACKUP_FILE"
echo "Total backups: $(ls -1 "$BACKUP_DIR"/poker-*.db 2>/dev/null | wc -l)"
echo ""
