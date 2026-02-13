#!/bin/bash
# Poker Database Restore Script
# Restores the SQLite database to the Docker volume
set -e

BACKUP_DIR="./backups"

echo "=========================================="
echo "Poker Database Restore"
echo "=========================================="
echo ""

# List available backups
if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A "$BACKUP_DIR"/poker-*.db 2>/dev/null)" ]; then
    echo "Error: No backups found in $BACKUP_DIR"
    echo "Run ./backup.sh first to create a backup."
    exit 1
fi

echo "Available backups:"
echo ""
select BACKUP_FILE in "$BACKUP_DIR"/poker-*.db "Cancel"; do
    if [ "$BACKUP_FILE" == "Cancel" ]; then
        echo "Restore cancelled."
        exit 0
    elif [ -n "$BACKUP_FILE" ]; then
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

echo ""
echo "Selected backup: $(basename "$BACKUP_FILE")"
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
BACKUP_DATE=$(stat -c %y "$BACKUP_FILE" 2>/dev/null || stat -f "%Sm" "$BACKUP_FILE" 2>/dev/null)
echo "Size: $BACKUP_SIZE"
echo "Created: $BACKUP_DATE"
echo ""

# Confirm
read -p "⚠️  This will REPLACE the current database. Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

# Check if container is running
if docker compose ps | grep -q "poker.*Up"; then
    echo ""
    echo "⚠️  Warning: The poker container is currently running."
    read -p "Stop the container now? (yes/no): " STOP_CONTAINER
    if [ "$STOP_CONTAINER" == "yes" ]; then
        echo "Stopping container..."
        docker compose down
    else
        echo "Restore cancelled. Please stop the container first with: docker compose down"
        exit 1
    fi
fi

# Restore backup
echo ""
echo "Restoring backup to poker_data volume..."
docker run --rm \
  -v poker_data:/data \
  -v "$(pwd)/backups:/backup" \
  debian:bookworm-slim \
  cp "/backup/$(basename "$BACKUP_FILE")" /data/poker.db

if [ $? -eq 0 ]; then
    echo "✓ Database restored successfully"
else
    echo "✗ Restore failed!"
    exit 1
fi

# Ask to restart
echo ""
read -p "Start the poker container now? (yes/no): " START_CONTAINER
if [ "$START_CONTAINER" == "yes" ]; then
    echo "Starting container..."
    docker compose up -d
    echo "✓ Container started"
fi

echo ""
echo "=========================================="
echo "Restore complete!"
echo "=========================================="
echo ""
