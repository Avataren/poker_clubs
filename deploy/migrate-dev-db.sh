#!/bin/bash
# Migrate development database to Docker volume
set -e

echo "=========================================="
echo "Migrate Development DB to Docker"
echo "=========================================="
echo ""

DEV_DB="/home/avataren/src/poker/backend/poker.db"

if [ ! -f "$DEV_DB" ]; then
    echo "Error: Development database not found at $DEV_DB"
    exit 1
fi

DB_SIZE=$(du -h "$DEV_DB" | cut -f1)
echo "Development database: $DEV_DB ($DB_SIZE)"
echo ""

# Check if container is running
if docker compose ps | grep -q "poker.*Up"; then
    echo "⚠️  Docker container is currently running."
    read -p "Stop container to proceed with migration? (yes/no): " STOP
    if [ "$STOP" != "yes" ]; then
        echo "Migration cancelled."
        exit 0
    fi
    echo "Stopping container..."
    docker compose down
    echo ""
fi

# Backup current Docker database first
echo "Creating backup of current Docker database..."
mkdir -p backups
BACKUP_FILE="backups/docker-before-migration-$(date +%Y%m%d-%H%M%S).db"
docker run --rm \
  -v deploy_poker_data:/data:ro \
  -v "$(pwd)/backups:/backup" \
  debian:bookworm-slim \
  cp /data/poker.db "/backup/$(basename $BACKUP_FILE)" 2>/dev/null || echo "No existing Docker database to backup"

echo "✓ Backup saved: $BACKUP_FILE"
echo ""

# Copy development database
echo "Copying development database to Docker volume..."
docker run --rm \
  -v deploy_poker_data:/data \
  -v /home/avataren/src/poker/backend:/source:ro \
  debian:bookworm-slim \
  cp /source/poker.db /data/poker.db

echo "✓ Database copied"
echo ""

# Start container
read -p "Start Docker container now? (yes/no): " START
if [ "$START" == "yes" ]; then
    echo "Starting container..."
    docker compose up -d
    sleep 2
    
    # Verify
    echo ""
    echo "Verifying migration..."
    USER_COUNT=$(docker compose exec poker sqlite3 /data/poker.db "SELECT COUNT(*) FROM users;" 2>/dev/null || echo "0")
    echo "Users in Docker database: $USER_COUNT"
fi

echo ""
echo "=========================================="
echo "Migration complete!"
echo "=========================================="
echo "Development DB → Docker volume"
echo ""
