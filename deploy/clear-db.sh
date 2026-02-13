#!/bin/bash
# Clear/Reset Docker Database
# Deletes the poker_data volume to start with a fresh database
set -e

echo "=========================================="
echo "Clear Docker Database"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will DELETE all data in the Docker database!"
echo "   - All users will be removed"
echo "   - All tables will be removed"
echo "   - All game history will be removed"
echo "   - This action CANNOT be undone!"
echo ""

# Confirm
read -p "Are you sure you want to clear the database? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
read -p "Type 'DELETE' to confirm: " CONFIRM2
if [ "$CONFIRM2" != "DELETE" ]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""

# Check if container is running
if docker compose ps | grep -q "poker.*Up"; then
    echo "Stopping Docker container..."
    docker compose down
    echo "✓ Container stopped"
else
    echo "Container is not running"
fi

echo ""

# Check if volume exists
if docker volume inspect deploy_poker_data >/dev/null 2>&1; then
    echo "Removing Docker volume: deploy_poker_data"
    docker volume rm deploy_poker_data
    echo "✓ Database volume deleted"
else
    echo "Volume deploy_poker_data does not exist (already clean)"
fi

echo ""
echo "=========================================="
echo "Database Cleared!"
echo "=========================================="
echo ""

# Ask to restart
read -p "Start container with fresh database? (yes/no): " START
if [ "$START" == "yes" ]; then
    echo ""
    echo "Starting container..."
    docker compose up -d
    
    # Wait for container to be ready
    echo "Waiting for services to start..."
    sleep 3
    
    # Check status
    if docker compose ps | grep -q "poker.*Up"; then
        echo "✓ Container started with fresh database"
        echo ""
        echo "Database is now empty. You can:"
        echo "  - Register a new user at http://localhost:8080"
        echo "  - The database will be automatically initialized"
    else
        echo "⚠️  Container failed to start. Check logs:"
        echo "   docker compose logs"
    fi
fi

echo ""
echo "Done!"
echo ""
