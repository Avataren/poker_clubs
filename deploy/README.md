# Poker Server - Docker Deployment

Run the poker backend and Flutter web client in a single container.

## Quick Start

1. **Copy your ONNX model** into the `models/` folder:

   ```bash
   cp /path/to/your/poker_model.onnx models/poker_model.onnx
   ```

2. **Edit `.env`** to configure settings (JWT secret, model path, etc.)

3. **Build and run:**

   ```bash
   docker compose up --build -d
   ```

4. **Open** `http://localhost:8080` in your browser.

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_PORT` | `8080` | Port exposed on the host |
| `JWT_SECRET` | dev key | JWT signing secret (change in prod) |
| `CORS_ALLOWED_ORIGINS` | localhost | Allowed CORS origins |
| `POKER_BOT_MODEL_ONNX` | `/models/poker_model.onnx` | Path to ONNX model inside container |
| `GOOGLE_CLIENT_ID` | empty | Google OAuth client ID |
| `APPLE_CLIENT_ID` | empty | Apple OAuth client ID |

## Architecture

```
┌──────────────────────────────────┐
│         Docker Container         │
│                                  │
│  ┌──────────┐    ┌────────────┐  │
│  │  nginx   │───▶│  Rust      │  │
│  │  :80     │    │  Backend   │  │
│  │          │    │  :3000     │  │
│  │  Flutter │    └────────────┘  │
│  │  Web App │                    │
│  └──────────┘                    │
│                                  │
│  /models/ ← host volume (ONNX)  │
│  /data/   ← named volume (DB)   │
└──────────────────────────────────┘
```

- **nginx** serves the Flutter web client and proxies `/api/*` and `/ws` to the Rust backend
- **models/** is bind-mounted read-only from the host `deploy/models/` directory
- **poker_data** is a named Docker volume for the SQLite database

## Useful Commands

```bash
# View logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build -d

# Clear database (interactive, with confirmation)
./clear-db.sh

# Reset database (immediate, destroys data)
docker compose down -v
```

## Database Management

### Clear Database (Development)

For development, use the interactive script:

```bash
./clear-db.sh
```

This will:
- ✓ Prompt for confirmation (requires typing "DELETE")
- ✓ Stop the container
- ✓ Delete the database volume
- ✓ Optionally restart with fresh database

### Quick Database Reset

For automation/scripts:

```bash
docker compose down -v  # Deletes volumes immediately
docker compose up -d    # Restart with fresh DB
```

## Database Backups

The SQLite database is stored in a Docker named volume `poker_data` and mounted at `/data/poker.db` inside the container.

### Quick Backup/Restore Scripts

**Create a backup:**
```bash
./backup.sh
```

**Restore from backup:**
```bash
./restore.sh  # Interactive - will show list of backups to choose from
```

Backups are stored in `./backups/` directory with timestamp filenames (e.g., `poker-20260213-140530.db`). The backup script automatically keeps only the last 30 backups.

### Manual Backup

```bash
# Create backup directory
mkdir -p backups

# Backup database from running container
docker compose exec poker sqlite3 /data/poker.db ".backup '/data/poker-backup.db'"
docker compose cp poker:/data/poker-backup.db ./backups/poker-$(date +%Y%m%d-%H%M%S).db

# Or using docker cp directly from the volume
docker run --rm -v poker_data:/data -v $(pwd)/backups:/backup \
  debian:bookworm-slim cp /data/poker.db /backup/poker-$(date +%Y%m%d-%H%M%S).db
```

### Automated Backup Script

Create `backup.sh`:

```bash
#!/bin/bash
set -e

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="$BACKUP_DIR/poker-$DATE.db"

mkdir -p "$BACKUP_DIR"

echo "Creating backup: $BACKUP_FILE"
docker run --rm \
  -v poker_data:/data:ro \
  -v $(pwd)/backups:/backup \
  debian:bookworm-slim \
  cp /data/poker.db /backup/poker-$DATE.db

# Keep only last 30 backups
ls -t "$BACKUP_DIR"/poker-*.db | tail -n +31 | xargs -r rm

echo "Backup complete. Keeping last 30 backups."
```

Make it executable and run:

```bash
chmod +x backup.sh
./backup.sh
```

### Schedule Automatic Backups (cron)

Add to crontab (`crontab -e`):

```bash
# Backup every day at 2 AM
0 2 * * * cd /path/to/poker/deploy && ./backup.sh >> backup.log 2>&1
```

### Restore from Backup

```bash
# Stop the container
docker compose down

# Restore backup to volume
docker run --rm \
  -v poker_data:/data \
  -v $(pwd)/backups:/backup \
  debian:bookworm-slim \
  cp /backup/poker-YYYYMMDD-HHMMSS.db /data/poker.db

# Start container
docker compose up -d
```

### Export Volume to Host (for migration)

```bash
# Export entire volume
docker run --rm \
  -v poker_data:/data:ro \
  -v $(pwd):/backup \
  debian:bookworm-slim \
  tar czf /backup/poker_data_backup.tar.gz -C /data .

# Import on new host
docker run --rm \
  -v poker_data:/data \
  -v $(pwd):/backup \
  debian:bookworm-slim \
  tar xzf /backup/poker_data_backup.tar.gz -C /data
```
