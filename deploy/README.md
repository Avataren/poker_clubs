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

# Reset database (destroys data)
docker compose down -v
```
