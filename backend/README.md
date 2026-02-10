# Poker Server - Backend MVP

A fully functional Texas Hold'em poker server built with Rust, Axum, and WebSockets.

## Features ✅

### Authentication
- User registration with bcrypt password hashing
- JWT-based authentication
- Secure token-based API access

### Club System
- Create poker clubs
- Join existing clubs
- Admin-managed play money (10,000 starting balance per club)
- View club memberships

### Table Management
- Create cash game tables
- Configure blinds, buy-ins, player limits
- Real-time table state via WebSocket

### Game Engine
- **Complete Texas Hold'em implementation**
- Cryptographically secure card shuffling (ChaCha20 RNG)
- Hand evaluation using rs-poker library
- Pot management with side pot support
- All betting actions: fold, check, call, raise, all-in
- Automatic game phases: Pre-flop, Flop, Turn, River, Showdown
- Winner determination with proper hand ranking

### WebSocket Real-time
- Live game updates
- Player join/leave notifications
- Action broadcasts
- Authenticated connections

## Tech Stack

- **Rust** - Systems programming language
- **Axum** - Modern async web framework
- **SQLite** - Embedded database (via SQLx)
- **WebSockets** - Real-time bidirectional communication
- **JWT** - Stateless authentication
- **ChaCha20** - Cryptographic RNG for card shuffling
- **rs_poker** - Poker hand evaluation

## Quick Start

### 1. Run the Server

```bash
cd /home/avataren/src/poker/backend
cargo run
```

Server starts on `http://127.0.0.1:3000`

### 2. Test the API

**Register a user:**
```bash
curl -X POST http://127.0.0.1:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"player1","email":"player1@poker.com","password":"password123"}'
```

**Login:**
```bash
curl -X POST http://127.0.0.1:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"player1","password":"password123"}'
```

Save the returned `token` for authenticated requests.

**Create a club:**
```bash
curl -X POST http://127.0.0.1:3000/api/clubs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{"name":"Friday Night Poker"}'
```

**Create a table:**
```bash
curl -X POST http://127.0.0.1:3000/api/tables \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "club_id":"CLUB_ID_FROM_PREVIOUS_STEP",
    "name":"Main Table",
    "small_blind":50,
    "big_blind":100
  }'
```

**Get your clubs:**
```bash
curl http://127.0.0.1:3000/api/clubs/my \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## API Endpoints

### Auth
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token

### Clubs
- `POST /api/clubs` - Create a new club
- `GET /api/clubs/my` - Get user's clubs
- `POST /api/clubs/join` - Join an existing club

### Tables
- `POST /api/tables` - Create a poker table
- `GET /api/tables/club/:club_id` - List tables in a club

### WebSocket
- `GET /ws?token=YOUR_JWT_TOKEN` - Connect to game server

## WebSocket Protocol

### Client Messages

```json
{
  "type": "JoinTable",
  "payload": {
    "table_id": "...",
    "buyin": 5000
  }
}
```

```json
{
  "type": "PlayerAction",
  "payload": {
    "action": "Raise",
    "amount": 200
  }
}
```

Actions: `Fold`, `Check`, `Call`, `Raise(amount)`, `AllIn`

```json
{
  "type": "AddBot",
  "payload": {
    "table_id": "TABLE_ID",
    "name": "Optional Bot Name",
    "strategy": "model"
  }
}
```

Bot strategies:
- `tight`
- `aggressive`
- `calling_station`
- `model` (loads ONNX path from `POKER_BOT_MODEL_ONNX`)
- `model:/absolute/path/to/model.onnx` (inline path override)

### Server Messages

```json
{
  "type": "TableState",
  "payload": {
    "table_id": "...",
    "phase": "Flop",
    "community_cards": [...],
    "pot_total": 500,
    "current_bet": 100,
    "current_player_seat": 2,
    "players": [...]
  }
}
```

## Project Structure

```
backend/
├── Cargo.toml              # Dependencies
├── poker.db               # SQLite database (created on first run)
└── src/
    ├── main.rs            # Server entry point
    ├── config.rs          # Configuration
    ├── error.rs           # Error handling
    ├── auth/              # JWT authentication
    │   ├── mod.rs
    │   └── jwt.rs
    ├── db/                # Database layer
    │   ├── mod.rs
    │   ├── models.rs
    │   └── migrations/
    │       └── 001_initial_schema.sql
    ├── api/               # REST API endpoints
    │   ├── mod.rs
    │   ├── auth.rs        # Register/Login
    │   ├── clubs.rs       # Club management
    │   └── tables.rs      # Table management
    ├── ws/                # WebSocket layer
    │   ├── mod.rs
    │   ├── handler.rs     # Connection handling
    │   └── messages.rs    # Protocol messages
    └── game/              # Poker game engine
        ├── mod.rs
        ├── deck.rs        # Card deck + shuffle
        ├── hand.rs        # Hand evaluation
        ├── player.rs      # Player state
        ├── pot.rs         # Pot calculation
        └── table.rs       # Game state machine
```

## Game Flow

1. **Create/Join Table** - Players connect via WebSocket with JWT token
2. **Buy-in** - Players specify buy-in amount (deducted from club balance)
3. **Game Starts** - Minimum 2 players
4. **Blinds Posted** - Small/Big blinds automatically posted
5. **Hole Cards Dealt** - Each player receives 2 private cards
6. **Betting Rounds**:
   - Pre-flop (after hole cards)
   - Flop (3 community cards)
   - Turn (4th community card)
   - River (5th community card)
7. **Showdown** - Best hand wins
8. **Auto-start Next Hand** - If players remain

## Card Representation

Cards are represented as:
- Rank: 2-10, 11=Jack, 12=Queen, 13=King, 14=Ace
- Suit: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades

Example: `{"rank": 14, "suit": 3}` = Ace of Spades

## Security Features

- **Password Hashing**: bcrypt with default cost
- **JWT Tokens**: 7-day expiration
- **Secure Shuffling**: ChaCha20 cryptographic RNG
- **Server Authority**: All game logic validated server-side
- **Private Cards**: Players only see their own hole cards
- **CORS Enabled**: For web client development

## Configuration

Environment variables (create `.env` file):

```bash
DATABASE_URL=sqlite:poker.db
JWT_SECRET=your_secret_key_change_in_production
SERVER_HOST=127.0.0.1
SERVER_PORT=3000
POKER_BOT_MODEL_ONNX=/absolute/path/to/as_model.onnx
```

## Database Schema

- **users** - Player accounts
- **clubs** - Poker clubs
- **club_members** - Membership + balances
- **tables** - Game table configurations
- **table_sessions** - Active player sessions
- **transactions** - Currency movement history

## Next Steps

### For Web Client (Simple HTML/JS)
Create a basic HTML page with:
1. Login form
2. WebSocket connection
3. Display game state
4. Betting buttons

### For Flutter Client
1. Create Flutter project: `flutter create poker_client`
2. Add dependencies: `web_socket_channel`, `http`
3. Implement:
   - Login screen
   - WebSocket service
   - Table screen with card display
   - Betting controls

### Improvements
- [ ] Table chat system
- [ ] Hand history logging
- [ ] Player statistics
- [ ] Sit & Go tournaments
- [ ] Multi-table tournaments
- [ ] Omaha poker variant
- [ ] Table spectators
- [ ] Better error handling
- [ ] Comprehensive testing

## Testing

Run the included tests:
```bash
cargo test
```

## License

MIT License - Feel free to modify and use as needed.

---

**Built with ❤️ using Rust + Axum**
