# Tournament Implementation Plan
## Sit-and-Go (SNG) and Multi-Table Tournament (MTT) Features

**Date:** February 2, 2026  
**Status:** ✅ Phase 1-7 Complete - Testing Remaining  
**Author:** Backend Code Review & Architecture Planning

---

## 🎯 Implementation Status

### ✅ Phase 1: Database Schema - COMPLETE
- ✅ Created migration `003_tournament_tables.sql`
- ✅ Added `tournaments`, `tournament_registrations`, `tournament_blind_levels`, `tournament_tables` tables
- ✅ Added Tournament, TournamentRegistration, TournamentBlindLevel, TournamentTable models
- ✅ Updated migration runner to include migration 003
- ✅ All migrations run successfully

### ✅ Phase 2: Tournament Manager Core - COMPLETE
- ✅ Created `/backend/src/tournament/` module structure
- ✅ Implemented `prizes.rs` with PrizeStructure and distribution logic
  - ✅ 4/4 unit tests passing
  - ✅ Supports heads-up, 3-player, 6-max, 9-player, 18-player, and large tournaments
- ✅ Implemented `manager.rs` with TournamentManager
  - ✅ `create_sng()` and `create_mtt()` methods
  - ✅ `register_player()` with balance deduction
  - ✅ `unregister_player()` with refunds
  - ✅ `start_tournament()` with status management
  - ✅ `advance_blind_level()` for blind progression
  - ✅ `eliminate_player()` with finish position tracking
  - ✅ `distribute_prizes()` with automatic payouts

### ✅ Phase 3: Game Engine Integration - COMPLETE
- ✅ Added `update_blinds()` method to PokerTable
- ✅ Added `apply_antes()` method for ante collection
- ✅ Added `check_eliminations()` for player elimination detection
- ✅ Added `tournament_finished()` for tournament end detection
- ✅ Added `Eliminated` state to PlayerState enum
- ✅ Added `get_remaining_players()` helper method
- ✅ Added `tournament_id` field to PokerTable for tracking

### ✅ Phase 4: API Endpoints - COMPLETE
- ✅ Created `backend/src/api/tournaments.rs` with full endpoint set
- ✅ Tournament creation endpoints (SNG/MTT)
- ✅ Registration and unregistration endpoints
- ✅ Tournament listing and detail endpoints
- ✅ Tournament administration endpoints
- ✅ Results and prizes endpoints
- ✅ Wired up tournament routes in main.rs and lib.rs
- ✅ Created TournamentAppState with TournamentManager

### ✅ Phase 5: Background Services - COMPLETE
- ✅ Added `check_all_blind_levels()` method to TournamentManager
- ✅ Background task in main.rs to check blind levels every 10 seconds
- ✅ Automatic blind level advancement based on timer

### ✅ Phase 6: Table Creation & Management - COMPLETE
- ✅ Implemented `start_sng_table()` for single-table SNG creation
- ✅ Implemented `start_mtt_tables()` for multi-table distribution
- ✅ Player seating across tables for SNGs and MTTs
- ✅ Tournament-to-table linking via `tournament_tables`
- ✅ Blind updates propagated to all tournament tables

### ✅ Phase 7: Tournament Lifecycle Integration - COMPLETE
- ✅ Added `check_tournament_eliminations()` to TournamentManager
- ✅ Background task checking eliminations every 5 seconds
- ✅ Tournament event broadcasting via WebSocket
  - ✅ TournamentStarted message
  - ✅ TournamentBlindLevelIncreased message
  - ✅ TournamentPlayerEliminated message
  - ✅ TournamentFinished message
- ✅ Automatic prize distribution on tournament completion
- ✅ Tournament status tracking from registration to completion

### ⏳ Phase 8: Testing & Documentation - NEXT
- ⏳ Integration tests for full tournament lifecycle
- ⏳ API documentation updates
- ⏳ Client-side tournament UI integration

---

## Executive Summary

This document provides a comprehensive plan to extend the existing poker server with full Sit-and-Go (SNG) and Multi-Table Tournament (MTT) functionality. The current codebase has excellent foundations in place with:
- ✅ Complete Texas Hold'em game engine
- ✅ Real-time WebSocket communication
- ✅ Extensible variant/format system (partially implemented)
- ✅ Bot support for testing
- ✅ Database schema with migrations

The foundation classes `SitAndGo` and `MultiTableTournament` exist in [format.rs](backend/src/game/format.rs) but are not yet integrated into the live system.

---

## Backend Code Review Summary

### Strengths ✅

1. **Clean Architecture**
   - Clear separation: API → WebSocket → Game Engine → Database
   - Trait-based polymorphism for variants and formats
   - Proper error handling with custom error types
   - Well-structured modules

2. **Game Engine Excellence**
   - Complete pot management with side pots
   - Cryptographically secure RNG (ChaCha20)
   - Comprehensive hand evaluation via rs-poker
   - Proper betting round logic
   - Auto-advance for all-in situations

3. **Real-time Infrastructure**
   - WebSocket handler with broadcast channels per table
   - Club-level and global broadcasts prepared
   - Efficient table state synchronization
   - Background tasks for auto-advance and bot actions

4. **Database Design**
   - SQLite with SQLx for compile-time query verification
   - Proper indexes on foreign keys
   - Migration system in place
   - `variant_id` and `format_id` columns already added

5. **Testing Foundation**
   - Integration tests with in-memory database
   - Test helper functions for app setup
   - Good coverage of auth and club endpoints

### Areas for Improvement 🔧

1. **Code Organization**
   - Multiple backup files in [game/](backend/src/game/) (`table.rs.backup`, `table.rs.backup2`, `handler_old.rs`)
   - Dead code warnings suppressed globally (`#![allow(dead_code)]`)
   - Some prepared features not yet integrated

2. **Tournament Integration Gaps**
   - Format trait implemented but not used in game loop
   - No blind level advancement logic in main game
   - Tournament status tracking exists but not persisted
   - Prize distribution not implemented

3. **Data Persistence**
   - Tournament state lives only in memory
   - No recovery if server restarts mid-tournament
   - Table sessions tracked but not used for cashouts

4. **API Completeness**
   - No endpoints for tournament registration
   - No prize payout endpoints
   - Missing tournament listing/detail endpoints

5. **Minor TODOs**
   - [betting.rs#L105](backend/src/game/betting.rs#L105): Pot-limit max calculation
   - [betting.rs#L112](backend/src/game/betting.rs#L112): Fixed-limit street detection

### Security & Best Practices ✅

- ✅ Password hashing with bcrypt
- ✅ JWT authentication
- ✅ CORS configured (currently set to `Any` - recommend restricting in production)
- ✅ SQL injection protection via SQLx
- ✅ Input validation on bet amounts
- ✅ Proper state machine for game phases

---

## Implementation Plan

### Phase 1: Database Schema Extensions

**Goal:** Add tournament persistence and registration tracking

**Files to Modify:**
- `backend/src/db/migrations/003_tournament_tables.sql` (new)
- `backend/src/db/models.rs`

**New Tables:**

```sql
-- Tournaments (SNGs and MTTs)
CREATE TABLE IF NOT EXISTS tournaments (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL,
    name TEXT NOT NULL,
    format_id TEXT NOT NULL CHECK (format_id IN ('sng', 'mtt')),
    variant_id TEXT NOT NULL,
    
    -- Buy-in and prizes
    buy_in INTEGER NOT NULL,
    starting_stack INTEGER NOT NULL,
    prize_pool INTEGER NOT NULL DEFAULT 0,
    
    -- Structure
    max_players INTEGER NOT NULL,
    registered_players INTEGER NOT NULL DEFAULT 0,
    remaining_players INTEGER NOT NULL DEFAULT 0,
    
    -- Blind structure
    current_blind_level INTEGER NOT NULL DEFAULT 0,
    level_duration_secs INTEGER NOT NULL,
    level_start_time TEXT,
    
    -- Status
    status TEXT NOT NULL CHECK (status IN ('registering', 'running', 'paused', 'finished', 'cancelled')),
    
    -- Timing
    scheduled_start TEXT,
    actual_start TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (club_id) REFERENCES clubs(id) ON DELETE CASCADE
);

-- Tournament registrations
CREATE TABLE IF NOT EXISTS tournament_registrations (
    tournament_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    registered_at TEXT NOT NULL DEFAULT (datetime('now')),
    starting_table_id TEXT,
    eliminated_at TEXT,
    finish_position INTEGER,
    prize_amount INTEGER DEFAULT 0,
    
    PRIMARY KEY (tournament_id, user_id),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Tournament blind levels
CREATE TABLE IF NOT EXISTS tournament_blind_levels (
    tournament_id TEXT NOT NULL,
    level_number INTEGER NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    ante INTEGER NOT NULL DEFAULT 0,
    
    PRIMARY KEY (tournament_id, level_number),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE
);

-- Tournament tables (for MTTs with multiple tables)
CREATE TABLE IF NOT EXISTS tournament_tables (
    tournament_id TEXT NOT NULL,
    table_id TEXT NOT NULL,
    table_number INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    
    PRIMARY KEY (tournament_id, table_id),
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE,
    FOREIGN KEY (table_id) REFERENCES tables(id) ON DELETE CASCADE
);

CREATE INDEX idx_tournaments_club ON tournaments(club_id);
CREATE INDEX idx_tournaments_status ON tournaments(status);
CREATE INDEX idx_tournament_registrations_user ON tournament_registrations(user_id);
CREATE INDEX idx_tournament_tables_tournament ON tournament_tables(tournament_id);
```

**New Models:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Tournament {
    pub id: String,
    pub club_id: String,
    pub name: String,
    pub format_id: String,
    pub variant_id: String,
    pub buy_in: i64,
    pub starting_stack: i64,
    pub prize_pool: i64,
    pub max_players: i32,
    pub registered_players: i32,
    pub remaining_players: i32,
    pub current_blind_level: i32,
    pub level_duration_secs: i64,
    pub level_start_time: Option<String>,
    pub status: String,
    pub scheduled_start: Option<String>,
    pub actual_start: Option<String>,
    pub finished_at: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TournamentRegistration {
    pub tournament_id: String,
    pub user_id: String,
    pub registered_at: String,
    pub starting_table_id: Option<String>,
    pub eliminated_at: Option<String>,
    pub finish_position: Option<i32>,
    pub prize_amount: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TournamentBlindLevel {
    pub tournament_id: String,
    pub level_number: i32,
    pub small_blind: i64,
    pub big_blind: i64,
    pub ante: i64,
}
```

**Estimated Effort:** 4-6 hours

---

### Phase 2: Core Tournament Manager

**Goal:** Centralized tournament lifecycle management

**Files to Create:**
- `backend/src/tournament/mod.rs` (new)
- `backend/src/tournament/manager.rs` (new)
- `backend/src/tournament/sng.rs` (new)
- `backend/src/tournament/mtt.rs` (new)
- `backend/src/tournament/prizes.rs` (new)

**Key Components:**

```rust
// tournament/manager.rs
pub struct TournamentManager {
    pool: Arc<SqlitePool>,
    game_server: Arc<GameServer>,
    tournaments: Arc<RwLock<HashMap<String, Box<dyn TournamentController>>>>,
}

impl TournamentManager {
    pub async fn create_sng(
        &self,
        club_id: &str,
        config: SngConfig,
    ) -> Result<Tournament>;
    
    pub async fn create_mtt(
        &self,
        club_id: &str,
        config: MttConfig,
    ) -> Result<Tournament>;
    
    pub async fn register_player(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<()>;
    
    pub async fn unregister_player(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<()>;
    
    pub async fn start_tournament(&self, tournament_id: &str) -> Result<()>;
    
    pub async fn eliminate_player(
        &self,
        tournament_id: &str,
        user_id: &str,
    ) -> Result<()>;
    
    pub async fn advance_blind_level(&self, tournament_id: &str) -> Result<()>;
    
    pub async fn distribute_prizes(&self, tournament_id: &str) -> Result<Vec<PrizeWinner>>;
}

// tournament/sng.rs
pub struct SngController {
    tournament_id: String,
    config: SngConfig,
    table: Option<PokerTable>,
    registrations: Vec<TournamentRegistration>,
}

impl SngController {
    pub async fn auto_start_when_full(&mut self) -> bool;
    pub async fn handle_elimination(&mut self, user_id: &str) -> Result<()>;
    pub async fn check_finish(&self) -> Option<Vec<PrizeWinner>>;
}

// tournament/mtt.rs
pub struct MttController {
    tournament_id: String,
    config: MttConfig,
    tables: Vec<PokerTable>,
    registrations: Vec<TournamentRegistration>,
}

impl MttController {
    pub async fn balance_tables(&mut self) -> Result<()>;
    pub async fn break_table(&mut self, table_id: &str) -> Result<()>;
    pub async fn move_player(&mut self, user_id: &str, from_table: &str, to_table: &str) -> Result<()>;
}
```

**Responsibilities:**
- Registration management with balance deduction
- Tournament lifecycle (registering → running → finished)
- Blind level timer and advancement
- Prize pool calculation and distribution
- Player elimination tracking

**Estimated Effort:** 12-16 hours

---

### Phase 3: Game Engine Integration

**Goal:** Make PokerTable tournament-aware

**Files to Modify:**
- `backend/src/game/table.rs`
- `backend/src/game/mod.rs`
- `backend/src/game/format.rs`

**Changes Required:**

1. **Blind Level Updates**
   ```rust
   impl PokerTable {
       pub fn update_blinds(&mut self, small_blind: i64, big_blind: i64) {
           self.small_blind = small_blind;
           self.big_blind = big_blind;
           self.min_raise = big_blind;
       }
       
       pub fn apply_antes(&mut self, ante: i64) -> GameResult<()> {
           // Collect antes from all active players
       }
   }
   ```

2. **Player Elimination**
   ```rust
   impl PokerTable {
       pub fn check_eliminations(&mut self) -> Vec<String> {
           if !self.format.eliminates_players() {
               return vec![];
           }
           
           let mut eliminated = vec![];
           for player in &mut self.players {
               if player.stack == 0 && player.state == PlayerState::Active {
                   player.state = PlayerState::Eliminated;
                   eliminated.push(player.user_id.clone());
               }
           }
           eliminated
       }
   }
   ```

3. **Tournament End Detection**
   ```rust
   impl PokerTable {
       pub fn tournament_finished(&self) -> bool {
           if !self.format.eliminates_players() {
               return false;
           }
           
           let active_count = self.players.iter()
               .filter(|p| p.state != PlayerState::Eliminated)
               .count();
           
           active_count <= self.format.players_to_end()
       }
   }
   ```

**Estimated Effort:** 6-8 hours

---

### Phase 4: API Endpoints

**Goal:** Complete REST API for tournament operations

**Files to Create:**
- `backend/src/api/tournaments.rs` (new)

**Files to Modify:**
- `backend/src/api/mod.rs`
- `backend/src/lib.rs`

**Endpoints:**

```rust
// Tournament Management
POST   /api/tournaments/sng          // Create SNG
POST   /api/tournaments/mtt          // Create MTT
GET    /api/tournaments/club/:id     // List club's tournaments
GET    /api/tournaments/:id          // Get tournament details
DELETE /api/tournaments/:id          // Cancel (admin only)

// Registration
POST   /api/tournaments/:id/register
DELETE /api/tournaments/:id/unregister
GET    /api/tournaments/:id/players

// Administration
POST   /api/tournaments/:id/start    // Force start (admin)
POST   /api/tournaments/:id/pause    // Pause (admin)
POST   /api/tournaments/:id/resume   // Resume (admin)

// Results
GET    /api/tournaments/:id/results  // Final standings
GET    /api/tournaments/:id/prizes   // Prize breakdown
```

**Request/Response Examples:**

```rust
#[derive(Deserialize)]
pub struct CreateSngRequest {
    pub club_id: String,
    pub name: String,
    pub buy_in: i64,
    pub max_players: i32,      // 2, 6, or 9
    pub starting_stack: i64,
    pub level_duration_mins: i32,
    pub variant_id: Option<String>,
}

#[derive(Serialize)]
pub struct TournamentDetailResponse {
    pub tournament: Tournament,
    pub blind_levels: Vec<TournamentBlindLevel>,
    pub registrations: Vec<PlayerRegistration>,
    pub is_registered: bool,
    pub can_register: bool,
}

#[derive(Serialize)]
pub struct TournamentResultsResponse {
    pub tournament: Tournament,
    pub results: Vec<PlayerResult>,
}

#[derive(Serialize)]
pub struct PlayerResult {
    pub user_id: String,
    pub username: String,
    pub finish_position: i32,
    pub prize_amount: i64,
    pub eliminated_at: Option<String>,
}
```

**Estimated Effort:** 8-10 hours

---

### Phase 5: Background Services

**Goal:** Automated tournament progression

**Files to Modify:**
- `backend/src/main.rs`

**Files to Create:**
- `backend/src/tournament/scheduler.rs` (new)

**Background Tasks:**

1. **Blind Level Timer**
   ```rust
   tokio::spawn(async move {
       let mut interval = tokio::time::interval(Duration::from_secs(10));
       loop {
           interval.tick().await;
           tournament_mgr.check_blind_level_advancement().await;
       }
   });
   ```

2. **Auto-Start SNGs**
   ```rust
   tokio::spawn(async move {
       let mut interval = tokio::time::interval(Duration::from_secs(5));
       loop {
           interval.tick().await;
           tournament_mgr.check_auto_start_sngs().await;
       }
   });
   ```

3. **Table Balancing (MTT)**
   ```rust
   tokio::spawn(async move {
       let mut interval = tokio::time::interval(Duration::from_secs(30));
       loop {
           interval.tick().await;
           tournament_mgr.balance_all_mtt_tables().await;
       }
   });
   ```

4. **Tournament Cleanup**
   ```rust
   tokio::spawn(async move {
       let mut interval = tokio::time::interval(Duration::from_secs(60));
       loop {
           interval.tick().await;
           tournament_mgr.finish_completed_tournaments().await;
       }
   });
   ```

**Estimated Effort:** 6-8 hours

---

### Phase 6: WebSocket Extensions

**Goal:** Real-time tournament updates to clients

**Files to Modify:**
- `backend/src/ws/messages.rs`
- `backend/src/ws/handler.rs`

**New Message Types:**

```rust
// Client → Server
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    // ... existing messages ...
    
    RegisterTournament { tournament_id: String },
    UnregisterTournament { tournament_id: String },
}

// Server → Client
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    // ... existing messages ...
    
    TournamentUpdate {
        tournament_id: String,
        status: String,
        registered_players: i32,
        remaining_players: i32,
    },
    
    BlindLevelIncrease {
        tournament_id: String,
        level: i32,
        small_blind: i64,
        big_blind: i64,
        ante: i64,
    },
    
    PlayerEliminated {
        tournament_id: String,
        user_id: String,
        username: String,
        position: i32,
    },
    
    TournamentStarting {
        tournament_id: String,
        table_id: String,
        seat: i32,
    },
    
    TournamentFinished {
        tournament_id: String,
        winner_id: String,
        winner_username: String,
        results: Vec<PlayerResult>,
    },
    
    TableBalancing {
        tournament_id: String,
        new_table_id: String,
        new_seat: i32,
    },
}
```

**Broadcast Strategy:**
- Tournament updates → All registered players
- Blind increases → All active players at tournament tables
- Eliminations → All tournament players (for leaderboard)
- Table moves → Specific player being moved

**Estimated Effort:** 6-8 hours

---

### Phase 7: Testing & Validation

**Goal:** Comprehensive test coverage

**Files to Create:**
- `backend/tests/tournament_tests.rs` (new)
- `backend/tests/sng_integration_test.rs` (new)

**Test Cases:**

**Unit Tests:**
- ✅ Blind schedule creation and advancement
- ✅ Prize pool calculation (various player counts)
- ✅ Player elimination detection
- ✅ Tournament status transitions
- ✅ Registration limits enforcement

**Integration Tests:**
```rust
#[tokio::test]
async fn test_complete_sng_lifecycle() {
    // 1. Create SNG
    // 2. Register 6 players
    // 3. Verify auto-start
    // 4. Play hands until 5 eliminations
    // 5. Verify prize distribution
    // 6. Verify balance updates
}

#[tokio::test]
async fn test_blind_level_increases() {
    // 1. Start SNG with short levels (10 seconds)
    // 2. Wait for level increase
    // 3. Verify new blinds at table
    // 4. Verify ante collection
}

#[tokio::test]
async fn test_mtt_table_balancing() {
    // 1. Create MTT with 18 players (2 tables)
    // 2. Eliminate players from one table
    // 3. Verify table break and player moves
    // 4. Verify final table formation
}

#[tokio::test]
async fn test_late_registration() {
    // 1. Create MTT allowing late reg
    // 2. Start tournament
    // 3. Register player after start
    // 4. Verify seating and stack
}
```

**Bot Testing:**
```rust
#[tokio::test]
async fn test_bot_sng() {
    // Run complete 6-player SNG with 5 bots + 1 human
    // Verify realistic timing and decision-making
}
```

**Estimated Effort:** 10-12 hours

---

### Phase 8: Client Integration (Flutter)

**Goal:** UI for tournament features

**Files to Modify/Create:**
- `poker_client/lib/models/tournament.dart` (new)
- `poker_client/lib/services/tournament_service.dart` (new)
- `poker_client/lib/screens/tournament_lobby.dart` (new)
- `poker_client/lib/screens/tournament_table.dart` (new)
- `poker_client/lib/widgets/tournament_info_widget.dart` (new)
- `poker_client/lib/widgets/blind_level_indicator.dart` (new)

**Key UI Components:**

1. **Tournament Lobby**
   - List available tournaments (filtering by status)
   - Show registered players count
   - Register/unregister button
   - Countdown to scheduled start
   - Blind structure preview

2. **Tournament Table UI Enhancements**
   - Blind level indicator (current/next)
   - Level timer countdown
   - Remaining players count
   - Prize pool display
   - "In the money" indicator

3. **Tournament Results**
   - Final standings table
   - Prize distribution
   - Hand history summary

**Estimated Effort:** 16-20 hours

---

## Rollout Strategy

### Phase A: SNG Only (MVP)
**Timeline:** 2-3 weeks
- Complete Phases 1-4
- Single table SNGs (2, 6, 9 players)
- Fixed blind structure
- Standard prize payouts
- Basic testing

**Deliverables:**
- Working SNG registration
- Auto-start when full
- Blind increases every 5 minutes
- Automatic prize distribution

### Phase B: Enhanced SNG
**Timeline:** 1 week
- Complete Phase 5-7
- Background blind timer
- Comprehensive testing
- Performance optimization

### Phase C: MTT Foundation
**Timeline:** 2-3 weeks
- Extend Phases 2, 4, 5 for MTT
- Multi-table support
- Table balancing algorithm
- Late registration
- Tournament breaks

### Phase D: Full Feature Set
**Timeline:** 2-3 weeks
- Complete Phase 8
- Client UI for all features
- Tournament scheduling
- Lobby management
- Results history

**Total Estimated Time:** 8-10 weeks

---

## Technical Considerations

### 1. State Recovery

**Challenge:** Tournaments must survive server restarts

**Solution:**
- Persist tournament state to database on each significant event
- On startup, load active tournaments from DB
- Reconstruct in-memory state from table sessions
- Resume blind timers with adjusted start times

```rust
impl TournamentManager {
    pub async fn recover_active_tournaments(&mut self) -> Result<()> {
        let active = sqlx::query_as::<_, Tournament>(
            "SELECT * FROM tournaments WHERE status IN ('registering', 'running', 'paused')"
        )
        .fetch_all(&self.pool)
        .await?;
        
        for tournament in active {
            self.load_tournament_controller(&tournament).await?;
        }
        
        Ok(())
    }
}
```

### 2. Prize Distribution Logic

**Standard Payout Tables:**
| Players | 1st | 2nd | 3rd | 4th | 5th | 6th |
|---------|-----|-----|-----|-----|-----|-----|
| 2       | 100%|     |     |     |     |     |
| 6       | 65% | 35% |     |     |     |     |
| 9       | 50% | 30% | 20% |     |     |     |
| 18      | 40% | 25% | 17% | 10% | 8%  |     |
| 45+     | Scale with top 15% paid |     |     |     |     |

**Edge Cases:**
- Simultaneous eliminations (same hand) → Split prizes
- Tournament cancelled → Full refund
- Player disconnect in SNG → Auto-fold but stay in

### 3. Table Balancing Algorithm

**Goal:** Keep tables balanced within ±1 player

**Algorithm:**
```python
def balance_tables(tables):
    total_players = sum(len(t.players) for t in tables)
    avg_per_table = total_players / len(tables)
    
    # Break tables with fewest players
    if any_table_has <= avg_per_table / 2:
        break_smallest_table()
    
    # Redistribute to keep balanced
    while max_players - min_players > 1:
        move_player_from_largest_to_smallest()
```

**Special Cases:**
- Final table threshold (e.g., 9 players → break to single table)
- Break order: preserve table numbers, break highest numbers first
- Player position: moved players get next available seat

### 4. Performance Optimization

**Concerns:**
- 100+ player MTT with 10+ tables
- Frequent blind level updates
- Table balancing moves

**Solutions:**
- Batch database updates
- Index tournament_id + user_id lookups
- Cache tournament state in memory
- Use separate broadcast channels per tournament
- Limit WebSocket message frequency (aggregate updates)

### 5. Fairness & Security

**Random Seating:**
```rust
fn assign_seats(player_ids: &[String], table_id: &str) -> Vec<(String, usize)> {
    let mut rng = ChaCha20Rng::from_entropy();
    let mut shuffled = player_ids.to_vec();
    shuffled.shuffle(&mut rng);
    
    shuffled.into_iter()
        .enumerate()
        .map(|(seat, id)| (id, seat))
        .collect()
}
```

**Prevent Collusion:**
- Randomize seating on table breaks
- Track suspicious patterns (future enhancement)
- Player notes/reporting system (future)

---

## Database Migration Strategy

**Migration 003:**
```bash
cd backend
cargo sqlx migrate add tournament_tables
# Edit migration file with schema from Phase 1
cargo sqlx migrate run
```

**Rollback Plan:**
```sql
-- backend/src/db/migrations/003_tournament_tables.down.sql
DROP TABLE IF EXISTS tournament_tables;
DROP TABLE IF EXISTS tournament_blind_levels;
DROP TABLE IF EXISTS tournament_registrations;
DROP TABLE IF EXISTS tournaments;
```

---

## API Examples

### Create a 6-max SNG

```bash
curl -X POST http://localhost:3000/api/tournaments/sng \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "club_id": "club-123",
    "name": "Friday Night SNG",
    "buy_in": 1000,
    "max_players": 6,
    "starting_stack": 1500,
    "level_duration_mins": 5
  }'
```

### Register for Tournament

```bash
curl -X POST http://localhost:3000/api/tournaments/tourn-456/register \
  -H "Authorization: Bearer $TOKEN"
```

### View Tournament Details

```bash
curl http://localhost:3000/api/tournaments/tourn-456 \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "tournament": {
    "id": "tourn-456",
    "name": "Friday Night SNG",
    "status": "registering",
    "buy_in": 1000,
    "starting_stack": 1500,
    "max_players": 6,
    "registered_players": 4,
    "prize_pool": 4000
  },
  "blind_levels": [
    {"level": 1, "small_blind": 25, "big_blind": 50, "ante": 0},
    {"level": 2, "small_blind": 50, "big_blind": 100, "ante": 0},
    {"level": 3, "small_blind": 75, "big_blind": 150, "ante": 25}
  ],
  "registrations": [
    {"user_id": "user-1", "username": "Alice", "registered_at": "..."},
    {"user_id": "user-2", "username": "Bob", "registered_at": "..."}
  ],
  "is_registered": true,
  "can_register": true
}
```

---

## Success Metrics

**Technical:**
- [ ] 100% tournament completion rate (no crashes mid-tournament)
- [ ] Sub-100ms API response times
- [ ] Sub-500ms WebSocket message delivery
- [ ] Zero prize distribution errors
- [ ] Proper state recovery after server restart

**Functional:**
- [ ] Support 100+ player MTTs
- [ ] Blind levels advance exactly on schedule
- [ ] Table balancing within 30 seconds
- [ ] Complete SNG in under 60 minutes (6-max, 5 min levels)

**User Experience:**
- [ ] Real-time updates for all tournament events
- [ ] Clear blind level countdown
- [ ] Instant registration confirmation
- [ ] Accurate prize calculations

---

## Future Enhancements (Post-MVP)

1. **Rebuy & Add-on Tournaments**
   - Allow players to rebuy during first X levels
   - Single add-on at break

2. **Satellite Tournaments**
   - Winners get entry to larger tournament
   - Fractional payouts for bubble

3. **Tournament Series**
   - Leaderboard across multiple events
   - Series championships

4. **Scheduled Tournaments**
   - Cron-like scheduling
   - Guaranteed prize pools
   - Late registration periods

5. **Heads-Up SNGs**
   - Winner-take-all
   - Bracket-style multi-round

6. **Turbo & Hyper-Turbo**
   - Faster blind structures
   - Shorter level durations

7. **Knockout Bounties**
   - Prize for eliminating each player
   - Progressive knockouts (PKO)

8. **Tournament Hand History**
   - Full hand replay
   - Key hand analysis
   - Export to external tools

9. **Tournament Statistics**
   - ROI tracking
   - ITM% (in the money percentage)
   - Average finish position

---

## Dependencies & Library Considerations

**Current Stack:** ✅ All dependencies already in place
- Tokio for async runtime
- Axum for HTTP/WebSocket
- SQLx for database
- Serde for JSON

**No Additional Crates Needed**

---

## Cleanup Recommendations

Before starting implementation, clean up:

1. **Remove backup files:**
   ```bash
   rm backend/src/game/table.rs.backup*
   rm backend/src/ws/handler_old.rs
   ```

2. **Address TODOs:**
   - Implement pot-limit betting ([betting.rs](backend/src/game/betting.rs))
   - Implement fixed-limit street detection

3. **Remove global dead_code suppression:**
   - Replace `#![allow(dead_code)]` with targeted `#[allow(dead_code)]`
   - Only on genuinely unused prepared features

4. **Improve CORS:**
   ```rust
   // In lib.rs, replace Any with specific origins
   let cors = CorsLayer::new()
       .allow_origin("http://localhost:8080".parse::<HeaderValue>().unwrap())
       .allow_methods([Method::GET, Method::POST, Method::DELETE])
       .allow_headers([AUTHORIZATION, CONTENT_TYPE]);
   ```

---

## Conclusion

The poker server has excellent foundations for tournament support. The trait system design (`PokerVariant` and `GameFormat`) shows foresight and will make implementation straightforward. Key next steps:

1. **Phase 1 (Database)** - Add tournament tables and models
2. **Phase 2 (Core Logic)** - Build TournamentManager
3. **Phase 3 (Integration)** - Make PokerTable tournament-aware
4. **Phase 4 (API)** - Expose tournament endpoints
5. **Test Early and Often** - Each phase should be testable in isolation

With focused effort, a working SNG implementation (MVP) is achievable in 2-3 weeks. Full MTT support with client integration will take approximately 8-10 weeks total.

The codebase quality is high, the architecture is sound, and the foundations are solid. 🎰♠️♥️♣️♦️

---

**Document Version:** 1.0  
**Last Updated:** February 2, 2026
