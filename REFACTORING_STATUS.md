# Poker Project Refactoring Status

## Overview

This document tracks the ongoing refactoring effort to improve the poker codebase for better extensibility, readability, and stability. The goal is to support thousands of players with multiple game types (Sit & Gos, Multi-Table Tournaments, Omaha variants, etc.).

## Current Architecture

### Backend (Rust + Axum)
- **Web Framework**: Axum 0.7 with tokio async runtime
- **Database**: SQLite via sqlx 0.7
- **WebSocket**: Real-time game state via tokio broadcast channels
- **Authentication**: JWT tokens via jsonwebtoken crate
- **Hand Evaluation**: rs_poker 2.0 crate
- **Card Shuffling**: ChaCha20 cryptographic RNG

### Frontend (Flutter/Dart)
- **State Management**: Provider pattern
- **Services**: ApiService (REST), WebSocketService (real-time)
- **Screens**: Login, Register, Clubs, Club Lobby, Game

---

## Completed Refactoring (Commits 1-6)

### ✅ Commit 1: Extract Game Constants
**File**: `backend/src/game/constants.rs`

Centralized magic numbers into named constants:
- `DEFAULT_MAX_SEATS` (9)
- `MIN_PLAYERS_TO_START` (2)
- `DEFAULT_STARTING_BALANCE` (10000)
- Timing delays (`DEFAULT_STREET_DELAY_MS`, `DEFAULT_SHOWDOWN_DELAY_MS`)
- Card counts per variant (`HOLDEM_HOLE_CARDS`, `OMAHA_HOLE_CARDS`)

### ✅ Commit 2: Add Typed GameError
**File**: `backend/src/game/error.rs`

Replaced `String` errors with typed `GameError` enum:
- Table errors: `TableFull`, `InvalidSeat`, `SeatOccupied`
- Player errors: `PlayerNotAtTable`, `NotEnoughChips`
- Action errors: `NotYourTurn`, `CannotCheck`, `RaiseTooSmall`
- Game state errors: `InvalidPhase`, `GameNotInProgress`

Benefits:
- Better error handling and pattern matching
- Clearer API contracts
- Foundation for error localization

### ✅ Commit 3: Add PokerVariant Trait System
**File**: `backend/src/game/variant.rs`

Created abstraction for poker game types:
- `PokerVariant` trait with methods for hole cards, streets, betting structure
- Implementations: `TexasHoldem`, `OmahaHi`, `OmahaHiLo`, `ShortDeckHoldem`
- `BettingStructure` enum: `NoLimit`, `PotLimit`, `FixedLimit`
- `HandRequirements` for Omaha-style "must use X cards" rules
- Factory function `variant_from_id()` for dynamic variant selection

### ✅ Commit 4: Extract BettingEngine
**File**: `backend/src/game/betting.rs`

Separated betting logic from table state:
- `BettingRound`: Tracks current bet, min raise, betting structure
- `BettingValidator`: Validates actions without modifying state
- `BettingExecutor`: Executes validated actions
- `BettingEngine`: Combined interface for validation + execution

Benefits:
- Easier testing of betting logic in isolation
- Reusable across different game formats
- Foundation for pot-limit and fixed-limit betting

### ✅ Commit 5: Add GameFormat Trait
**File**: `backend/src/game/format.rs`

Created abstraction for game structures:
- `GameFormat` trait: `can_join()`, `can_cash_out()`, `has_increasing_blinds()`, etc.
- `CashGame`: Open entry, static blinds, no eliminations
- `SitAndGo`: Fixed players, increasing blinds, plays to one winner
- `MultiTableTournament`: Foundation for MTT support
- `BlindSchedule`: Manages blind level progression
- `PrizeStructure`: Calculates payouts by position

### ✅ Commit 6: Clean Up Warnings
Added appropriate `#[allow(dead_code)]` attributes to new infrastructure modules and pre-existing code designed for future features. Fixed visibility issues. Result: **zero compiler warnings**.

### ✅ Commit 7: Integrate PokerVariant into PokerTable
**Files**: `backend/src/game/table.rs`, `backend/src/game/variant.rs`, `backend/src/game/mod.rs`

Wired the `PokerVariant` trait into the main game loop:
- Added `clone_box()` and `Debug` to `PokerVariant` trait for trait object support
- Added `variant: Box<dyn PokerVariant>` field to `PokerTable`
- New constructor `with_variant()` to create tables with specific variants
- Updated `deal_hole_cards()` to use `variant.hole_cards_count()`
- Updated `resolve_showdown()` to use `variant.evaluate_hand()`
- Added `variant_id` and `variant_name` to `PublicTableState` for clients
- Exported variant types from `game/mod.rs`
- Added 5 new tests for variant integration

### ✅ Commit 8: Integrate GameFormat into PokerTable
**Files**: `backend/src/game/table.rs`, `backend/src/game/format.rs`, `backend/src/game/mod.rs`

Wired the `GameFormat` trait into PokerTable:
- Added `format_id()` and `clone_box()` to `GameFormat` trait
- Added `format: Box<dyn GameFormat>` field to `PokerTable`
- New constructor `with_variant_and_format()` for full customization
- Added `format_id`, `format_name`, `can_cash_out`, `can_top_up` to `PublicTableState`
- Exported format types from `game/mod.rs`
- Added 3 new tests for format integration

### ✅ Commit 9: Update Flutter Client
**Files**: `poker_client/lib/models/game_state.dart`, `poker_client/lib/screens/game_screen.dart`

Updated Flutter client to display new backend fields:
- Added variant and format fields to `GameState` model
- Added `gameTypeDescription` getter for UI display
- Game screen now shows variant and format info
- Top-up button respects format's `canTopUp` setting
- Backwards compatible with default values

### ✅ Commit 10: Implement Omaha Hand Evaluation
**Files**: `backend/src/game/hand.rs`, `backend/src/game/variant.rs`

Implemented proper Omaha hand evaluation (must use exactly 2 hole + 3 community cards):
- Added `evaluate_omaha_hand()` function with combination generation
- Added `combinations()` helper for generating k-combinations from a slice
- Overrode `evaluate_hand()` in `OmahaHi` and `OmahaHiLo` variants
- Added 4 new tests for Omaha evaluation (combinations, two-hole-cards rule, edge cases)
- Added 1 new test for variant `evaluate_hand()` integration

### ✅ Commit 11: Add API Endpoints for Variant/Format Table Creation
**Files**: `backend/src/api/tables.rs`, `backend/src/game/format.rs`, `backend/src/game/mod.rs`, `backend/src/ws/handler.rs`, `backend/src/error.rs`

Added REST API endpoints for creating tables with specific variants and formats:
- Updated `POST /api/tables` to accept optional `variant_id` and `format_id` parameters
- Added `GET /api/tables/variants` endpoint to list all available poker variants
- Added `GET /api/tables/formats` endpoint to list all available game formats
- Added `format_from_id()` and `available_formats()` factory functions
- Added `create_table_with_options()` to GameServer for variant/format support
- Added `BadRequest` error variant for validation errors

### ✅ Commit 12: Update Flutter Client with Variant/Format Support
**Files**: `poker_client/lib/services/api_service.dart`, `poker_client/lib/screens/tables_screen.dart`

Updated Flutter client to support variant/format selection:
- Added `getVariants()` and `getFormats()` methods to ApiService
- Added `VariantInfo` and `FormatInfo` model classes
- Updated `createTable()` to accept optional `variantId` and `formatId`
- Added variant/format dropdown selectors in TablesScreen
- Uses new initialValue pattern for dropdowns (avoid deprecation warnings)

---

## Current Test Status

```
running 46 tests
- game::betting::tests (9 tests) ✅
- game::deck::tests (5 tests) ✅
- game::error::tests (2 tests) ✅
- game::format::tests (4 tests) ✅
- game::hand::tests (8 tests) ✅   <- 4 NEW (Omaha evaluation)
- game::pot::tests (4 tests) ✅
- game::table::tests (8 tests) ✅
- game::variant::tests (6 tests) ✅ <- 1 NEW (Omaha evaluate_hand)

test result: ok. 46 passed; 0 failed
```

---

## Future Refactoring Goals

### Phase 1: Integrate New Infrastructure into PokerTable (MOSTLY COMPLETE)

**Priority: HIGH**

Tasks:
1. ~~Add `PokerVariant` field to `PokerTable`~~ ✅
2. ~~Add `GameFormat` field to `PokerTable`~~ ✅
3. ~~Update `deal_hole_cards()` to use `variant.hole_cards_count()`~~ ✅
4. ~~Update hand evaluation to use `variant.evaluate_hand()`~~ ✅
5. ~~Update Flutter client for new fields~~ ✅
6. Replace inline betting logic with `BettingEngine` (optional - current code works)
7. ~~Implement Omaha-specific hand evaluation (must use 2 hole + 3 community)~~ ✅
8. ~~Add API endpoint to create tables with different variants/formats~~ ✅

### Phase 2: Actor Model for Table Management

**Priority: HIGH**

Currently tables are managed via `Arc<RwLock<HashMap>>`. This creates potential for lock contention at scale.

Proposed approach:
```rust
// Each table runs as an independent actor
struct TableActor {
    table: PokerTable,
    rx: mpsc::Receiver<TableCommand>,
    broadcast_tx: broadcast::Sender<TableEvent>,
}

enum TableCommand {
    JoinTable { user_id: String, buyin: i64, reply: oneshot::Sender<Result<(), GameError>> },
    PlayerAction { user_id: String, action: PlayerAction, reply: oneshot::Sender<Result<(), GameError>> },
    // ...
}
```

Benefits:
- No lock contention between tables
- Each table processes commands sequentially (no race conditions)
- Tables can be distributed across threads/processes
- Natural fit for tournament table balancing

### Phase 3: Tournament System

**Priority: MEDIUM**

Build on the `GameFormat` infrastructure to implement full tournament support.

Tasks:
1. Tournament registration and lobby
2. Blind level timer with automatic increases
3. Table balancing for MTTs (move players when tables become uneven)
4. Bubble play handling
5. Final table consolidation
6. Prize distribution

### Phase 4: Database Improvements

**Priority: MEDIUM**

Current schema is minimal. Improvements needed:

1. **Hand History**: Store completed hands for replay/analysis
   ```sql
   CREATE TABLE hand_history (
       id TEXT PRIMARY KEY,
       table_id TEXT,
       hand_number INTEGER,
       actions JSON,  -- [{player, action, amount, timestamp}]
       board JSON,    -- Community cards
       results JSON,  -- Winners, amounts
       created_at TEXT
   );
   ```

2. **Session Tracking**: Track player sessions for statistics
3. **Transaction Ledger**: Full audit trail of chip movements
4. **Connection pooling**: Currently using single pool, may need per-request connections

### Phase 5: Frontend Refactoring

**Priority: LOW** (after backend stabilizes)

1. **State Management**: Consider Riverpod for more granular rebuilds
2. **Model Alignment**: Ensure Flutter models match Rust structs exactly
3. **Offline Support**: Queue actions when disconnected
4. **Animations**: Smoother card dealing, chip movements
5. **Tournament UI**: Registration, blind clock, prize pool display

### Phase 6: Scalability & DevOps

**Priority: LOW**

1. **Horizontal Scaling**: 
   - Stateless API servers behind load balancer
   - Table actors distributed via consistent hashing
   - Redis for session/presence data

2. **Monitoring**:
   - Prometheus metrics for game operations
   - Distributed tracing with OpenTelemetry
   - Error tracking with Sentry

3. **Testing**:
   - Integration tests for WebSocket flows
   - Load testing with k6 or similar
   - Fuzzing for game logic edge cases

---

## File Structure After Refactoring

```
backend/src/
├── main.rs
├── config.rs
├── error.rs              # App-level errors
├── api/
│   ├── mod.rs
│   ├── admin.rs
│   ├── auth.rs
│   ├── clubs.rs
│   └── tables.rs
├── auth/
│   ├── mod.rs
│   └── jwt.rs
├── db/
│   ├── mod.rs
│   ├── models.rs
│   └── migrations/
├── game/
│   ├── mod.rs
│   ├── betting.rs       # NEW: Separated betting logic
│   ├── constants.rs     # NEW: Centralized constants
│   ├── deck.rs
│   ├── error.rs         # NEW: Typed game errors
│   ├── format.rs        # NEW: Cash/SNG/MTT formats
│   ├── hand.rs
│   ├── player.rs
│   ├── pot.rs
│   ├── table.rs
│   └── variant.rs       # NEW: Holdem/Omaha variants
└── ws/
    ├── mod.rs
    ├── handler.rs
    └── messages.rs
```

---

## How to Continue

To pick up refactoring from here:

1. **Run backend tests**: `cd backend && cargo test`
2. **Run Flutter analyzer**: `cd poker_client && flutter analyze`
3. **Next tasks**:
   - ~~Implement Omaha hand evaluation (must use exactly 2 hole + 3 community)~~ ✅
   - ~~Add API endpoint to create tables with different variants/formats~~ ✅
   - ~~Update Flutter client to use new variant/format API endpoints~~ ✅
   - Implement SNG tournament flow (registration → play → payout)
   - Add table creation UI with variant/format selection (done in Commit 12)

The new infrastructure modules have comprehensive tests. When integrating, ensure existing tests continue to pass while adding new tests for variant-specific behavior.

---

*Last updated: January 30, 2026*
