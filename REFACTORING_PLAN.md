# Poker Project Refactoring Plan

Incremental refactoring for maintainability and readability.
Each section is a self-contained, compile-green chunk.

## Backend

### B1. Extract `current_timestamp_ms()` helper
- Replace 3 identical `SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64` calls
- Files: `phase.rs`, `showdown.rs`
- Add to a new `game/util.rs` with proper error handling (no unwrap)

### B2. Add `transition_phase()` helper
- Replace 8 repeated `if let Ok(next) = self.phase.transition_to(X) { self.phase = next; }` blocks
- Files: `phase.rs`, `dealing.rs`
- Add method to `PokerTable` in `phase.rs`

### B3. Extract `build_player_bets()` helper
- Replace 3 duplicated 7-line iterator chains building `Vec<(usize, i64, bool)>`
- File: `showdown.rs`

### B4. Centralize player eligibility predicates
- Add `is_eligible_for_action()`, `is_eligible_for_blind()`, `is_eligible_for_button()` to Player
- Replace inline filter closures in `mod.rs`, `blinds.rs`, `dealing.rs`

### B5. Extract magic numbers to constants
- `MAX_RAISES_PER_ROUND = 4` in `actions.rs`
- `HEADS_UP_PLAYER_COUNT = 2` used in `blinds.rs`
- Move to `game/constants.rs`

### B6. Split `check_auto_advance()`
- Extract `check_tournament_waiting_window()` and `check_street_delay()` from the 106-line function
- File: `phase.rs`

## Client

### C1. Extract `FormatUtils` shared utility
- `formatChips()` — deduplicate from 4 files
- `formatRelativeTime()` — deduplicate from 3 files
- New file: `lib/utils/format_utils.dart`

### C2. Extract `TournamentInfoState` model
- Pull 11 tournament state variables out of `game_screen.dart`
- New file: `lib/models/tournament_info_state.dart`

### C3. Extract reusable dialog widgets
- `ConfirmationDialog` and `InputDialog`
- Reduce 8+ identical dialog patterns
- New file: `lib/widgets/dialogs.dart`

### C4. Extract shared tournament widgets
- `StatusBadge` — deduplicate from 2 files
- `InfoRow` — deduplicate from 3 files
- New file: `lib/widgets/shared/status_badge.dart`, `info_row.dart`

### C5. Centralize responsive breakpoints
- Replace 5 hardcoded breakpoint values with `ResponsiveBreakpoints` constants
- New file: `lib/constants.dart`

### C6. Split game_screen.dart build methods
- Extract `_buildGameHeader()`, `_buildShowCardsSection()`, `_buildActionButtons()`
- Keep in same file but as focused private methods

## Order of execution
B1 → B2 → B3 → B5 → B4 → B6 → C1 → C2 → C5 → C3 → C4 → C6
