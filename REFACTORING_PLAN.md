# Poker Project Refactoring Plan

Incremental refactoring for maintainability and readability.
Each section is a self-contained, compile-green chunk.

**Status: All 12 tasks complete.**

## Backend

### B1. Extract `current_timestamp_ms()` helper ✅
- Replaced 3 identical `SystemTime` unwrap chains with helper in `mod.rs`
- Files: `phase.rs`, `showdown.rs`

### B2. Add `try_transition()` helper ✅
- Replaced 8 repeated phase transition blocks
- Files: `phase.rs`, `dealing.rs`

### B3. Extract `player_bets()` helper ✅
- Replaced duplicated 7-line iterator chains
- Files: `showdown.rs`, `state.rs`

### B4. Centralize player eligibility predicates ✅
- Added `is_eligible_for_button()`, `is_eligible_for_tournament_blind()` to Player
- Replaced inline closures in `mod.rs`, `blinds.rs`, `player.rs`

### B5. Extract magic numbers to constants ✅
- `MAX_RAISES_PER_ROUND = 4`, `HEADS_UP_PLAYER_COUNT = 2`
- Files: `constants.rs`, `actions.rs`, `blinds.rs`

### B6. Split `check_auto_advance()` ✅
- Extracted `check_tournament_waiting()` and `advance_after_showdown()`
- File: `phase.rs`

## Client

### C1. Extract `FormatUtils` shared utility ✅
- `formatChips()`, `formatRelativeTime()`, `formatCountdown()`, `formatAbsolute()`
- New file: `lib/utils/format_utils.dart`

### C2. Extract `TournamentInfoState` model ✅
- Pulled 11 tournament state variables out of `game_screen.dart`
- New file: `lib/models/tournament_info_state.dart`

### C3. Extract reusable dialog widgets ✅
- `ConfirmationDialog` and `InputDialog`
- Deduplicated 7 dialog patterns across 2 screens
- New file: `lib/widgets/dialogs.dart`

### C4. Extract shared `TournamentStatusBadge` widget ✅
- Deduplicated `_buildStatusBadge()` from 2 files with compact mode
- New file: `lib/widgets/tournament_status_badge.dart`

### C5. Centralize responsive breakpoints ✅
- `Breakpoints.mobile`, `compact`, `tablet`
- New file: `lib/constants.dart`

### C6. Split game_screen.dart build methods ✅
- Extracted `_buildGameHeader()`, `_buildShowCardsSection()`, `_buildActionButtons()`
- Same file, focused private methods

## Order of execution
B1 → B2 → B3 → B5 → B4 → B6 → C1 → C2 → C5 → C3 → C4 → C6
