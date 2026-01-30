# Flutter Poker Client

A cross-platform Texas Hold'em poker client built with Flutter.

## Features

- User authentication (register/login)
- Create and join poker clubs
- Create and join cash game tables
- Real-time Texas Hold'em poker gameplay
- WebSocket-based communication for live game updates
- Responsive UI for web, iOS, and Android

## Prerequisites

- Flutter SDK (3.0 or higher)
- Dart SDK
- Running backend server (default: http://127.0.0.1:3000)

## Dependencies

- `http`: REST API communication
- `web_socket_channel`: WebSocket communication for real-time game updates
- `provider`: State management

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── models/                   # Data models
│   ├── card.dart            # Poker card model
│   ├── club.dart            # Club and table models
│   ├── game_state.dart      # Game state model
│   └── player.dart          # Player model
├── screens/                 # UI screens
│   ├── clubs_screen.dart    # View and create clubs
│   ├── game_screen.dart     # Main poker game interface
│   ├── login_screen.dart    # Login/register screen
│   └── tables_screen.dart   # View and create tables
├── services/                # Business logic and API
│   ├── api_service.dart     # REST API client
│   └── websocket_service.dart # WebSocket client
└── widgets/                 # Reusable UI components
    ├── card_widget.dart     # Playing card widget
    └── player_widget.dart   # Player info widget
```

## Configuration

Update the API base URL in `lib/services/api_service.dart`:

```dart
static const String baseUrl = 'http://127.0.0.1:3000';
```

Update the WebSocket URL in `lib/services/websocket_service.dart`:

```dart
static const String wsUrl = 'ws://127.0.0.1:3000/ws';
```

## Running the App

### Web
```bash
flutter run -d chrome
```

### iOS Simulator
```bash
flutter run -d iPhone
```

### Android Emulator
```bash
flutter run -d emulator-5554
```

### List available devices
```bash
flutter devices
```

## Gameplay Flow

1. **Register/Login**: Create an account or login with existing credentials
2. **Create Club**: Create a new poker club (you become the admin)
3. **Create Table**: Set up a cash game table with blinds
4. **Join Table**: Join a table with your buy-in amount
5. **Play Poker**: Play Texas Hold'em with real-time updates
6. **Actions**: Fold, Check, Call, Raise, or go All-In
7. **Leave Table**: Cash out and leave the table when done

## WebSocket Events

The client handles these real-time events:

- `GameState`: Full game state update (cards, pot, players, current turn)
- `TableUpdate`: Player joined/left notifications
- `Error`: Error messages from the server

## State Management

Uses Provider for dependency injection and state management:

- `ApiService`: Injected at app root, provides authentication and REST API access
- Game state managed locally in `GameScreen` via WebSocket updates

## Models

### Card
- `rank`: 2-14 (Jack=11, Queen=12, King=13, Ace=14)
- `suit`: 0-3 (Clubs, Diamonds, Hearts, Spades)

### Player
- Contains username, stack, current bet, hole cards, and state

### GameState
- Current game phase (Waiting, PreFlop, Flop, Turn, River, Showdown)
- Community cards, pot total, current bet
- All players at the table
- Current player's turn

## Building for Production

### Web
```bash
flutter build web
```

### iOS
```bash
flutter build ios
```

### Android
```bash
flutter build apk
```

## Troubleshooting

### WebSocket connection fails
- Ensure backend server is running
- Check WebSocket URL in `websocket_service.dart`
- Verify JWT token is being sent in connection query parameter

### API calls fail
- Verify backend server is running on correct port
- Check base URL in `api_service.dart`
- Ensure CORS is enabled on backend

### Cards not displaying
- Check that `PokerCard.toString()` is properly formatting card strings
- Verify WebSocket `GameState` messages contain valid card data

## License

MIT
