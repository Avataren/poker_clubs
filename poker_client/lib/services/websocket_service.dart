import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/game_state.dart';
import '../models/tournament.dart';

class WebSocketService {
  WebSocketChannel? _channel;
  Function(GameState)? onGameStateUpdate;
  Function(String)? onError;
  Function()? onConnected;
  Function()? onClubUpdate;
  Function()? onGlobalUpdate;

  // Tournament event callbacks
  Function(String tournamentId, String tournamentName, String? tableId)?
  onTournamentStarted;
  Function(
    String tournamentId,
    int level,
    int smallBlind,
    int bigBlind,
    int ante,
  )?
  onTournamentBlindLevelIncreased;
  Function(String tournamentId, String username, int position, int prize)?
  onTournamentPlayerEliminated;
  Function(
    String tournamentId,
    String tournamentName,
    List<TournamentWinner> winners,
  )?
  onTournamentFinished;
  Function(String tournamentId, String tournamentName, String reason)?
  onTournamentCancelled;
  // Live tournament info broadcast (every second)
  Function(
    String tournamentId,
    String serverTime,
    int level,
    int smallBlind,
    int bigBlind,
    int ante,
    String levelStartTime,
    int levelDurationSecs,
    int levelTimeRemainingSecs,
    int? nextSmallBlind,
    int? nextBigBlind,
  )?
  onTournamentInfo;

  bool get isConnected => _channel != null;

  void connect(String token) {
    final wsUrl = 'ws://127.0.0.1:3000/ws?token=$token';
    _channel = WebSocketChannel.connect(Uri.parse(wsUrl));

    _channel!.stream.listen(
      (message) {
        _handleMessage(message);
      },
      onError: (error) {
        print('WebSocket error: $error');
        onError?.call(error.toString());
      },
      onDone: () {
        print('WebSocket closed');
        _channel = null;
      },
    );
  }

  void _handleMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      final type = data['type'];

      switch (type) {
        case 'Connected':
          print('Connected to server');
          onConnected?.call();
          break;

        case 'TableState':
          final gameState = GameState.fromJson(data['payload']);
          onGameStateUpdate?.call(gameState);
          break;

        case 'ClubUpdate':
          print('Club update broadcast received');
          onClubUpdate?.call();
          break;

        case 'GlobalUpdate':
          print('Global update broadcast received');
          onGlobalUpdate?.call();
          break;

        case 'Error':
          final errorMsg = data['payload']['message'];
          print('Server error: $errorMsg');
          onError?.call(errorMsg);
          break;

        case 'TournamentStarted':
          final payload = data['payload'];
          print('Tournament started: ${payload['tournament_name']}');
          onTournamentStarted?.call(
            payload['tournament_id'],
            payload['tournament_name'],
            payload['table_id'],
          );
          onGlobalUpdate?.call(); // Refresh tournament lists
          break;

        case 'TournamentBlindLevelIncreased':
          final payload = data['payload'];
          print('Tournament blind level increased: Level ${payload['level']}');
          onTournamentBlindLevelIncreased?.call(
            payload['tournament_id'],
            payload['level'],
            payload['small_blind'],
            payload['big_blind'],
            payload['ante'],
          );
          break;

        case 'TournamentPlayerEliminated':
          final payload = data['payload'];
          print(
            'Player eliminated: ${payload['username']} - Position ${payload['position']}',
          );
          onTournamentPlayerEliminated?.call(
            payload['tournament_id'],
            payload['username'],
            payload['position'],
            payload['prize'],
          );
          break;

        case 'TournamentFinished':
          final payload = data['payload'];
          print('Tournament finished: ${payload['tournament_name']}');
          final winners = (payload['winners'] as List)
              .map((w) => TournamentWinner.fromJson(w))
              .toList();
          onTournamentFinished?.call(
            payload['tournament_id'],
            payload['tournament_name'],
            winners,
          );
          onGlobalUpdate?.call(); // Refresh tournament lists
          break;

        case 'TournamentCancelled':
          final payload = data['payload'];
          print('Tournament cancelled: ${payload['tournament_name']}');
          onTournamentCancelled?.call(
            payload['tournament_id'],
            payload['tournament_name'],
            payload['reason'],
          );
          onGlobalUpdate?.call(); // Refresh tournament lists
          break;

        case 'TournamentInfo':
          final payload = data['payload'];
          onTournamentInfo?.call(
            payload['tournament_id'],
            payload['server_time'],
            payload['level'],
            payload['small_blind'],
            payload['big_blind'],
            payload['ante'],
            payload['level_start_time'],
            payload['level_duration_secs'],
            payload['level_time_remaining_secs'],
            payload['next_small_blind'],
            payload['next_big_blind'],
          );
          break;

        default:
          print('Unknown message type: $type');
      }
    } catch (e) {
      print('Error parsing message: $e');
      onError?.call(e.toString());
    }
  }

  void joinTable(String tableId, int buyin) {
    if (_channel == null) {
      throw Exception('Not connected');
    }

    final message = jsonEncode({
      'type': 'JoinTable',
      'payload': {'table_id': tableId, 'buyin': buyin},
    });

    _channel!.sink.add(message);
  }

  void playerAction(String action, {int? amount}) {
    if (_channel == null) {
      throw Exception('Not connected');
    }

    dynamic actionPayload;

    if (action == 'Raise' && amount != null) {
      actionPayload = {'action': 'Raise', 'amount': amount};
    } else {
      actionPayload = {'action': action};
    }

    final message = jsonEncode({
      'type': 'PlayerAction',
      'payload': {'action': actionPayload},
    });

    _channel!.sink.add(message);
  }

  void leaveTable() {
    if (_channel == null) return;

    final message = jsonEncode({'type': 'LeaveTable'});
    _channel!.sink.add(message);
  }

  void viewingClubsList() {
    if (_channel == null) return;

    final message = jsonEncode({'type': 'ViewingClubsList'});
    _channel!.sink.add(message);
  }

  void viewingClub(String clubId) {
    if (_channel == null) return;

    final message = jsonEncode({
      'type': 'ViewingClub',
      'payload': {'club_id': clubId},
    });
    _channel!.sink.add(message);
  }

  void leavingView() {
    if (_channel == null) return;

    final message = jsonEncode({'type': 'LeavingView'});
    _channel!.sink.add(message);
  }

  void takeSeat(String tableId, int seatNumber, int buyin) {
    if (_channel == null) {
      throw Exception('Not connected');
    }

    final message = jsonEncode({
      'type': 'TakeSeat',
      'payload': {'table_id': tableId, 'seat': seatNumber, 'buyin': buyin},
    });

    _channel!.sink.add(message);
  }

  void standUp() {
    if (_channel == null) {
      throw Exception('Not connected');
    }

    final message = jsonEncode({'type': 'StandUp'});
    _channel!.sink.add(message);
  }

  void topUp(int amount) {
    if (_channel == null) {
      throw Exception('Not connected');
    }

    final message = jsonEncode({
      'type': 'TopUp',
      'payload': {'amount': amount},
    });
    _channel!.sink.add(message);
  }

  void addBot(String tableId, {String? name, String? strategy}) {
    if (_channel == null) return;

    final payload = <String, dynamic>{'table_id': tableId};
    if (name != null) payload['name'] = name;
    if (strategy != null) payload['strategy'] = strategy;

    final message = jsonEncode({'type': 'AddBot', 'payload': payload});
    _channel!.sink.add(message);
  }

  void removeBot(String tableId, String botUserId) {
    if (_channel == null) return;

    final message = jsonEncode({
      'type': 'RemoveBot',
      'payload': {'table_id': tableId, 'bot_user_id': botUserId},
    });
    _channel!.sink.add(message);
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }
}
