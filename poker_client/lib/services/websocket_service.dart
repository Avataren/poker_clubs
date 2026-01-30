import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/game_state.dart';

class WebSocketService {
  WebSocketChannel? _channel;
  Function(GameState)? onGameStateUpdate;
  Function(String)? onError;
  Function()? onConnected;
  Function()? onClubUpdate;
  Function()? onGlobalUpdate;

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

    final message = jsonEncode({
      'type': 'AddBot',
      'payload': payload,
    });
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
