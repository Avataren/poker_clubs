import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../config.dart';
import '../models/game_state.dart';
import '../models/tournament.dart';

enum ConnectionStatus { disconnected, connecting, connected, reconnecting }

class WebSocketService {
  WebSocketChannel? _channel;
  String? _token;
  String _serverHost = AppConfig.serverHost;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectDelaySecs = 30;
  bool _intentionalDisconnect = false;
  ConnectionStatus _connectionStatus = ConnectionStatus.disconnected;

  // Last known table/club for re-subscribing after reconnect
  String? _lastTableId;
  int? _lastBuyin;
  String? _lastClubId;

  Function(GameState)? onGameStateUpdate;
  Function(String)? onError;
  Function()? onConnected;
  Function()? onClubUpdate;
  Function()? onGlobalUpdate;
  Function(ConnectionStatus)? onConnectionStatusChanged;

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
  Function(String tournamentId, String tableId, String userId)?
  onTournamentTableChanged;
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

  bool get isConnected => _connectionStatus == ConnectionStatus.connected;
  ConnectionStatus get connectionStatus => _connectionStatus;

  void setServerHost(String host) {
    _serverHost = host;
  }

  void _setConnectionStatus(ConnectionStatus status) {
    if (_connectionStatus != status) {
      _connectionStatus = status;
      onConnectionStatusChanged?.call(status);
    }
  }

  void connect(String token) {
    _token = token;
    _intentionalDisconnect = false;
    _reconnectAttempts = 0;
    _connectInternal();
  }

  void _connectInternal() {
    _setConnectionStatus(
      _reconnectAttempts > 0
          ? ConnectionStatus.reconnecting
          : ConnectionStatus.connecting,
    );

    final wsUrl = 'ws://$_serverHost/ws?token=$_token';
    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
    } catch (e) {
      _setConnectionStatus(ConnectionStatus.disconnected);
      _scheduleReconnect();
      return;
    }

    _channel!.stream.listen(
      (message) {
        _handleMessage(message);
      },
      onError: (error) {
        onError?.call(error.toString());
        _onDisconnected();
      },
      onDone: () {
        _onDisconnected();
      },
    );
  }

  void _onDisconnected() {
    _channel = null;
    _setConnectionStatus(ConnectionStatus.disconnected);
    if (!_intentionalDisconnect) {
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    final delaySecs = min(
      pow(2, _reconnectAttempts).toInt(),
      _maxReconnectDelaySecs,
    );
    _reconnectAttempts++;
    _reconnectTimer = Timer(Duration(seconds: delaySecs), () {
      if (!_intentionalDisconnect && _token != null) {
        _connectInternal();
      }
    });
  }

  void _onReconnected() {
    // Re-subscribe to the last table/club if we had one
    if (_lastTableId != null) {
      joinTable(_lastTableId!, _lastBuyin ?? 0);
    } else if (_lastClubId != null) {
      viewingClub(_lastClubId!);
    }
  }

  void _handleMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      final type = data['type'];

      switch (type) {
        case 'Connected':
          _setConnectionStatus(ConnectionStatus.connected);
          final wasReconnect = _reconnectAttempts > 0;
          _reconnectAttempts = 0;
          onConnected?.call();
          if (wasReconnect) {
            _onReconnected();
          }
          break;

        case 'TableState':
          final gameState = GameState.fromJson(data['payload']);
          onGameStateUpdate?.call(gameState);
          break;

        case 'ClubUpdate':
          onClubUpdate?.call();
          break;

        case 'GlobalUpdate':
          onGlobalUpdate?.call();
          break;

        case 'Pong':
          // Server responded to our application-level ping
          break;

        case 'Error':
          final errorMsg = data['payload']['message'];
          onError?.call(errorMsg);
          break;

        case 'TournamentStarted':
          final payload = data['payload'];
          onTournamentStarted?.call(
            payload['tournament_id'],
            payload['tournament_name'],
            payload['table_id'],
          );
          onGlobalUpdate?.call();
          break;

        case 'TournamentBlindLevelIncreased':
          final payload = data['payload'];
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
          onTournamentPlayerEliminated?.call(
            payload['tournament_id'],
            payload['username'],
            payload['position'],
            payload['prize'],
          );
          break;

        case 'TournamentFinished':
          final payload = data['payload'];
          final winners = (payload['winners'] as List)
              .map((w) => TournamentWinner.fromJson(w))
              .toList();
          onTournamentFinished?.call(
            payload['tournament_id'],
            payload['tournament_name'],
            winners,
          );
          onGlobalUpdate?.call();
          break;

        case 'TournamentCancelled':
          final payload = data['payload'];
          onTournamentCancelled?.call(
            payload['tournament_id'],
            payload['tournament_name'],
            payload['reason'],
          );
          onGlobalUpdate?.call();
          break;

        case 'TournamentTableChanged':
          final payload = data['payload'];
          onTournamentTableChanged?.call(
            payload['tournament_id'],
            payload['table_id'],
            payload['user_id'],
          );
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
          break;
      }
    } catch (e) {
      onError?.call(e.toString());
    }
  }

  void _send(Map<String, dynamic> message) {
    _channel?.sink.add(jsonEncode(message));
  }

  void joinTable(String tableId, int buyin) {
    _lastTableId = tableId;
    _lastBuyin = buyin;
    _send({
      'type': 'JoinTable',
      'payload': {'table_id': tableId, 'buyin': buyin},
    });
  }

  void playerAction(String action, {int? amount}) {
    if (_channel == null) return;

    dynamic actionPayload;
    if (action == 'Raise' && amount != null) {
      actionPayload = {'action': 'Raise', 'amount': amount};
    } else {
      actionPayload = {'action': action};
    }

    _send({
      'type': 'PlayerAction',
      'payload': {'action': actionPayload},
    });
  }

  void leaveTable() {
    _lastTableId = null;
    _lastBuyin = null;
    _send({'type': 'LeaveTable'});
  }

  void viewingClubsList() {
    _lastClubId = null;
    _send({'type': 'ViewingClubsList'});
  }

  void viewingClub(String clubId) {
    _lastClubId = clubId;
    _send({
      'type': 'ViewingClub',
      'payload': {'club_id': clubId},
    });
  }

  void leavingView() {
    _lastClubId = null;
    _send({'type': 'LeavingView'});
  }

  void takeSeat(String tableId, int seatNumber, int buyin) {
    _lastTableId = tableId;
    _lastBuyin = buyin;
    _send({
      'type': 'TakeSeat',
      'payload': {'table_id': tableId, 'seat': seatNumber, 'buyin': buyin},
    });
  }

  void standUp() {
    _send({'type': 'StandUp'});
  }

  void topUp(int amount) {
    _send({
      'type': 'TopUp',
      'payload': {'amount': amount},
    });
  }

  void addBot(String tableId, {String? name, String? strategy}) {
    final payload = <String, dynamic>{'table_id': tableId};
    if (name != null) payload['name'] = name;
    if (strategy != null) payload['strategy'] = strategy;
    _send({'type': 'AddBot', 'payload': payload});
  }

  void showCards(List<int> cardIndices) {
    _send({
      'type': 'ShowCards',
      'payload': {'card_indices': cardIndices},
    });
  }

  void removeBot(String tableId, String botUserId) {
    _send({
      'type': 'RemoveBot',
      'payload': {'table_id': tableId, 'bot_user_id': botUserId},
    });
  }

  void disconnect() {
    _intentionalDisconnect = true;
    _reconnectTimer?.cancel();
    _channel?.sink.close();
    _channel = null;
    _lastTableId = null;
    _lastBuyin = null;
    _lastClubId = null;
    _setConnectionStatus(ConnectionStatus.disconnected);
  }
}
