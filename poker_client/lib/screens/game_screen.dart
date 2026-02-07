import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/club.dart';
import '../models/game_state.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import '../services/sound_service.dart';
import '../widgets/card_widget.dart';
import '../widgets/table_seat_widget.dart';

class GameScreen extends StatefulWidget {
  final PokerTable table;

  const GameScreen({super.key, required this.table});

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  late WebSocketService _wsService;
  final _soundService = SoundService();
  GameState? _gameState;
  GameState? _previousGameState;
  String _statusMessage = 'Connecting...';
  final _raiseController = TextEditingController(text: '200');
  final _buyinController = TextEditingController(text: '5000');
  final _topUpController = TextEditingController(text: '5000');
  bool _isSeated = false;

  // Tournament info from live broadcast
  String? _tournamentId;
  String? _serverTime;
  int? _tournamentLevel;
  int? _tournamentSmallBlind;
  int? _tournamentBigBlind;
  int? _tournamentAnte;
  String? _levelStartTime;
  int? _levelDurationSecs;
  int? _levelTimeRemainingSecs; // Server-calculated remaining time
  int? _nextSmallBlind;
  int? _nextBigBlind;

  @override
  void initState() {
    super.initState();
    _wsService = WebSocketService();
    _wsService.onGameStateUpdate = (gameState) {
      _playActionSounds(gameState);
      print('DEBUG: GameState tournamentId = ${gameState.tournamentId}');
      setState(() {
        _previousGameState = _gameState;
        _gameState = gameState;
        _updateStatusMessage();
        _checkIfSeated();
      });
    };
    _wsService.onTournamentInfo =
        (
          tournamentId,
          serverTime,
          level,
          smallBlind,
          bigBlind,
          ante,
          levelStartTime,
          levelDurationSecs,
          levelTimeRemainingSecs,
          nextSmallBlind,
          nextBigBlind,
        ) {
          print(
            'DEBUG: TournamentInfo received - Level $level, Blinds $smallBlind/$bigBlind, Remaining: ${levelTimeRemainingSecs}s',
          );
          setState(() {
            _tournamentId = tournamentId;
            _serverTime = serverTime;
            _tournamentLevel = level;
            _tournamentSmallBlind = smallBlind;
            _tournamentBigBlind = bigBlind;
            _tournamentAnte = ante;
            _levelStartTime = levelStartTime;
            _levelDurationSecs = levelDurationSecs;
            _levelTimeRemainingSecs = levelTimeRemainingSecs;
            _nextSmallBlind = nextSmallBlind;
            _nextBigBlind = nextBigBlind;
          });
        };
    _wsService.onError = (error) {
      setState(() => _statusMessage = 'Error: $error');
    };
    _wsService.onConnected = () {
      setState(() => _statusMessage = 'Connected - Choose a seat');
      // Join table as observer (without seat)
      _wsService.joinTable(widget.table.id, 0);
    };

    final token = context.read<ApiService>().token!;
    _wsService.connect(token);
  }

  void _checkIfSeated() {
    final myUserId = context.read<ApiService>().userId;
    _isSeated = _gameState?.players.any((p) => p.userId == myUserId) ?? false;
  }

  void _updateStatusMessage() {
    if (_gameState == null) return;

    final myUserId = context.read<ApiService>().userId;
    final myPlayer = _gameState!.players
        .where((p) => p.userId == myUserId)
        .firstOrNull;

    if (myPlayer == null) {
      setState(() => _statusMessage = '');
      return;
    }

    // Check if game is waiting for more players
    if (_gameState!.phase.toLowerCase() == 'waiting') {
      setState(() => _statusMessage = 'Waiting for players...');
      return;
    }

    final isMyTurn = _gameState!.currentPlayer?.userId == myPlayer.userId;

    // Debug logging
    print('DEBUG: currentPlayerSeat=${_gameState!.currentPlayerSeat}');
    print('DEBUG: players.length=${_gameState!.players.length}');
    for (var p in _gameState!.players) {
      print('DEBUG: player ${p.username} seat=${p.seat} userId=${p.userId}');
    }
    print(
      'DEBUG: currentPlayer=${_gameState!.currentPlayer?.username ?? "NULL"}',
    );
    print('DEBUG: myPlayer=${myPlayer.username} userId=${myPlayer.userId}');
    print('DEBUG: isMyTurn=$isMyTurn');

    setState(() {
      _statusMessage = isMyTurn ? 'Your turn!' : 'Waiting for your turn...';
    });
  }

  void _playActionSounds(GameState newState) {
    // Play shuffle sound when entering PreFlop phase
    if (_previousGameState?.phase != 'PreFlop' && newState.phase == 'PreFlop') {
      print('DEBUG: Playing shuffle sound - entering PreFlop');
    }

    // Detect player actions by comparing states
    if (_previousGameState == null) return;

    for (final player in newState.players) {
      final prevPlayer = _previousGameState!.players
          .where((p) => p.userId == player.userId)
          .firstOrNull;

      if (prevPlayer == null) continue;

      // Detect top-up (stack increased while not in a betting round)
      final stackIncreased =
          player.stack > prevPlayer.stack &&
          player.currentBet == prevPlayer.currentBet;
      if (stackIncreased) {
        _soundService.playGameStart();
        continue; // Skip other sound checks for this player
      }

      // Check if player took an action (lastAction changed or currentBet changed)
      final actionChanged = player.lastAction != prevPlayer.lastAction;
      final betChanged = player.currentBet != prevPlayer.currentBet;

      if (actionChanged || betChanged) {
        final action = player.lastAction?.toLowerCase() ?? '';

        // Play check sound
        if (action.contains('check')) {
          print('Playing CHECK sound for ${player.username}');
          _soundService.playCheck();
        }
        // Play fold sound
        else if (action.contains('fold')) {
          print('Playing FOLD sound for ${player.username}');
          _soundService.playFold();
        }
        // Play all-in sound
        else if (player.isAllIn && !prevPlayer.isAllIn) {
          print('Playing ALL-IN sound for ${player.username}');
          _soundService.playAllIn();
        }
        // Play chip sounds for bet/call/raise
        else if (betChanged && player.currentBet > prevPlayer.currentBet) {
          final betAmount = player.currentBet - prevPlayer.currentBet;
          print(
            'Playing CHIP sound for ${player.username}: \$$betAmount (action: ${player.lastAction})',
          );
          _soundService.playChipBet(betAmount);
        }
      }
    }
  }

  void _takeSeat(int seatNumber) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Take Seat ${seatNumber + 1}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Enter buy-in amount:'),
            const SizedBox(height: 8),
            TextField(
              controller: _buyinController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                hintText: 'Buy-in amount',
                prefixText: '\$',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final amount = int.tryParse(_buyinController.text) ?? 5000;
              _wsService.takeSeat(widget.table.id, seatNumber, amount);
              Navigator.pop(context);
            },
            child: const Text('Take Seat'),
          ),
        ],
      ),
    );
  }

  void _standUp() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Stand Up'),
        content: const Text(
          'Are you sure you want to leave your seat?\n\nYou will leave immediately if not in a hand, or after the current hand concludes.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              _wsService.standUp();
              Navigator.pop(context);
            },
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('Stand Up'),
          ),
        ],
      ),
    );
  }

  void _topUp() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Top Up'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Add chips to your stack:'),
            const SizedBox(height: 8),
            TextField(
              controller: _topUpController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                hintText: 'Top-up amount',
                prefixText: '\$',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final amount = int.tryParse(_topUpController.text) ?? 5000;
              _wsService.topUp(amount);
              Navigator.pop(context);
            },
            child: const Text('Top Up'),
          ),
        ],
      ),
    );
  }

  void _showAddBotDialog() {
    String selectedStrategy = 'balanced';

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Add Bot'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Select bot strategy:'),
              const SizedBox(height: 8),
              DropdownButton<String>(
                value: selectedStrategy,
                isExpanded: true,
                items: const [
                  DropdownMenuItem(value: 'balanced', child: Text('Balanced')),
                  DropdownMenuItem(value: 'tight', child: Text('Tight')),
                  DropdownMenuItem(
                    value: 'aggressive',
                    child: Text('Aggressive'),
                  ),
                  DropdownMenuItem(
                    value: 'calling_station',
                    child: Text('Calling Station'),
                  ),
                ],
                onChanged: (value) {
                  setDialogState(() => selectedStrategy = value!);
                },
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                _wsService.addBot(widget.table.id, strategy: selectedStrategy);
                Navigator.pop(context);
              },
              child: const Text('Add Bot'),
            ),
          ],
        ),
      ),
    );
  }

  void _removeBot(String botUserId) {
    _wsService.removeBot(widget.table.id, botUserId);
  }

  void _playerAction(String action, {int? amount}) {
    try {
      _wsService.playerAction(action, amount: amount);
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  Widget _buildTableStack({
    required BoxConstraints constraints,
    required String? myUserId,
    required bool isShowdown,
  }) {
    // Calculate responsive table dimensions - match table widget sizing
    final maxWidth = constraints.maxWidth * 0.75;
    final maxHeight = constraints.maxHeight * 0.75;

    double tableWidth, tableHeight;
    if (maxWidth / maxHeight > 1.6) {
      tableHeight = maxHeight;
      tableWidth = tableHeight * 1.6;
    } else {
      tableWidth = maxWidth;
      tableHeight = tableWidth / 1.6;
    }

    // Calculate responsive positioning for overlay elements
    // Position community cards higher in the center area
    final cardTop =
        (constraints.maxHeight - tableHeight) / 2 + tableHeight * 0.25;

    // Calculate card dimensions with proper aspect ratio (1.4:1)
    final cardWidth = (tableWidth * 0.08).clamp(40.0, 60.0);
    final cardHeight = cardWidth * 1.4;

    return Stack(
      alignment: Alignment.center,
      children: [
        // The table layout with seats around it
        PokerTableWidget(
          maxSeats: _gameState!.maxSeats,
          players: _gameState!.players,
          currentPlayerSeat: _gameState!.currentPlayerSeat,
          myUserId: myUserId,
          onTakeSeat: !_isSeated ? _takeSeat : null,
          onRemoveBot: _removeBot,
          showingDown: isShowdown,
          gamePhase: _gameState!.phase,
          winningHand: _gameState!.winningHand,
          dealerSeat: _gameState!.dealerSeat,
          smallBlindSeat: _gameState!.smallBlindSeat,
          bigBlindSeat: _gameState!.bigBlindSeat,
          smallBlind: widget.table.smallBlind,
          potTotal: _gameState!.potTotal,
          pots: _gameState!.pots,
        ),

        // Table center overlay (community cards) - responsive positioning
        Positioned(
          top: cardTop,
          left: 0,
          right: 0,
          child: IgnorePointer(
            child: Container(
              constraints: BoxConstraints(
                maxWidth: tableWidth * 0.8,
                maxHeight: tableHeight * 0.5,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Community cards - responsive sizing with proper aspect ratio
                  // Always reserve space to prevent layout shifts
                  SizedBox(
                    height: cardHeight + 16, // Card height + padding
                    child: _gameState!.communityCards.isNotEmpty
                        ? Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Colors.black26,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: _gameState!.communityCards
                                  .map(
                                    (card) => CardWidget(
                                      card: card,
                                      width: cardWidth,
                                      height: cardHeight,
                                      isShowdown: isShowdown,
                                    ),
                                  )
                                  .toList(),
                            ),
                          )
                        : const SizedBox.shrink(),
                  ),
                  // Tournament info (if tournament table)
                  if (_tournamentId != null || _gameState!.tournamentId != null)
                    SizedBox(height: tableHeight * 0.02),
                  if (_tournamentId != null || _gameState!.tournamentId != null)
                    _buildTournamentInfo(),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final myUserId = context.read<ApiService>().userId;
    final myPlayer = _gameState?.players
        .where((p) => p.userId == myUserId)
        .firstOrNull;
    final isMyTurn =
        myPlayer != null && _gameState?.currentPlayer?.userId == myUserId;
    final isShowdown = _gameState?.phase.toLowerCase() == 'showdown';
    final canTopUp = _gameState?.canTopUp ?? true;
    final isPortrait = MediaQuery.of(context).orientation == Orientation.portrait;

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.table.name),
        backgroundColor: Colors.grey[850],
        actions: [
          // Only show top-up button if format allows it
          if (_isSeated &&
              canTopUp &&
              myPlayer != null &&
              myPlayer.stack < 10000)
            IconButton(
              icon: const Icon(Icons.add_circle),
              tooltip: 'Top Up',
              onPressed: _topUp,
            ),
          if (_isSeated)
            IconButton(
              icon: const Icon(Icons.event_seat),
              tooltip: 'Stand Up',
              onPressed: _standUp,
            ),
          IconButton(
            icon: const Icon(Icons.smart_toy),
            tooltip: 'Add Bot',
            onPressed: _showAddBotDialog,
          ),
          IconButton(
            icon: const Icon(Icons.exit_to_app),
            onPressed: () {
              _wsService.leaveTable();
              _wsService.disconnect();
              Navigator.pop(context);
            },
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [const Color(0xFF1a1a2e), const Color(0xFF16213e)],
          ),
        ),
        child: Column(
          children: [
            // Phase and game type indicator
            Container(
              padding: const EdgeInsets.all(8),
              child: Column(
                children: [
                  // Variant and format info
                  if (_gameState != null)
                    Text(
                      _gameState!.gameTypeDescription,
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 14,
                      ),
                    ),
                  const SizedBox(height: 4),
                  Text(
                    _gameState?.phase.toUpperCase() ?? 'WAITING',
                    style: const TextStyle(
                      color: Colors.amber,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),

            // Poker table with seats
            Expanded(
              child: Center(
                child: _gameState != null
                    ? LayoutBuilder(
                        builder: (context, constraints) {
                          if (!isPortrait) {
                            return _buildTableStack(
                              constraints: constraints,
                              myUserId: myUserId,
                              isShowdown: isShowdown,
                            );
                          }

                          return FittedBox(
                            fit: BoxFit.contain,
                            child: SizedBox(
                              width: constraints.maxHeight,
                              height: constraints.maxWidth,
                              child: RotatedBox(
                                quarterTurns: 1,
                                child: LayoutBuilder(
                                  builder: (context, rotatedConstraints) {
                                    return _buildTableStack(
                                      constraints: rotatedConstraints,
                                      myUserId: myUserId,
                                      isShowdown: isShowdown,
                                    );
                                  },
                                ),
                              ),
                            ),
                          );
                        },
                      )
                    : const Center(
                        child: CircularProgressIndicator(color: Colors.white),
                      ),
              ),
            ),

            // Status message - only show if there's a message
            if (_statusMessage.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(12),
                color: isMyTurn ? Colors.amber[700] : Colors.black45,
                child: Text(
                  _statusMessage,
                  style: TextStyle(
                    color: isMyTurn ? Colors.black : Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),

            // Show Cards buttons (fold-win showdown: winner can reveal cards)
            if (_isSeated &&
                isShowdown &&
                myPlayer != null &&
                myPlayer.isWinner &&
                !myPlayer.isBot &&
                myPlayer.shownCards != null &&
                _gameState?.winningHand == null)
              Container(
                padding: const EdgeInsets.all(12),
                color: Colors.black45,
                child: Wrap(
                  spacing: 8,
                  alignment: WrapAlignment.center,
                  children: [
                    for (int i = 0; i < (myPlayer.holeCards?.length ?? 0); i++)
                      if (myPlayer.shownCards != null &&
                          i < myPlayer.shownCards!.length &&
                          !myPlayer.shownCards![i])
                        ElevatedButton(
                          onPressed: () => _wsService.showCards([i]),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.teal,
                          ),
                          child: Text('Show Card ${i + 1}'),
                        ),
                    if (myPlayer.shownCards != null &&
                        myPlayer.shownCards!.any((s) => !s))
                      ElevatedButton(
                        onPressed: () {
                          final indices = <int>[];
                          for (int i = 0; i < myPlayer.shownCards!.length; i++) {
                            if (!myPlayer.shownCards![i]) indices.add(i);
                          }
                          _wsService.showCards(indices);
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.amber[700],
                        ),
                        child: const Text('Show All'),
                      ),
                  ],
                ),
              ),

            // Action buttons (only show if seated and it's my turn)
            if (_isSeated && isMyTurn)
              Container(
                padding: const EdgeInsets.all(16),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  alignment: WrapAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: () => _playerAction('Fold'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.red,
                      ),
                      child: const Text('Fold'),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction('Check'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                      ),
                      child: const Text('Check'),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction('Call'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                      ),
                      child: const Text('Call'),
                    ),
                    SizedBox(
                      width: 100,
                      child: TextField(
                        controller: _raiseController,
                        keyboardType: TextInputType.number,
                        style: const TextStyle(color: Colors.white),
                        decoration: const InputDecoration(
                          hintText: 'Amount',
                          hintStyle: TextStyle(color: Colors.white54),
                          filled: true,
                          fillColor: Colors.black26,
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 8,
                          ),
                        ),
                      ),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction(
                        'Raise',
                        amount: int.tryParse(_raiseController.text),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                      ),
                      child: const Text('Raise'),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction('AllIn'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.purple,
                      ),
                      child: const Text('All In'),
                    ),
                  ],
                ),
              )
            else if (_isSeated)
              Container(
                padding: const EdgeInsets.all(16),
                child: const Text(
                  'Waiting for your turn...',
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
              )
            else
              Container(
                padding: const EdgeInsets.all(16),
                child: const Text(
                  'Click on an empty seat to join the game',
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTournamentInfo() {
    // Use live tournament info from broadcast if available, otherwise use gameState data
    final tournamentId = _tournamentId ?? _gameState?.tournamentId;
    if (tournamentId == null) {
      return const SizedBox.shrink();
    }

    final level =
        (_tournamentLevel ?? _gameState?.tournamentBlindLevel ?? 0) + 1;
    // Prioritize tournament blinds from either source, fallback to table blinds only if neither available
    final smallBlind = _tournamentSmallBlind ?? 
                      _gameState?.tournamentSmallBlind ?? 
                      widget.table.smallBlind;
    final bigBlind = _tournamentBigBlind ?? 
                    _gameState?.tournamentBigBlind ?? 
                    widget.table.bigBlind;
    final nextSmall = _nextSmallBlind ?? _gameState?.tournamentNextSmallBlind;
    final nextBig = _nextBigBlind ?? _gameState?.tournamentNextBigBlind;

    // Use server-calculated remaining time (no client-side calculation)
    String countdown = '--:--';
    if (_levelTimeRemainingSecs != null) {
      final remaining = _levelTimeRemainingSecs!;
      if (remaining <= 0) {
        countdown = '00:00';
      } else {
        final mins = remaining ~/ 60;
        final secs = remaining % 60;
        countdown =
            '${mins.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
      }
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber.withOpacity(0.3)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Current level and countdown
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.timer, color: Colors.amber, size: 16),
              const SizedBox(width: 6),
              Text(
                'Level $level',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                countdown,
                style: TextStyle(
                  color:
                      countdown.startsWith('00:') ||
                          countdown.startsWith('0:') && countdown.length <= 4
                      ? Colors.red
                      : Colors.white70,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          // Current and next blinds
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Blinds: \$$smallBlind/\$$bigBlind',
                style: const TextStyle(color: Colors.white, fontSize: 13),
              ),
              if (nextSmall != null && nextBig != null) ...[
                const SizedBox(width: 8),
                const Icon(
                  Icons.arrow_forward,
                  color: Colors.white54,
                  size: 14,
                ),
                const SizedBox(width: 8),
                Text(
                  '\$$nextSmall/\$$nextBig',
                  style: const TextStyle(color: Colors.white54, fontSize: 13),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _wsService.disconnect();
    _raiseController.dispose();
    _buyinController.dispose();
    _topUpController.dispose();
    super.dispose();
  }
}
