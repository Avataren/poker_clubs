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
  static const int _maxFeedEntries = 10;

  late WebSocketService _wsService;
  final _soundService = SoundService();
  GameState? _gameState;
  GameState? _previousGameState;
  final _raiseController = TextEditingController(text: '200');
  final _buyinController = TextEditingController(text: '5000');
  final _topUpController = TextEditingController(text: '5000');
  final List<_TableFeedEntry> _eventFeed = <_TableFeedEntry>[];
  int _unreadFeedCount = 0;
  bool _isSeated = false;
  bool _hasSeenPlayersAtTable = false;
  bool _isTableClosed = false;

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
      final feedEvents = _collectStateFeedEvents(gameState);
      print('DEBUG: GameState tournamentId = ${gameState.tournamentId}');
      setState(() {
        final wasSeated = _isSeated;
        final wasTableClosed = _isTableClosed;
        _previousGameState = _gameState;
        _gameState = gameState;
        _updateTableClosedState(gameState);
        _checkIfSeated();
        if (!wasSeated && _isSeated) {
          _appendFeedEvent('You are now seated.');
        } else if (wasSeated && !_isSeated) {
          _appendFeedEvent('You left your seat.');
        }
        if (!wasTableClosed && _isTableClosed) {
          _appendFeedEvent('Table closed. Waiting for table move.');
        }
        for (final event in feedEvents) {
          _appendFeedEvent(event);
        }
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
      final normalized = error.toLowerCase();
      final tableGone =
          normalized.contains('table not found') ||
          normalized.contains('table disappeared');

      final isTournamentContext =
          _gameState?.tournamentId != null || _tournamentId != null;
      if (tableGone && isTournamentContext) {
        setState(() {
          _isTableClosed = true;
          _appendFeedEvent('Table closed. Waiting for table move.');
        });
        return;
      }

      _recordFeedEvent('Error: $error');
    };
    _wsService.onConnected = () {
      _recordFeedEvent('Connected to server.');
      // Join table as observer (without seat)
      _wsService.joinTable(widget.table.id, 0);
      _recordFeedEvent('Joined table as observer.');
    };

    final token = context.read<ApiService>().token!;
    _wsService.connect(token);
  }

  void _checkIfSeated() {
    final myUserId = context.read<ApiService>().userId;
    _isSeated = _gameState?.players.any((p) => p.userId == myUserId) ?? false;
  }

  void _updateTableClosedState(GameState gameState) {
    if (gameState.tournamentId == null) return;

    final hasActivePlayers = gameState.players.any((p) => !p.isEliminated);
    if (hasActivePlayers) {
      _hasSeenPlayersAtTable = true;
      _isTableClosed = false;
      return;
    }

    if (_hasSeenPlayersAtTable) {
      _isTableClosed = true;
    }
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
      print('Action error: $e');
      _recordFeedEvent('Action failed: $e');
    }
  }

  ButtonStyle _actionButtonStyle(Color color) {
    return ElevatedButton.styleFrom(
      backgroundColor: color,
      foregroundColor: Colors.white,
      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
    );
  }

  List<String> _collectStateFeedEvents(GameState newState) {
    final events = <String>[];
    final previousState = _gameState;
    if (previousState == null) return events;

    if (previousState.phase != newState.phase) {
      events.add('Phase: ${newState.phase}');
    }

    final winnerMessage = newState.lastWinnerMessage?.trim();
    if (winnerMessage != null &&
        winnerMessage.isNotEmpty &&
        winnerMessage != previousState.lastWinnerMessage?.trim()) {
      events.add(winnerMessage);
    }

    final previousPlayersById = {
      for (final player in previousState.players) player.userId: player,
    };
    for (final player in newState.players) {
      final previousPlayer = previousPlayersById[player.userId];
      if (previousPlayer == null) continue;
      if (player.lastAction != null &&
          player.lastAction!.isNotEmpty &&
          player.lastAction != previousPlayer.lastAction) {
        events.add('${player.username}: ${player.lastAction}');
      }
    }

    final myUserId = context.read<ApiService>().userId;
    final wasMyTurn = previousState.currentPlayer?.userId == myUserId;
    final isMyTurn = newState.currentPlayer?.userId == myUserId;
    if (!wasMyTurn && isMyTurn) {
      events.add('Your turn.');
    }

    return events;
  }

  void _recordFeedEvent(String message) {
    if (!mounted) return;
    setState(() {
      _appendFeedEvent(message);
    });
  }

  void _appendFeedEvent(String message) {
    final text = message.trim();
    if (text.isEmpty) return;

    _eventFeed.insert(0, _TableFeedEntry(time: DateTime.now(), message: text));
    if (_eventFeed.length > _maxFeedEntries) {
      _eventFeed.removeRange(_maxFeedEntries, _eventFeed.length);
    }
    _unreadFeedCount = (_unreadFeedCount + 1).clamp(0, 99);
  }

  String _formatFeedTime(DateTime time) {
    final h = time.hour.toString().padLeft(2, '0');
    final m = time.minute.toString().padLeft(2, '0');
    final s = time.second.toString().padLeft(2, '0');
    return '$h:$m:$s';
  }

  Widget _buildFeedIcon() {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        const Icon(Icons.feed_outlined),
        if (_unreadFeedCount > 0)
          Positioned(
            right: -6,
            top: -6,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
              constraints: const BoxConstraints(minWidth: 16, minHeight: 16),
              decoration: BoxDecoration(
                color: Colors.redAccent,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                _unreadFeedCount > 9 ? '9+' : '$_unreadFeedCount',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
      ],
    );
  }

  void _openFeedSheet() {
    setState(() {
      _unreadFeedCount = 0;
    });

    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF121b2e),
      isScrollControlled: true,
      builder: (context) {
        return SafeArea(
          child: SizedBox(
            height: MediaQuery.of(context).size.height * 0.45,
            child: Column(
              children: [
                const SizedBox(height: 12),
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Table Feed',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 8),
                const Divider(color: Colors.white24, height: 1),
                Expanded(
                  child: _eventFeed.isEmpty
                      ? const Center(
                          child: Text(
                            'No messages yet',
                            style: TextStyle(color: Colors.white70),
                          ),
                        )
                      : ListView.separated(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 12,
                          ),
                          itemCount: _eventFeed.length,
                          separatorBuilder: (_, _) =>
                              const SizedBox(height: 10),
                          itemBuilder: (context, index) {
                            final event = _eventFeed[index];
                            return Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _formatFeedTime(event.time),
                                  style: const TextStyle(
                                    color: Colors.white54,
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: Text(
                                    event.message,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 14,
                                    ),
                                  ),
                                ),
                              ],
                            );
                          },
                        ),
                ),
              ],
            ),
          ),
        );
      },
    );
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
          onTakeSeat: (!_isSeated && !_isTableClosed) ? _takeSeat : null,
          onRemoveBot: _isTableClosed ? null : _removeBot,
          showingDown: isShowdown,
          gamePhase: _gameState!.phase,
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
    final isPortrait =
        MediaQuery.of(context).orientation == Orientation.portrait;

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.table.name),
        backgroundColor: Colors.grey[850],
        actions: [
          // Only show top-up button if format allows it
          if (_isSeated &&
              !_isTableClosed &&
              canTopUp &&
              myPlayer != null &&
              myPlayer.stack < 10000)
            IconButton(
              icon: const Icon(Icons.add_circle),
              tooltip: 'Top Up',
              onPressed: _topUp,
            ),
          if (_isSeated && !_isTableClosed)
            IconButton(
              icon: const Icon(Icons.event_seat),
              tooltip: 'Stand Up',
              onPressed: _standUp,
            ),
          if (!_isTableClosed)
            IconButton(
              icon: const Icon(Icons.smart_toy),
              tooltip: 'Add Bot',
              onPressed: _showAddBotDialog,
            ),
          IconButton(
            icon: _buildFeedIcon(),
            tooltip: 'Table Feed',
            onPressed: _openFeedSheet,
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

            // Show Cards buttons (fold-win showdown: winner can reveal cards)
            if (_isSeated &&
                !_isTableClosed &&
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
                          for (
                            int i = 0;
                            i < myPlayer.shownCards!.length;
                            i++
                          ) {
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
            if (_isSeated && !_isTableClosed && isMyTurn)
              Container(
                padding: const EdgeInsets.all(16),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  alignment: WrapAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: () => _playerAction('Fold'),
                      style: _actionButtonStyle(Colors.red),
                      child: const Text('Fold'),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction('Check'),
                      style: _actionButtonStyle(Colors.blue),
                      child: const Text('Check'),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction('Call'),
                      style: _actionButtonStyle(Colors.orange),
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
                      style: _actionButtonStyle(Colors.green),
                      child: const Text('Raise'),
                    ),
                    ElevatedButton(
                      onPressed: () => _playerAction('AllIn'),
                      style: _actionButtonStyle(Colors.purple),
                      child: const Text('All In'),
                    ),
                  ],
                ),
              )
            else
              const SizedBox.shrink(),
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
    final smallBlind =
        _tournamentSmallBlind ??
        _gameState?.tournamentSmallBlind ??
        widget.table.smallBlind;
    final bigBlind =
        _tournamentBigBlind ??
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

class _TableFeedEntry {
  final DateTime time;
  final String message;

  const _TableFeedEntry({required this.time, required this.message});
}
