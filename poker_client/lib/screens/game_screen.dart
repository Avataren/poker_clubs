import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/club.dart';
import '../models/game_state.dart';
import '../models/player.dart';
import '../models/tournament.dart';
import '../models/tournament_info_state.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import '../services/sound_service.dart';
import '../widgets/card_widget.dart';
import '../widgets/dialogs.dart';
import '../widgets/bet_sizing_panel.dart';
import '../widgets/table_seat_widget.dart';

class GameScreen extends StatefulWidget {
  final PokerTable table;

  const GameScreen({super.key, required this.table});

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  static const int _maxFeedEntries = 10;
  static const double _tableSceneWidth = 1400;
  static const double _tableSceneHeight = 900;
  static const double _tableSceneUpwardShiftFactor = 0.04;

  late WebSocketService _wsService;
  final _soundService = SoundService();
  GameState? _gameState;
  GameState? _previousGameState;
  final _raiseController = TextEditingController(text: '200');
  bool _showBetPanel = false;
  final _buyinController = TextEditingController(text: '5000');
  final _topUpController = TextEditingController(text: '5000');
  final List<_TableFeedEntry> _eventFeed = <_TableFeedEntry>[];
  final List<_TournamentPlacement> _tournamentPlacements =
      <_TournamentPlacement>[];
  int _unreadFeedCount = 0;
  bool _isSeated = false;
  bool _hasSeenPlayersAtTable = false;
  bool _isTableClosed = false;
  bool _showTournamentResultsOverlay = false;
  bool _loadingTournamentPlacements = false;
  bool _tournamentResultIsFinal = false;
  String? _tournamentResultsError;
  String? _finishedTournamentId;
  String? _finishedTournamentName;
  String? _waitingResultsProbeTournamentId;
  DateTime? _singlePlayerWaitingSince;

  // Tournament info from live broadcast
  final _tournamentInfo = TournamentInfoState();

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
        _showBetPanel = false;
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
      _maybeProbeTournamentCompletion(gameState);
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
            _tournamentInfo.update(
              tournamentId: tournamentId,
              serverTime: serverTime,
              level: level,
              smallBlind: smallBlind,
              bigBlind: bigBlind,
              ante: ante,
              levelStartTime: levelStartTime,
              levelDurationSecs: levelDurationSecs,
              levelTimeRemainingSecs: levelTimeRemainingSecs,
              nextSmallBlind: nextSmallBlind,
              nextBigBlind: nextBigBlind,
            );
          });
        };
    _wsService.onTournamentFinished = (tournamentId, tournamentName, winners) =>
        _handleTournamentFinished(tournamentId, tournamentName, winners);
    _wsService.onTournamentTableChanged = (tournamentId, tableId, userId) {
      final myUserId = context.read<ApiService>().userId;
      if (userId != myUserId) return;
      _wsService.joinTable(tableId, 0);
      _recordFeedEvent('Moved to new table.');
      setState(() {
        _isTableClosed = false;
        _hasSeenPlayersAtTable = false;
      });
    };
    _wsService.onError = (error) {
      final normalized = error.toLowerCase();
      final tableGone =
          normalized.contains('table not found') ||
          normalized.contains('table disappeared');

      final isTournamentContext =
          _gameState?.tournamentId != null ||
          _tournamentInfo.tournamentId != null;
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

  void _takeSeat(int seatNumber) async {
    final result = await InputDialog.show(
      context,
      title: 'Take Seat ${seatNumber + 1}',
      prompt: 'Enter buy-in amount:',
      controller: _buyinController,
      hintText: 'Buy-in amount',
      prefixText: '\$',
      confirmLabel: 'Take Seat',
      keyboardType: TextInputType.number,
    );
    if (result != null) {
      final amount = int.tryParse(result) ?? 5000;
      _wsService.takeSeat(widget.table.id, seatNumber, amount);
    }
  }

  void _standUp() async {
    final confirmed = await ConfirmationDialog.show(
      context,
      title: 'Stand Up',
      content:
          'Are you sure you want to leave your seat?\n\nYou will leave immediately if not in a hand, or after the current hand concludes.',
      confirmLabel: 'Stand Up',
      confirmColor: Colors.red,
    );
    if (confirmed) _wsService.standUp();
  }

  void _topUp() async {
    final result = await InputDialog.show(
      context,
      title: 'Top Up',
      prompt: 'Add chips to your stack:',
      controller: _topUpController,
      hintText: 'Top-up amount',
      prefixText: '\$',
      confirmLabel: 'Top Up',
      keyboardType: TextInputType.number,
    );
    if (result != null) {
      final amount = int.tryParse(result) ?? 5000;
      _wsService.topUp(amount);
    }
  }

  void _showAddBotDialog() {
    String botType = 'onnx'; // Default to ONNX models
    String scriptedStrategy = 'balanced';
    String onnxPersonality = 'onnx_gto'; // Default to GTO

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Add Bot'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Bot Type:', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              DropdownButton<String>(
                value: botType,
                isExpanded: true,
                items: const [
                  DropdownMenuItem(value: 'onnx', child: Text('ONNX AI Model')),
                  DropdownMenuItem(value: 'scripted', child: Text('Scripted Strategy')),
                ],
                onChanged: (value) {
                  setDialogState(() => botType = value!);
                },
              ),
              const SizedBox(height: 16),
              if (botType == 'onnx') ...[
                const Text('AI Personality:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                DropdownButton<String>(
                  value: onnxPersonality,
                  isExpanded: true,
                  items: const [
                    DropdownMenuItem(value: 'onnx_gto', child: Text('GTO')),
                    DropdownMenuItem(value: 'onnx_pro', child: Text('Pro')),
                    DropdownMenuItem(value: 'onnx_nit', child: Text('Nit')),
                    DropdownMenuItem(value: 'onnx_calling_station', child: Text('Calling Station')),
                    DropdownMenuItem(value: 'onnx_maniac', child: Text('Maniac')),
                  ],
                  onChanged: (value) {
                    setDialogState(() => onnxPersonality = value!);
                  },
                ),
              ] else ...[
                const Text('Strategy:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                DropdownButton<String>(
                  value: scriptedStrategy,
                  isExpanded: true,
                  items: const [
                    DropdownMenuItem(value: 'balanced', child: Text('Balanced')),
                    DropdownMenuItem(value: 'tight', child: Text('Tight')),
                    DropdownMenuItem(value: 'aggressive', child: Text('Aggressive')),
                    DropdownMenuItem(value: 'calling_station', child: Text('Calling Station')),
                  ],
                  onChanged: (value) {
                    setDialogState(() => scriptedStrategy = value!);
                  },
                ),
              ],
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                final strategy = botType == 'onnx' ? onnxPersonality : scriptedStrategy;
                _wsService.addBot(widget.table.id, strategy: strategy);
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
      foregroundColor: Colors.black,
      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      elevation: 4,
      shadowColor: Colors.black87,
      // Add border for depth
      side: BorderSide(color: Colors.black.withOpacity(0.4), width: 1.5),
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

  Future<void> _handleTournamentFinished(
    String tournamentId,
    String tournamentName,
    List<TournamentWinner> winners,
  ) async {
    var resolvedTournamentName = tournamentName;
    var isFinalTournamentResult = winners.isNotEmpty;

    // Probe path (winners empty): still show the dialog so table closures/
    // rebalances are visible, but hide payouts until the tournament is truly
    // finished.
    if (winners.isEmpty) {
      try {
        final detail = await context.read<ApiService>().getTournamentDetail(
          tournamentId,
        );
        if (!mounted) return;
        resolvedTournamentName = detail.tournament.name;
        isFinalTournamentResult = detail.tournament.status == 'finished';
      } catch (_) {
        // If lookup fails, keep showing the dialog without payouts.
        isFinalTournamentResult = false;
      }
    }

    final activeTournamentId =
        _gameState?.tournamentId ?? _tournamentInfo.tournamentId;
    if (activeTournamentId != null && activeTournamentId != tournamentId) {
      return;
    }

    if (_finishedTournamentId == tournamentId &&
        _showTournamentResultsOverlay) {
      // The probe path may have fetched results before prizes were distributed,
      // showing $0 for all entries. If the WS TournamentFinished message arrives
      // with actual winner data, allow re-processing to get correct prizes.
      final hasPrizes = _tournamentPlacements.any((p) => p.prize > 0);
      if (hasPrizes || winners.isEmpty) {
        return;
      }
    }

    _recordFeedEvent('Tournament finished: $resolvedTournamentName');
    setState(() {
      _isTableClosed = true;
      _waitingResultsProbeTournamentId = tournamentId;
      _finishedTournamentId = tournamentId;
      _finishedTournamentName = resolvedTournamentName;
      _tournamentResultIsFinal = isFinalTournamentResult;
      _showTournamentResultsOverlay = true;
      _loadingTournamentPlacements = true;
      _tournamentResultsError = null;
      _tournamentPlacements.clear();
    });

    List<_TournamentPlacement> placements = const [];
    String? fetchError;

    try {
      final results = await context.read<ApiService>().getTournamentResults(
        tournamentId,
      );
      placements = _placementsFromResults(results);
    } catch (e) {
      fetchError = e.toString();
    }

    // Fall back to WS winners if the API returned no results, or if the API
    // returned results but none have prizes yet (race: prizes not distributed).
    final apiHasPrizes = placements.any((p) => p.prize > 0);
    if (placements.isEmpty || (!apiHasPrizes && winners.isNotEmpty)) {
      final wsPlacements = _placementsFromWinners(winners);
      if (wsPlacements.isNotEmpty) {
        placements = wsPlacements;
      }
    }

    if (!mounted || _finishedTournamentId != tournamentId) {
      return;
    }

    setState(() {
      _loadingTournamentPlacements = false;
      _tournamentPlacements
        ..clear()
        ..addAll(placements);
      _tournamentResultsError = placements.isEmpty
          ? (isFinalTournamentResult
                ? 'Could not load tournament payouts.'
                : 'No current standings available.')
          : null;
    });

    if (fetchError != null) {
      _recordFeedEvent('Results lookup failed: $fetchError');
    }
  }

  void _maybeProbeTournamentCompletion(GameState gameState) {
    final tournamentId = gameState.tournamentId;
    if (tournamentId == null) {
      _singlePlayerWaitingSince = null;
      return;
    }

    final isWaiting = gameState.phase.toLowerCase() == 'waiting';
    final activePlayers = gameState.players
        .where((p) => !p.isEliminated)
        .length;
    final singlePlayerRemaining = activePlayers <= 1;

    if (!isWaiting || !singlePlayerRemaining || _showTournamentResultsOverlay) {
      _singlePlayerWaitingSince = null;
      return;
    }

    final now = DateTime.now();
    _singlePlayerWaitingSince ??= now;
    if (now.difference(_singlePlayerWaitingSince!).inSeconds < 2) {
      return;
    }

    if (_loadingTournamentPlacements ||
        _finishedTournamentId == tournamentId ||
        _waitingResultsProbeTournamentId == tournamentId) {
      return;
    }

    _waitingResultsProbeTournamentId = tournamentId;
    _handleTournamentFinished(tournamentId, widget.table.name, const []);
  }

  List<_TournamentPlacement> _placementsFromResults(
    List<TournamentResult> results,
  ) {
    final placements =
        results
            .map(
              (result) => _TournamentPlacement(
                username: result.username,
                position: result.finishPosition,
                prize: result.prizeAmount,
              ),
            )
            .where((result) => result.position > 0)
            .toList()
          ..sort((a, b) => a.position.compareTo(b.position));

    return placements;
  }

  List<_TournamentPlacement> _placementsFromWinners(
    List<TournamentWinner> winners,
  ) {
    final placements =
        winners
            .map(
              (winner) => _TournamentPlacement(
                username: winner.username,
                position: winner.position.toInt(),
                prize: winner.prize.toInt(),
              ),
            )
            .where((winner) => winner.position > 0)
            .toList()
          ..sort((a, b) => a.position.compareTo(b.position));

    return placements;
  }

  List<_TournamentPlacement> _displayedPlacements() {
    if (_tournamentPlacements.isEmpty) {
      return const <_TournamentPlacement>[];
    }

    final paidPlaces = _tournamentPlacements.where((p) => p.prize > 0).length;
    final targetCount = paidPlaces > 3 ? paidPlaces : 3;
    final count = targetCount > _tournamentPlacements.length
        ? _tournamentPlacements.length
        : targetCount;

    return _tournamentPlacements.take(count).toList();
  }

  String _placementBadge(int position) {
    switch (position) {
      case 1:
        return '1st';
      case 2:
        return '2nd';
      case 3:
        return '3rd';
      default:
        return '#$position';
    }
  }

  String _formatPayout(int value) {
    final sign = value < 0 ? '-' : '';
    final digits = value.abs().toString();
    final buffer = StringBuffer();
    for (int i = 0; i < digits.length; i++) {
      if (i > 0 && (digits.length - i) % 3 == 0) {
        buffer.write(',');
      }
      buffer.write(digits[i]);
    }
    return '$sign\$${buffer.toString()}';
  }

  Widget _buildTournamentResultsOverlay() {
    final placements = _displayedPlacements();
    final showPayouts = _tournamentResultIsFinal;
    final title = showPayouts ? 'Tournament Finished' : 'Table Closed';

    return Positioned.fill(
      child: Container(
        color: Colors.black.withValues(alpha: 0.82),
        child: SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: BoxConstraints(
                maxWidth: 560,
                maxHeight: MediaQuery.of(context).size.height * 0.84,
              ),
              child: Container(
                margin: const EdgeInsets.all(16),
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFF16213e),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.amber.shade300, width: 1.6),
                ),
                child: Column(
                  children: [
                    Text(
                      title,
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    if (_finishedTournamentName != null) ...[
                      const SizedBox(height: 6),
                      Text(
                        _finishedTournamentName!,
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.white70),
                      ),
                    ],
                    if (!showPayouts) ...[
                      const SizedBox(height: 6),
                      const Text(
                        'Tournament still running. Payouts show after the final table.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ],
                    const SizedBox(height: 16),
                    Expanded(
                      child: _loadingTournamentPlacements
                          ? const Center(
                              child: CircularProgressIndicator(
                                color: Colors.amber,
                              ),
                            )
                          : placements.isEmpty
                          ? Center(
                              child: Text(
                                _tournamentResultsError ??
                                    'No tournament results available.',
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: Colors.white70),
                              ),
                            )
                          : ListView.separated(
                              itemCount: placements.length,
                              separatorBuilder: (_, _) =>
                                  const SizedBox(height: 10),
                              itemBuilder: (context, index) {
                                final placement = placements[index];
                                return Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 14,
                                    vertical: 12,
                                  ),
                                  decoration: BoxDecoration(
                                    color: Colors.black26,
                                    borderRadius: BorderRadius.circular(10),
                                    border: Border.all(color: Colors.white24),
                                  ),
                                  child: Row(
                                    children: [
                                      SizedBox(
                                        width: 48,
                                        child: Text(
                                          _placementBadge(placement.position),
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 18,
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ),
                                      Expanded(
                                        child: Text(
                                          placement.username,
                                          overflow: TextOverflow.ellipsis,
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 16,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ),
                                      if (showPayouts)
                                        Text(
                                          _formatPayout(placement.prize),
                                          style: const TextStyle(
                                            color: Colors.lightGreenAccent,
                                            fontSize: 16,
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                    ],
                                  ),
                                );
                              },
                            ),
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 10,
                      runSpacing: 10,
                      alignment: WrapAlignment.center,
                      children: [
                        if (!_loadingTournamentPlacements &&
                            placements.isEmpty &&
                            _finishedTournamentId != null)
                          OutlinedButton(
                            onPressed: () {
                              final tid = _finishedTournamentId;
                              final tname =
                                  _finishedTournamentName ?? 'Tournament';
                              if (tid == null) return;
                              _handleTournamentFinished(tid, tname, const []);
                            },
                            child: const Text('Retry'),
                          ),
                        ElevatedButton(
                          onPressed: () {
                            setState(() {
                              _showTournamentResultsOverlay = false;
                            });
                          },
                          child: const Text('Close'),
                        ),
                        ElevatedButton(
                          onPressed: () {
                            _wsService.leaveTable();
                            _wsService.disconnect();
                            Navigator.pop(context);
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.red,
                            foregroundColor: Colors.black,
                          ),
                          child: const Text('Leave Table'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTableStack({
    required double sceneWidth,
    required double sceneHeight,
    required String? myUserId,
    required bool isVisualShowdown,
  }) {
    final tableVerticalShift = sceneHeight * _tableSceneUpwardShiftFactor;

    // Calculate responsive table dimensions - match table widget sizing
    final maxWidth = sceneWidth * 0.75;
    final maxHeight = sceneHeight * 0.75;

    double tableWidth, tableHeight;
    if (maxWidth / maxHeight > 1.6) {
      tableHeight = maxHeight;
      tableWidth = tableHeight * 1.6;
    } else {
      tableWidth = maxWidth;
      tableHeight = tableWidth / 1.6;
    }

    // Calculate responsive positioning for overlay elements.
    // Keep community cards closer to table center for better balance.
    final cardTop = (sceneHeight - tableHeight) / 2 + tableHeight * 0.36;

    // Calculate card dimensions with proper aspect ratio (1.4:1)
    final cardWidth = (tableWidth * 0.105).clamp(52.0, 80.0);
    final cardHeight = cardWidth * 1.4;

    return Transform.translate(
      offset: Offset(0, -tableVerticalShift),
      child: Stack(
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
            showingDown: isVisualShowdown,
            gamePhase: _gameState!.phase,
            dealerSeat: _gameState!.dealerSeat,
            smallBlindSeat: _gameState!.smallBlindSeat,
            bigBlindSeat: _gameState!.bigBlindSeat,
            smallBlind: widget.table.smallBlind,
            potTotal: _gameState!.potTotal,
            pots: _gameState!.pots,
            tournamentId: _gameState!.tournamentId,
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
                                        isShowdown: isVisualShowdown,
                                      ),
                                    )
                                    .toList(),
                              ),
                            )
                          : const SizedBox.shrink(),
                    ),
                    // Tournament info (if tournament table)
                    if (_tournamentInfo.tournamentId != null ||
                        _gameState!.tournamentId != null)
                      SizedBox(height: tableHeight * 0.02),
                    if (_tournamentInfo.tournamentId != null ||
                        _gameState!.tournamentId != null)
                      _buildTournamentInfo(),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
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
    final isShowdownPhase = _gameState?.phase.toLowerCase() == 'showdown';
    final activeInHandCount =
        _gameState?.players.where((p) => p.isActive || p.isAllIn).length ?? 0;
    final inferredUncontestedWin = isShowdownPhase && activeInHandCount == 1;
    final isUncontestedWin =
        (_gameState?.wonWithoutShowdown ?? false) || inferredUncontestedWin;
    final isVisualShowdown = isShowdownPhase && !isUncontestedWin;
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
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [const Color(0xFF1a1a2e), const Color(0xFF16213e)],
              ),
            ),
            child: Column(
              children: [
                _buildGameHeader(),

                // Poker table with seats — fills all remaining space
                Expanded(
                  child: Center(
                    child: _gameState != null
                        ? (() {
                            final tableScene = SizedBox(
                              width: _tableSceneWidth,
                              height: _tableSceneHeight,
                              child: _buildTableStack(
                                sceneWidth: _tableSceneWidth,
                                sceneHeight: _tableSceneHeight,
                                myUserId: myUserId,
                                isVisualShowdown: isVisualShowdown,
                              ),
                            );

                            if (!isPortrait) {
                              return ClipRect(
                                child: FittedBox(
                                  fit: BoxFit.cover,
                                  child: tableScene,
                                ),
                              );
                            }

                            return ClipRect(
                              child: FittedBox(
                                fit: BoxFit.cover,
                                child: SizedBox(
                                  width: _tableSceneHeight,
                                  height: _tableSceneWidth,
                                  child: RotatedBox(
                                    quarterTurns: 1,
                                    child: tableScene,
                                  ),
                                ),
                              ),
                            );
                          })()
                        : const Center(
                            child: CircularProgressIndicator(
                              color: Colors.white,
                            ),
                          ),
                  ),
                ),
              ],
            ),
          ),
          // Action buttons + show cards overlaid at the bottom
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: SafeArea(
              top: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildShowCardsSection(
                    myPlayer,
                    isShowdownPhase,
                    isUncontestedWin,
                  ),
                  _buildActionButtons(isMyTurn, myPlayer),
                ],
              ),
            ),
          ),
          if (_showTournamentResultsOverlay) _buildTournamentResultsOverlay(),
        ],
      ),
    );
  }

  Widget _buildGameHeader() {
    final phaseLabel = _gameState == null
        ? 'WAITING'
        : (_gameState!.wonWithoutShowdown
              ? 'HAND OVER'
              : _gameState!.phase.toUpperCase());

    return Container(
      padding: const EdgeInsets.all(8),
      child: Column(
        children: [
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
            phaseLabel,
            style: const TextStyle(
              color: Colors.amber,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildShowCardsSection(
    Player? myPlayer,
    bool isShowdownPhase,
    bool isUncontestedWin,
  ) {
    if (!_isSeated ||
        _isTableClosed ||
        !isShowdownPhase ||
        !isUncontestedWin ||
        myPlayer == null ||
        !myPlayer.isWinner ||
        myPlayer.isBot ||
        myPlayer.shownCards == null ||
        _gameState?.winningHand != null) {
      return const SizedBox.shrink();
    }

    return Container(
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
                  foregroundColor: Colors.black,
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
                foregroundColor: Colors.black,
              ),
              child: const Text('Show All'),
            ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(bool isMyTurn, Player? myPlayer) {
    if (!_isSeated || _isTableClosed || !isMyTurn || myPlayer == null) {
      if (_showBetPanel) {
        setState(() => _showBetPanel = false);
      }
      return const SizedBox.shrink();
    }

    final gs = _gameState!;
    final toCall = gs.currentBet - myPlayer.currentBet;
    final canCheck = toCall <= 0;
    final canRaise = myPlayer.stack > toCall;

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          // Fold
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: ElevatedButton(
              onPressed: () => _playerAction('Fold'),
              style: _actionButtonStyle(Colors.red),
              child: const Text('Fold'),
            ),
          ),
          // Check or Call
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: canCheck
                ? ElevatedButton(
                    onPressed: () => _playerAction('Check'),
                    style: _actionButtonStyle(Colors.blue),
                    child: const Text('Check'),
                  )
                : ElevatedButton(
                    onPressed: () => _playerAction('Call'),
                    style: _actionButtonStyle(Colors.orange),
                    child: Text('Call \$$toCall'),
                  ),
          ),
          // Bet/Raise with popup stacked above, or All-In
          if (canRaise)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: IntrinsicWidth(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (_showBetPanel)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: BetSizingPopup(
                          isPreflop: gs.phase.toLowerCase() == 'preflop',
                          bigBlind: gs.bigBlind > 0
                              ? gs.bigBlind
                              : widget.table.bigBlind,
                          potTotal: gs.potTotal,
                          currentBet: gs.currentBet,
                          playerCurrentBet: myPlayer.currentBet,
                          playerStack: myPlayer.stack,
                          minRaise: gs.minRaise > 0 ? gs.minRaise : gs.bigBlind,
                          onSelect: (amount) {
                            setState(() => _showBetPanel = false);
                            _playerAction('Raise', amount: amount);
                          },
                        ),
                      ),
                    ElevatedButton(
                      onPressed: () =>
                          setState(() => _showBetPanel = !_showBetPanel),
                      style: _actionButtonStyle(
                        _showBetPanel ? Colors.grey : Colors.green,
                      ),
                      child: Text(gs.currentBet > 0 ? 'Raise' : 'Bet'),
                    ),
                  ],
                ),
              ),
            )
          else
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: ElevatedButton(
                onPressed: () => _playerAction('AllIn'),
                style: _actionButtonStyle(Colors.purple),
                child: const Text('All In'),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildTournamentInfo() {
    // Use live tournament info from broadcast if available, otherwise use gameState data
    final tournamentId =
        _tournamentInfo.tournamentId ?? _gameState?.tournamentId;
    if (tournamentId == null) {
      return const SizedBox.shrink();
    }

    final level =
        (_tournamentInfo.level ?? _gameState?.tournamentBlindLevel ?? 0) + 1;
    // Prioritize tournament blinds from either source, fallback to table blinds only if neither available
    final smallBlind =
        _tournamentInfo.smallBlind ??
        _gameState?.tournamentSmallBlind ??
        widget.table.smallBlind;
    final bigBlind =
        _tournamentInfo.bigBlind ??
        _gameState?.tournamentBigBlind ??
        widget.table.bigBlind;
    final nextSmall =
        _tournamentInfo.nextSmallBlind ?? _gameState?.tournamentNextSmallBlind;
    final nextBig =
        _tournamentInfo.nextBigBlind ?? _gameState?.tournamentNextBigBlind;

    // Use server-calculated remaining time (no client-side calculation)
    String countdown = '--:--';
    if (_tournamentInfo.levelTimeRemainingSecs != null) {
      final remaining = _tournamentInfo.levelTimeRemainingSecs!;
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
    _wsService.onTournamentFinished = null;
    _wsService.onTournamentTableChanged = null;
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

class _TournamentPlacement {
  final String username;
  final int position;
  final int prize;

  const _TournamentPlacement({
    required this.username,
    required this.position,
    required this.prize,
  });
}
