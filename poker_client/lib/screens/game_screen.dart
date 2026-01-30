import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/club.dart';
import '../models/game_state.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
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
  GameState? _gameState;
  String _statusMessage = 'Connecting...';
  final _raiseController = TextEditingController(text: '200');
  final _buyinController = TextEditingController(text: '5000');
  final _topUpController = TextEditingController(text: '5000');
  bool _isSeated = false;

  @override
  void initState() {
    super.initState();
    _wsService = WebSocketService();
    _wsService.onGameStateUpdate = (gameState) {
      setState(() {
        _gameState = gameState;
        _updateStatusMessage();
        _checkIfSeated();
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
      setState(() => _statusMessage = 'Select a seat to join the game');
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
                  DropdownMenuItem(value: 'aggressive', child: Text('Aggressive')),
                  DropdownMenuItem(value: 'calling_station', child: Text('Calling Station')),
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

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.table.name),
        backgroundColor: Colors.green[800],
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
            colors: [Colors.green[900]!, Colors.green[700]!],
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
                    ? Stack(
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
                          ),

                          // Table center overlay (pot and community cards)
                          IgnorePointer(
                          child: Container(
                            constraints: const BoxConstraints(
                              maxWidth: 400,
                              maxHeight: 200,
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                // Pot
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 16,
                                    vertical: 8,
                                  ),
                                  decoration: BoxDecoration(
                                    color: Colors.black45,
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: Text(
                                    'POT: \$${_gameState!.potTotal}',
                                    style: const TextStyle(
                                      color: Colors.amber,
                                      fontSize: 24,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                                const SizedBox(height: 12),

                                // Community cards
                                if (_gameState!.communityCards.isNotEmpty)
                                  Container(
                                    padding: const EdgeInsets.all(8),
                                    decoration: BoxDecoration(
                                      color: Colors.black26,
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: _gameState!.communityCards
                                          .map(
                                            (card) => CardWidget(
                                              card: card,
                                              width: 50,
                                              height: 70,
                                              isShowdown: isShowdown,
                                            ),
                                          )
                                          .toList(),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                          ),
                        ],
                      )
                    : const Center(
                        child: CircularProgressIndicator(color: Colors.white),
                      ),
              ),
            ),

            // Status message
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

  @override
  void dispose() {
    _wsService.disconnect();
    _raiseController.dispose();
    _buyinController.dispose();
    _topUpController.dispose();
    super.dispose();
  }
}
