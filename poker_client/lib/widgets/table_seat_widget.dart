import 'dart:math';
import 'package:flutter/material.dart';
import '../models/player.dart';
import 'card_widget.dart';
import 'chip_stack_widget.dart';

class TableSeatWidget extends StatelessWidget {
  final int seatNumber;
  final Player? player;
  final bool isCurrentTurn;
  final bool isMe;
  final VoidCallback? onTakeSeat;
  final VoidCallback? onRemoveBot;
  final bool showingDown;
  final String? winningHand;
  final bool isDealer;
  final bool isSmallBlind;
  final bool isBigBlind;
  final int smallBlind;

  const TableSeatWidget({
    super.key,
    required this.seatNumber,
    this.player,
    this.isCurrentTurn = false,
    this.isMe = false,
    this.onTakeSeat,
    this.onRemoveBot,
    this.showingDown = false,
    this.winningHand,
    this.isDealer = false,
    this.isSmallBlind = false,
    this.isBigBlind = false,
    this.smallBlind = 10,
  });

  bool get _hasCards {
    return player != null &&
        player!.holeCards != null &&
        player!.holeCards!.isNotEmpty;
  }

  @override
  Widget build(BuildContext context) {
    final isEmpty = player == null;

    return Stack(
      clipBehavior: Clip.none,
      alignment: Alignment.center,
      children: [
        Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Cards (above the seat circle)
            if (_hasCards)
              Row(
                mainAxisSize: MainAxisSize.min,
                children: player!.holeCards!
                    .map(
                      (card) => CardWidget(
                        card: card,
                        width: 35,
                        height: 50,
                        isShowdown: showingDown,
                      ),
                    )
                    .toList(),
              )
            else
              const SizedBox(height: 50),

            const SizedBox(height: 4),

            // Seat circle with winner badge overlay
            Stack(
              alignment: Alignment.center,
              clipBehavior: Clip.none,
              children: [
                // Seat circle
                GestureDetector(
                  onTap: isEmpty ? onTakeSeat : null,
                  onLongPress:
                      (!isEmpty && player!.isBot && onRemoveBot != null)
                      ? onRemoveBot
                      : null,
                  child: Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isEmpty
                          ? Colors.black26
                          : (isCurrentTurn
                                ? Colors.amber[700]
                                : Colors.blue[900]),
                      border: Border.all(
                        color: isMe
                            ? Colors.green
                            : (isEmpty ? Colors.white30 : Colors.white70),
                        width: isMe ? 3 : 2,
                      ),
                      boxShadow: isCurrentTurn
                          ? [
                              BoxShadow(
                                color: Colors.amber.withValues(alpha: 0.5),
                                blurRadius: 10,
                                spreadRadius: 2,
                              ),
                            ]
                          : null,
                    ),
                    child: Center(
                      child: isEmpty
                          ? Text(
                              'Seat\n${seatNumber + 1}',
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                color: Colors.white54,
                                fontSize: 12,
                              ),
                            )
                          : Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  player!.username,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    color: isCurrentTurn
                                        ? Colors.black
                                        : Colors.white,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 11,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  '\$${player!.stack}',
                                  style: TextStyle(
                                    color: isCurrentTurn
                                        ? Colors.black87
                                        : Colors.green[300],
                                    fontWeight: FontWeight.bold,
                                    fontSize: 12,
                                  ),
                                ),
                                if (player!.currentBet > 0)
                                  Text(
                                    'Bet: \$${player!.currentBet}',
                                    style: TextStyle(
                                      color: isCurrentTurn
                                          ? Colors.black54
                                          : Colors.white70,
                                      fontSize: 9,
                                    ),
                                  ),
                              ],
                            ),
                    ),
                  ),
                ),

                // Dealer / Blind badge
                if (!isEmpty && (isDealer || isSmallBlind || isBigBlind))
                  Positioned(
                    bottom: -4,
                    left: -4,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 5,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: isDealer
                            ? Colors.white
                            : isSmallBlind
                            ? Colors.blue[700]
                            : Colors.orange[700],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: isDealer
                              ? Colors.black54
                              : isSmallBlind
                              ? Colors.blue[900]!
                              : Colors.orange[900]!,
                          width: 1,
                        ),
                      ),
                      child: Text(
                        isDealer
                            ? 'D'
                            : isSmallBlind
                            ? 'SB'
                            : 'BB',
                        style: TextStyle(
                          color: isDealer ? Colors.black : Colors.white,
                          fontSize: 9,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),

                // Bot indicator
                if (player != null && player!.isBot)
                  Positioned(
                    top: -4,
                    right: -4,
                    child: Container(
                      padding: const EdgeInsets.all(2),
                      decoration: BoxDecoration(
                        color: Colors.grey[800],
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white54, width: 1),
                      ),
                      child: const Icon(
                        Icons.smart_toy,
                        size: 12,
                        color: Colors.white70,
                      ),
                    ),
                  ),

                // Winner badge overlay (on top of circle)
                if (player != null && player!.isWinner && showingDown)
                  Positioned(
                    child: Container(
                      constraints: const BoxConstraints(maxWidth: 70),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.amber[600],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: Colors.amber[900]!,
                          width: 1.5,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.4),
                            blurRadius: 6,
                            spreadRadius: 1,
                          ),
                        ],
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Text('🏆', style: TextStyle(fontSize: 16)),
                          if (winningHand != null)
                            Text(
                              winningHand!,
                              style: const TextStyle(
                                color: Colors.black,
                                fontSize: 9,
                                fontWeight: FontWeight.bold,
                              ),
                              textAlign: TextAlign.center,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),

            // Player state indicator
            if (!isEmpty && (player!.isFolded || !player!.isActive))
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  player!.state,
                  style: const TextStyle(
                    color: Colors.white54,
                    fontSize: 10,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),

            // Last action indicator
            if (!isEmpty && player!.lastAction != null)
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    player!.lastAction!,
                    style: const TextStyle(
                      color: Colors.amber,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ],
    );
  }
}

class PokerTableWidget extends StatefulWidget {
  final int maxSeats;
  final List<Player> players;
  final int currentPlayerSeat;
  final String? myUserId;
  final Function(int)? onTakeSeat;
  final Function(String)? onRemoveBot;
  final bool showingDown;
  final String gamePhase;
  final String? winningHand;
  final int? dealerSeat;
  final int? smallBlindSeat;
  final int? bigBlindSeat;
  final int smallBlind;
  final int potTotal;

  const PokerTableWidget({
    super.key,
    required this.maxSeats,
    required this.players,
    required this.currentPlayerSeat,
    this.myUserId,
    this.onTakeSeat,
    this.onRemoveBot,
    this.showingDown = false,
    this.gamePhase = 'waiting',
    this.winningHand,
    this.dealerSeat,
    this.smallBlindSeat,
    this.bigBlindSeat,
    this.smallBlind = 10,
    this.potTotal = 0,
  });

  @override
  State<PokerTableWidget> createState() => _PokerTableWidgetState();
}

class _PokerTableWidgetState extends State<PokerTableWidget> {
  bool _animatingPot = false;
  bool _lastShowingDown = false;
  Offset? _potTargetOffset; // Where the pot should move to
  Map<String, int> _winnerPots = {}; // Map of userId to pot amount won
  String? _lastPhase;

  // Card dealing animation
  final Set<String> _dealingCardsTo =
      {}; // Set of player IDs currently being dealt cards
  String? _lastPhaseForCards;

  @override
  void initState() {
    super.initState();
    _lastPhaseForCards = widget.gamePhase;
  }

  @override
  void dispose() {
    super.dispose();
  }

  @override
  void didUpdateWidget(PokerTableWidget oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Detect new hand starting - reset pot animation and trigger card dealing animation
    if (_lastPhaseForCards != 'PreFlop' && widget.gamePhase == 'PreFlop') {
      // Reset pot animation state for new hand
      setState(() {
        _animatingPot = false;
        _potTargetOffset = null;
        _winnerPots.clear();
      });

      // Clear any existing animation state
      _dealingCardsTo.clear();

      // Schedule card dealing animation with stagger
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;

        // Add a delay for shuffle sound to play (250ms)
        Future.delayed(const Duration(milliseconds: 250), () {
          if (!mounted) return;

          // Deal cards to each player with staggered timing
          for (int i = 0; i < widget.players.length; i++) {
            final player = widget.players[i];
            if (player.holeCards != null && player.holeCards!.isNotEmpty) {
              Future.delayed(Duration(milliseconds: i * 30), () {
                if (mounted) {
                  setState(() {
                    _dealingCardsTo.add(player.userId);
                  });

                  // Remove from dealing set after animation completes
                  Future.delayed(const Duration(milliseconds: 150), () {
                    if (mounted) {
                      setState(() {
                        _dealingCardsTo.remove(player.userId);
                      });
                    }
                  });
                }
              });
            }
          }
        });
      });
    }
    _lastPhaseForCards = widget.gamePhase;

    // Don't trigger pot animation if one is already running
    if (_animatingPot) return;

    // Detect all winners and their pot amounts
    final winners = widget.players.where((p) => p.isWinner && p.potWon > 0).toList();
    
    // Only trigger if we have winners and haven't animated these specific winnings yet
    if (winners.isNotEmpty) {
      final newWinners = <String, int>{};
      for (final winner in winners) {
        // Check if this is a new win (not already in our tracked winners)
        if (!_winnerPots.containsKey(winner.userId)) {
          newWinners[winner.userId] = winner.potWon;
        }
      }
      
      if (newWinners.isNotEmpty) {
        print('Starting pot animation for ${newWinners.length} winner(s):');
        for (final entry in newWinners.entries) {
          final winner = widget.players.firstWhere((p) => p.userId == entry.key);
          print('  - ${winner.username}: \$${entry.value}');
        }

        // Schedule animation to start in next frame
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) {
            setState(() {
              _animatingPot = true;
              _winnerPots.addAll(newWinners);
            });
            print('Pot animation started for ${newWinners.length} winner(s)');
          }
        });
      }
    }

    _lastShowingDown = widget.showingDown;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate table size with fixed aspect ratio - use 85% of available space
        final maxWidth = constraints.maxWidth * 0.85;
        final maxHeight = constraints.maxHeight * 0.85;

        // Oval table aspect ratio (width:height = 1.6:1)
        double tableWidth, tableHeight;
        if (maxWidth / maxHeight > 1.6) {
          tableHeight = maxHeight;
          tableWidth = tableHeight * 1.6;
        } else {
          tableWidth = maxWidth;
          tableHeight = tableWidth / 1.6;
        }

        // Calculate pot center position
        final centerOffset = Offset(tableWidth / 2, tableHeight / 2 - 150);

        return SizedBox(
          width: tableWidth,
          height: tableHeight,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              // Table surface
              Center(
                child: Container(
                  width: tableWidth * 0.7,
                  height: tableHeight * 0.7,
                  decoration: BoxDecoration(
                    color: Colors.green[800],
                    borderRadius: BorderRadius.circular(
                      (tableHeight * 0.7) / 2,
                    ),
                    border: Border.all(color: Colors.brown, width: 8),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 15,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                ),
              ),
              // Position seats around the table
              for (int i = 0; i < widget.maxSeats; i++)
                _buildSeatAtPosition(i, tableWidth, tableHeight),
              // Animated dealing cards (fly from center to players)
              for (int i = 0; i < widget.maxSeats; i++)
                _buildDealingCardsAnimation(i, tableWidth, tableHeight),
              // Player bet chips on table (between player and center)
              for (int i = 0; i < widget.maxSeats; i++)
                _buildChipsAtPosition(i, tableWidth, tableHeight),
              
              // Pot chips - if animating, show separate chips for each winner
              if (_animatingPot && _winnerPots.isNotEmpty)
                // Animate each winner's pot portion separately
                ..._winnerPots.entries.map((entry) {
                  final winnerId = entry.key;
                  final potAmount = entry.value;
                  final winner = widget.players.firstWhere((p) => p.userId == winnerId);
                  
                  // Calculate winner seat position
                  final angle = (2 * pi * winner.seat / widget.maxSeats) - pi / 2;
                  final radiusX = tableWidth * 0.35;
                  final radiusY = tableHeight * 0.35;
                  final targetX = radiusX * cos(angle);
                  final targetY = radiusY * sin(angle);
                  final targetOffset = Offset(
                    (tableWidth / 2) + targetX,
                    (tableHeight / 2) + targetY - 40,
                  );
                  
                  return AnimatedPositioned(
                    duration: const Duration(milliseconds: 1000),
                    curve: Curves.easeInOut,
                    left: targetOffset.dx,
                    top: targetOffset.dy,
                    child: FractionalTranslation(
                      translation: const Offset(-0.5, 0),
                      child: AnimatedScale(
                        duration: const Duration(milliseconds: 1000),
                        scale: 0.8,
                        child: ChipStackWidget(
                          amount: potAmount,
                          smallBlind: widget.smallBlind,
                          scale: 1.0,
                          showAmount: true,
                        ),
                      ),
                    ),
                  );
                }).toList()
              // Show centered pot when not animating
              else if (widget.potTotal > 0)
                AnimatedPositioned(
                  duration: const Duration(milliseconds: 0),
                  left: centerOffset.dx,
                  top: centerOffset.dy,
                  child: FractionalTranslation(
                    translation: const Offset(-0.5, 0),
                    child: ChipStackWidget(
                      amount: widget.potTotal,
                      smallBlind: widget.smallBlind,
                      scale: 1.1,
                      showAmount: true,
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildSeatAtPosition(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    // Calculate position around oval
    final angle =
        (2 * pi * seatIndex / widget.maxSeats) - pi / 2; // Start from top

    // Oval dimensions - seats positioned outside the table surface
    // Table surface is 70% of container, so we position seats around the full container
    final radiusX = (tableWidth / 2) - 30;
    final radiusY =
        (tableHeight / 2) +
        10; // Even larger Y radius to move seats away from center

    final x = radiusX * cos(angle);
    final y = radiusY * sin(angle);

    // Find player at this seat
    final player = widget.players.where((p) => p.seat == seatIndex).firstOrNull;
    final isMe = player?.userId == widget.myUserId;
    final isCurrentTurn =
        widget.gamePhase.toLowerCase() != 'waiting' &&
        widget.currentPlayerSeat == seatIndex;

    return Positioned(
      left:
          (tableWidth / 2) + x - 40, // Center - offset for widget width (80/2)
      top:
          (tableHeight / 2) +
          y -
          95, // Center - offset for widget height, adjusted downward
      child: TableSeatWidget(
        seatNumber: seatIndex,
        player: player,
        isCurrentTurn: isCurrentTurn,
        isMe: isMe,
        onTakeSeat: player == null && widget.onTakeSeat != null
            ? () => widget.onTakeSeat!(seatIndex)
            : null,
        onRemoveBot:
            player != null && player.isBot && widget.onRemoveBot != null
            ? () => widget.onRemoveBot!(player.userId)
            : null,
        showingDown: widget.showingDown,
        winningHand: widget.winningHand,
        isDealer: widget.dealerSeat == seatIndex,
        isSmallBlind: widget.smallBlindSeat == seatIndex,
        isBigBlind: widget.bigBlindSeat == seatIndex,
        smallBlind: widget.smallBlind,
      ),
    );
  }

  Widget _buildChipsAtPosition(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    // Find player at this seat
    final player = widget.players.where((p) => p.seat == seatIndex).firstOrNull;

    // No chips if no player or no bet
    if (player == null || player.currentBet <= 0) {
      return const SizedBox.shrink();
    }

    // Calculate position around oval - same angle as seat
    final angle = (2 * pi * seatIndex / widget.maxSeats) - pi / 2;

    // Position chips at 60% of the seat radius (on the table, between player and center)
    final chipRadiusX = ((tableWidth / 2) - 30) * 0.6;
    final chipRadiusY = ((tableHeight / 2) + 10) * 0.6;

    final x = chipRadiusX * cos(angle);
    final y = chipRadiusY * sin(angle);

    return Positioned(
      left: (tableWidth / 2) + x - 20, // Center chip stack
      top: (tableHeight / 2) + y - 30,
      child: ChipStackWidget(
        amount: player.currentBet,
        smallBlind: widget.smallBlind,
        scale: 1.0,
        showAmount: true,
      ),
    );
  }

  Widget _buildDealingCardsAnimation(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    // Find player at this seat
    final player = widget.players.where((p) => p.seat == seatIndex).firstOrNull;

    // Only show animation if this player is being dealt cards
    if (player == null || !_dealingCardsTo.contains(player.userId)) {
      return const SizedBox.shrink();
    }

    // Calculate target position (same as seat position but offset for cards)
    final angle = (2 * pi * seatIndex / widget.maxSeats) - pi / 2;
    final radiusX = (tableWidth / 2) - 30;
    final radiusY = (tableHeight / 2) + 10;

    final targetX = radiusX * cos(angle);
    final targetY = radiusY * sin(angle);

    return AnimatedPositioned(
      duration: const Duration(milliseconds: 150),
      curve: Curves.easeOut,
      // Start from center, move to player position
      left: (tableWidth / 2) + targetX - 40,
      top: (tableHeight / 2) + targetY - 120, // Above seat (where cards appear)
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 150),
        opacity: 1.0,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: (player.holeCards ?? [])
              .map(
                (card) => CardWidget(
                  card: card,
                  width: 35,
                  height: 50,
                  isShowdown: false,
                ),
              )
              .toList(),
        ),
      ),
    );
  }
}
