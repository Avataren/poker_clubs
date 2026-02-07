import 'dart:math';
import 'package:flutter/material.dart';
import '../models/game_state.dart';
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
  final bool isDealingCards;
  final double seatSize;
  final double cardWidth;
  final double cardHeight;

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
    this.isDealingCards = false,
    this.seatSize = 80.0,
    this.cardWidth = 35.0,
    this.cardHeight = 50.0,
  });

  bool get _hasCards {
    return player != null &&
        player!.holeCards != null &&
        player!.holeCards!.isNotEmpty;
  }

  @override
  Widget build(BuildContext context) {
    final isEmpty = player == null;
    
    // Calculate responsive font sizes
    final usernameFontSize = (seatSize * 0.14).clamp(9.0, 13.0);
    final stackFontSize = (seatSize * 0.15).clamp(10.0, 14.0);
    final betFontSize = (seatSize * 0.11).clamp(8.0, 11.0);
    final badgeFontSize = (seatSize * 0.11).clamp(8.0, 11.0);

    return Stack(
      clipBehavior: Clip.none,
      alignment: Alignment.center,
      children: [
        Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Cards (above the seat circle)
            // Hide cards during dealing animation to avoid showing both animated and static cards
            // Use Opacity instead of conditionally rendering to prevent layout shifts
            SizedBox(
              height: cardHeight,
              child: _hasCards
                  ? Opacity(
                      opacity: isDealingCards ? 0.0 : 1.0,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: player!.holeCards!
                            .map(
                              (card) => CardWidget(
                                card: card,
                                width: cardWidth,
                                height: cardHeight,
                                isShowdown: showingDown,
                              ),
                            )
                            .toList(),
                      ),
                    )
                  : null,
            ),

            SizedBox(height: seatSize * 0.05),

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
                    width: seatSize,
                    height: seatSize,
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
                              style: TextStyle(
                                color: Colors.white54,
                                fontSize: usernameFontSize,
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
                                    fontSize: usernameFontSize,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                                SizedBox(height: seatSize * 0.025),
                                Text(
                                  '\$${player!.stack}',
                                  style: TextStyle(
                                    color: isCurrentTurn
                                        ? Colors.black87
                                        : Colors.green[300],
                                    fontWeight: FontWeight.bold,
                                    fontSize: stackFontSize,
                                  ),
                                ),
                                if (player!.currentBet > 0)
                                  Text(
                                    'Bet: \$${player!.currentBet}',
                                    style: TextStyle(
                                      color: isCurrentTurn
                                          ? Colors.black54
                                          : Colors.white70,
                                      fontSize: betFontSize,
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
                    bottom: -(seatSize * 0.05),
                    left: -(seatSize * 0.05),
                    child: Container(
                      padding: EdgeInsets.symmetric(
                        horizontal: seatSize * 0.06,
                        vertical: seatSize * 0.025,
                      ),
                      decoration: BoxDecoration(
                        color: isDealer
                            ? Colors.white
                            : isSmallBlind
                            ? Colors.blue[700]
                            : Colors.orange[700],
                        borderRadius: BorderRadius.circular(seatSize * 0.1),
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
                          fontSize: badgeFontSize,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),

                // Bot indicator
                if (player != null && player!.isBot)
                  Positioned(
                    top: -(seatSize * 0.05),
                    right: -(seatSize * 0.05),
                    child: Container(
                      padding: EdgeInsets.all(seatSize * 0.025),
                      decoration: BoxDecoration(
                        color: Colors.grey[800],
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white54, width: 1),
                      ),
                      child: Icon(
                        Icons.smart_toy,
                        size: seatSize * 0.15,
                        color: Colors.white70,
                      ),
                    ),
                  ),

                // Winner badge overlay (on top of circle)
                if (player != null && player!.isWinner && showingDown)
                  Positioned(
                    child: Container(
                      constraints: BoxConstraints(maxWidth: seatSize * 0.875),
                      padding: EdgeInsets.symmetric(
                        horizontal: seatSize * 0.075,
                        vertical: seatSize * 0.025,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.amber[600],
                        borderRadius: BorderRadius.circular(seatSize * 0.1),
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
                          Text('🏆', style: TextStyle(fontSize: seatSize * 0.2)),
                          if (winningHand != null)
                            Text(
                              winningHand!,
                              style: TextStyle(
                                color: Colors.black,
                                fontSize: badgeFontSize,
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
                padding: EdgeInsets.only(top: seatSize * 0.05),
                child: Text(
                  player!.state,
                  style: TextStyle(
                    color: Colors.white54,
                    fontSize: betFontSize,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),

            // Last action indicator
            if (!isEmpty && player!.lastAction != null)
              Padding(
                padding: EdgeInsets.only(top: seatSize * 0.025),
                child: Container(
                  padding: EdgeInsets.symmetric(
                    horizontal: seatSize * 0.075,
                    vertical: seatSize * 0.025,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(seatSize * 0.1),
                  ),
                  child: Text(
                    player!.lastAction!,
                    style: TextStyle(
                      color: Colors.amber,
                      fontSize: betFontSize,
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
  final List<PotInfo> pots;

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
    this.pots = const [],
  });

  @override
  State<PokerTableWidget> createState() => _PokerTableWidgetState();
}

class _PokerTableWidgetState extends State<PokerTableWidget> {
  bool _animatingPot = false;
  bool _lastShowingDown = false;
  final Map<String, int> _winnerPots = {}; // Map of userId to pot amount won
  final double _lastTableWidth = 0;
  final double _lastTableHeight = 0;

  // Card dealing animation
  final Map<String, int> _dealingCardsTo =
      {}; // Map of player ID to card number (1 or 2) being dealt
  final Map<String, bool> _cardAnimationStarted = {}; // Track if animation has started for player's current card
  String? _lastPhase;

  @override
  void initState() {
    super.initState();
    _lastPhase = widget.gamePhase;
  }

  @override
  void dispose() {
    super.dispose();
  }

  @override
  void didUpdateWidget(PokerTableWidget oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Trigger animation when entering PreFlop phase
    if (_lastPhase != 'PreFlop' && widget.gamePhase == 'PreFlop') {
      print('DEBUG: Entering PreFlop - starting card dealing animation');

      // Reset pot animation state for new hand
      setState(() {
        _animatingPot = false;
        _winnerPots.clear();
      });

      // Clear any existing animation state
      _dealingCardsTo.clear();
      _cardAnimationStarted.clear();

      // Schedule card dealing animation with stagger
      // Note: shuffle sound plays immediately in game_screen.dart
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;

        // Get players with cards sorted by dealing order (starting from small blind)
        final playersWithCards = widget.players
            .where((p) => p.holeCards != null && p.holeCards!.isNotEmpty)
            .toList();

        if (playersWithCards.isEmpty) {
          print('DEBUG: No players with cards found');
          return;
        }

        print('DEBUG: Dealing to ${playersWithCards.length} players');

        // Sort players by seat position starting from small blind
        final smallBlindSeat = widget.smallBlindSeat;
        if (smallBlindSeat != null) {
          playersWithCards.sort((a, b) {
            int seatA = (a.seat - smallBlindSeat) % widget.maxSeats;
            int seatB = (b.seat - smallBlindSeat) % widget.maxSeats;
            return seatA.compareTo(seatB);
          });
        }

        final numPlayers = playersWithCards.length;
        // Time per card: 500ms total / (2 rounds) = 250ms per round
        // Delay between each card in a round: 250ms / numPlayers
        final delayBetweenCards = (250 / numPlayers).ceil();
        final cardAnimDuration = 200; // Each card animation duration

        // Deal first card to all players
        for (int i = 0; i < numPlayers; i++) {
          final player = playersWithCards[i];
          Future.delayed(Duration(milliseconds: i * delayBetweenCards), () {
            if (mounted) {
              setState(() {
                _dealingCardsTo[player.userId] = 1; // First card
                _cardAnimationStarted[player.userId] = false; // Not started yet
              });

              // Start the animation on next frame (to trigger AnimatedPositioned)
              Future.delayed(const Duration(milliseconds: 10), () {
                if (mounted) {
                  setState(() {
                    _cardAnimationStarted[player.userId] = true;
                  });
                }
              });

              // Remove from dealing set after animation completes
              Future.delayed(Duration(milliseconds: cardAnimDuration + 10), () {
                if (mounted) {
                  setState(() {
                    _dealingCardsTo.remove(player.userId);
                    _cardAnimationStarted.remove(player.userId);
                  });
                }
              });
            }
          });
        }

        // Deal second card to all players (after first round completes)
        final secondRoundDelay = numPlayers * delayBetweenCards;
        for (int i = 0; i < numPlayers; i++) {
          final player = playersWithCards[i];
          Future.delayed(
            Duration(milliseconds: secondRoundDelay + i * delayBetweenCards),
            () {
              if (mounted) {
                setState(() {
                  _dealingCardsTo[player.userId] = 2; // Second card
                  _cardAnimationStarted[player.userId] = false; // Not started yet
                });

                // Start the animation on next frame
                Future.delayed(const Duration(milliseconds: 10), () {
                  if (mounted) {
                    setState(() {
                      _cardAnimationStarted[player.userId] = true;
                    });
                  }
                });

                // Remove from dealing set after animation completes
                Future.delayed(Duration(milliseconds: cardAnimDuration + 10), () {
                  if (mounted) {
                    setState(() {
                      _dealingCardsTo.remove(player.userId);
                      _cardAnimationStarted.remove(player.userId);
                    });
                  }
                });
              }
            },
          );
        }
      });
    }

    _lastPhase = widget.gamePhase;

    // Don't trigger pot animation if one is already running
    if (_animatingPot) return;

    // Detect all winners and their pot amounts
    final winners = widget.players
        .where((p) => p.isWinner && p.potWon > 0)
        .toList();

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
          final winner = widget.players.firstWhere(
            (p) => p.userId == entry.key,
          );
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
        // Calculate table size with fixed aspect ratio - use 75% of available space
        // to leave room for seats positioned around the edges
        final maxWidth = constraints.maxWidth * 0.75;
        final maxHeight = constraints.maxHeight * 0.75;

        // Oval table aspect ratio (width:height = 1.6:1)
        double tableWidth, tableHeight;
        if (maxWidth / maxHeight > 1.6) {
          tableHeight = maxHeight;
          tableWidth = tableHeight * 1.6;
        } else {
          tableWidth = maxWidth;
          tableHeight = tableWidth / 1.6;
        }

        // Calculate pot center position - responsive to table size
        // Position pot above the center to avoid overlapping with community cards
        final potTopOffset = tableHeight * 0.25; // 25% from center
        final centerOffset = Offset(tableWidth / 2, tableHeight / 2 - potTopOffset);

        // Track dimension changes to force animation update on significant resize
        final dimensionKey =
            '${(tableWidth / 10).round()}_${(tableHeight / 10).round()}';
        
        // Calculate responsive border width
        final borderWidth = (tableWidth * 0.012).clamp(4.0, 10.0);

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
                    border: Border.all(color: Colors.brown, width: borderWidth),
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
                  final winner = widget.players.firstWhere(
                    (p) => p.userId == winnerId,
                  );

                  // Calculate winner seat position
                  final angle =
                      (2 * pi * winner.seat / widget.maxSeats) - pi / 2;
                  final radiusX = tableWidth * 0.35;
                  final radiusY = tableHeight * 0.35;
                  final targetX = radiusX * cos(angle);
                  final targetY = radiusY * sin(angle);
                  final seatOffset = tableHeight * 0.08; // Responsive offset
                  final targetOffset = Offset(
                    (tableWidth / 2) + targetX,
                    (tableHeight / 2) + targetY - seatOffset,
                  );

                  return AnimatedPositioned(
                    key: ValueKey('pot_${winnerId}_$dimensionKey'),
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
                          scale: (tableWidth / 600).clamp(0.7, 1.2),
                          showAmount: true,
                          textColor: Colors.lightGreenAccent,
                        ),
                      ),
                    ),
                  );
                })
              // Show centered pot(s) when not animating
              else if (widget.potTotal > 0)
                ..._buildPotDisplay(centerOffset, tableWidth),
            ],
          ),
        );
      },
    );
  }

  List<Widget> _buildPotDisplay(Offset centerOffset, double tableWidth) {
    final pots = widget.pots;
    final scale = (tableWidth / 500).clamp(0.8, 1.3);

    // Single pot or no detailed pots: show like before (no label)
    if (pots.length <= 1) {
      return [
        AnimatedPositioned(
          duration: const Duration(milliseconds: 0),
          left: centerOffset.dx,
          top: centerOffset.dy,
          child: FractionalTranslation(
            translation: const Offset(-0.5, 0),
            child: ChipStackWidget(
              amount: widget.potTotal,
              smallBlind: widget.smallBlind,
              scale: scale,
              showAmount: true,
              textColor: Colors.lightGreenAccent,
            ),
          ),
        ),
      ];
    }

    // Multiple pots: lay out horizontally, side pots slightly smaller
    return [
      AnimatedPositioned(
        duration: const Duration(milliseconds: 0),
        left: centerOffset.dx,
        top: centerOffset.dy,
        child: FractionalTranslation(
          translation: const Offset(-0.5, 0),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              for (int i = 0; i < pots.length; i++)
                if (pots[i].amount > 0)
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 4.0 * scale),
                    child: ChipStackWidget(
                      amount: pots[i].amount,
                      smallBlind: widget.smallBlind,
                      scale: i == 0 ? scale : scale * 0.85,
                      showAmount: true,
                      textColor: i == 0
                          ? Colors.lightGreenAccent
                          : Colors.yellow[200]!,
                    ),
                  ),
            ],
          ),
        ),
      ),
    ];
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
    // Reduce radius slightly to prevent overflow at edges
    final radiusX = (tableWidth / 2) * 0.92;
    final radiusY = (tableHeight / 2) * 0.95;

    final x = radiusX * cos(angle);
    final y = radiusY * sin(angle);

    // Find player at this seat - exclude eliminated players (they're invisible)
    final player = widget.players
        .where((p) => p.seat == seatIndex && !p.isEliminated)
        .firstOrNull;
    final isMe = player?.userId == widget.myUserId;
    final isCurrentTurn =
        widget.gamePhase.toLowerCase() != 'waiting' &&
        widget.currentPlayerSeat == seatIndex;

    // Calculate responsive seat size
    final seatSize = (tableWidth * 0.13).clamp(60.0, 90.0);
    final cardWidth = (tableWidth * 0.058).clamp(28.0, 42.0);
    final cardHeight = (tableHeight * 0.095).clamp(40.0, 60.0);

    return Positioned(
      left: (tableWidth / 2) + x - (seatSize / 2),
      top: (tableHeight / 2) + y - (seatSize + cardHeight + 8) / 2 - (tableHeight * 0.03),
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: seatSize * 1.2,
          maxHeight: seatSize + cardHeight + 50, // Extra space for badges and text
        ),
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
          isDealingCards: player != null && _dealingCardsTo.containsKey(player.userId),
          seatSize: seatSize,
          cardWidth: cardWidth,
          cardHeight: cardHeight,
        ),
      ),
    );
  }

  Widget _buildChipsAtPosition(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    // Find player at this seat - exclude eliminated players
    final player = widget.players
        .where((p) => p.seat == seatIndex && !p.isEliminated)
        .firstOrNull;

    // No chips if no player or no bet
    if (player == null || player.currentBet <= 0) {
      return const SizedBox.shrink();
    }

    // Calculate position around oval - same angle as seat
    final angle = (2 * pi * seatIndex / widget.maxSeats) - pi / 2;

    // Position chips at 60% of the seat radius (on the table, between player and center)
    final chipRadiusX = ((tableWidth / 2) * 0.92) * 0.6;
    final chipRadiusY = ((tableHeight / 2) * 0.95) * 0.6;

    final x = chipRadiusX * cos(angle);
    final y = chipRadiusY * sin(angle);

    // Responsive chip stack sizing
    final chipStackSize = (tableWidth * 0.067).clamp(30.0, 50.0);

    return Positioned(
      left: (tableWidth / 2) + x - (chipStackSize / 2),
      top: (tableHeight / 2) + y - (chipStackSize * 0.75),
      child: ChipStackWidget(
        amount: player.currentBet,
        smallBlind: widget.smallBlind,
        scale: (tableWidth / 600).clamp(0.7, 1.2),
        showAmount: true,
      ),
    );
  }

  Widget _buildDealingCardsAnimation(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    // Find player at this seat - exclude eliminated players
    final player = widget.players
        .where((p) => p.seat == seatIndex && !p.isEliminated)
        .firstOrNull;

    // Only show animation if this player is being dealt cards
    if (player == null || !_dealingCardsTo.containsKey(player.userId)) {
      return const SizedBox.shrink();
    }

    final cardNumber = _dealingCardsTo[player.userId]!;
    final card =
        (player.holeCards != null && player.holeCards!.length >= cardNumber)
        ? player.holeCards![cardNumber - 1]
        : null;

    if (card == null) {
      return const SizedBox.shrink();
    }

    // Calculate target position (same as seat position but offset for cards)
    final angle = (2 * pi * seatIndex / widget.maxSeats) - pi / 2;
    final radiusX = (tableWidth / 2) * 0.92;
    final radiusY = (tableHeight / 2) * 0.95;

    final targetX = radiusX * cos(angle);
    final targetY = radiusY * sin(angle);

    // Calculate responsive card dimensions
    final cardWidth = (tableWidth * 0.058).clamp(28.0, 42.0);
    final cardHeight = (tableHeight * 0.095).clamp(40.0, 60.0);
    final seatSize = (tableWidth * 0.13).clamp(60.0, 90.0);

    // Offset second card slightly to the right
    final cardOffset = (cardNumber - 1) * (cardWidth * 0.57);

    // Check if animation has started (determines position)
    final bool animationStarted = _cardAnimationStarted[player.userId] ?? false;

    return AnimatedPositioned(
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeOut,
      // Start from center, move to player position after animation starts
      left: animationStarted 
          ? (tableWidth / 2) + targetX - (seatSize / 2) + cardOffset
          : (tableWidth / 2) - (cardWidth / 2),
      top: animationStarted
          ? (tableHeight / 2) + targetY - (seatSize + cardHeight + 8) / 2
          : (tableHeight / 2) - (cardHeight / 2),
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 100),
        opacity: 1.0,
        child: CardWidget(
          card: card,
          width: cardWidth,
          height: cardHeight,
          isShowdown: false,
        ),
      ),
    );
  }
}
