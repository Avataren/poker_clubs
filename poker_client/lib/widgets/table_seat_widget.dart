import 'dart:math';
import 'package:flutter/material.dart';
import '../models/game_state.dart';
import '../models/player.dart';
import 'card_widget.dart';
import 'chip_stack_widget.dart';

class _SeatVisualLayout {
  static const double playerScale = 1.6;
  static const double cardAspectRatio = 1.4;
  static const double _fanSpread = 0.34;
  static const double _fanRotation = 0.16;

  static double seatSizeForTable(double tableWidth) {
    final baseSeatSize = (tableWidth * 0.13).clamp(60.0, 90.0).toDouble();
    return baseSeatSize * playerScale;
  }

  static double cardWidthForTable(double tableWidth) {
    final baseCardWidth = (tableWidth * 0.058).clamp(28.0, 42.0).toDouble();
    return baseCardWidth * playerScale;
  }

  static double cardHeightForWidth(double cardWidth) {
    return cardWidth * cardAspectRatio;
  }

  static double frameWidth(double seatSize, double cardWidth) {
    return max(seatSize * 1.5, cardWidth * 2.45);
  }

  static double frameHeight(double seatSize, double cardHeight) {
    return max(seatSize * 2.0, seatSize * 1.4 + cardHeight);
  }

  static double seatVerticalOffset(double tableHeight) => tableHeight * 0.015;

  static double avatarSize(double seatSize) => seatSize * 0.58;

  static double avatarTop(double seatSize) => seatSize * 0.18;

  static double avatarCenterYInFrame(double seatSize) {
    final size = avatarSize(seatSize);
    return avatarTop(seatSize) + (size / 2);
  }

  static double namePlateTop(double seatSize, double avatarSize) {
    return avatarTop(seatSize) + (avatarSize * 0.78);
  }

  static double cardsTop(
    double seatSize,
    double avatarSize,
    double cardHeight,
  ) {
    final centerY = avatarTop(seatSize) + (avatarSize * 0.58);
    return centerY - (cardHeight / 2);
  }

  static double cardLeftInFrame(
    int cardIndex,
    int totalCards,
    double frameWidth,
    double cardWidth,
  ) {
    return (frameWidth - cardWidth) / 2 +
        cardFanOffsetX(cardIndex, totalCards, cardWidth);
  }

  static double cardFanOffsetX(
    int cardIndex,
    int totalCards,
    double cardWidth,
  ) {
    if (totalCards <= 1) return 0;
    final centerIndex = (totalCards - 1) / 2.0;
    return (cardIndex - centerIndex) * (cardWidth * _fanSpread);
  }

  static double cardFanRotation(int cardIndex, int totalCards) {
    if (totalCards <= 1) return 0;
    final centerIndex = (totalCards - 1) / 2.0;
    final spread = max(centerIndex, 1.0);
    return (cardIndex - centerIndex) * (_fanRotation / spread);
  }
}

class _SeatGeometry {
  final double seatSize;
  final double cardWidth;
  final double cardHeight;
  final double frameWidth;
  final double frameHeight;
  final double left;
  final double top;

  const _SeatGeometry({
    required this.seatSize,
    required this.cardWidth,
    required this.cardHeight,
    required this.frameWidth,
    required this.frameHeight,
    required this.left,
    required this.top,
  });
}

class TableSeatWidget extends StatelessWidget {
  final int seatNumber;
  final Player? player;
  final bool isCurrentTurn;
  final bool isMe;
  final VoidCallback? onTakeSeat;
  final VoidCallback? onRemoveBot;
  final bool showingDown;
  final bool isDealer;
  final bool isSmallBlind;
  final bool isBigBlind;
  final int smallBlind;
  final bool isDealingCards;
  final double seatSize;
  final double cardWidth;
  final double cardHeight;
  final String? lastAction;
  final String? avatarUrl;

  const TableSeatWidget({
    super.key,
    required this.seatNumber,
    this.player,
    this.isCurrentTurn = false,
    this.isMe = false,
    this.onTakeSeat,
    this.onRemoveBot,
    this.showingDown = false,
    this.isDealer = false,
    this.isSmallBlind = false,
    this.isBigBlind = false,
    this.smallBlind = 10,
    this.isDealingCards = false,
    this.seatSize = 80.0,
    this.cardWidth = 35.0,
    this.cardHeight = 50.0,
    this.lastAction,
    this.avatarUrl,
  });

  bool get _hasCards {
    return player != null &&
        player!.holeCards != null &&
        player!.holeCards!.isNotEmpty;
  }

  bool get _isDimmed {
    if (player == null) return false;
    return player!.isFolded || player!.isSittingOut || player!.isDisconnected;
  }

  @override
  Widget build(BuildContext context) {
    final isEmpty = player == null;

    final usernameFontSize = (seatSize * 0.14).clamp(12.0, 24.0);
    final stackFontSize = (seatSize * 0.13).clamp(11.0, 22.0);
    final badgeFontSize = (seatSize * 0.11).clamp(10.0, 20.0);
    final avatarSize = _SeatVisualLayout.avatarSize(seatSize);
    final frameWidth = _SeatVisualLayout.frameWidth(seatSize, cardWidth);
    final frameHeight = _SeatVisualLayout.frameHeight(seatSize, cardHeight);
    final avatarTop = _SeatVisualLayout.avatarTop(seatSize);
    final avatarLeft = (frameWidth - avatarSize) / 2;
    final namePlateTop = _SeatVisualLayout.namePlateTop(seatSize, avatarSize);
    final cardsTop = _SeatVisualLayout.cardsTop(
      seatSize,
      avatarSize,
      cardHeight,
    );

    if (isEmpty) {
      return _buildEmptySeat(
        frameWidth,
        frameHeight,
        avatarSize,
        avatarTop,
        usernameFontSize,
      );
    }

    return Opacity(
      opacity: _isDimmed ? 0.5 : 1.0,
      child: SizedBox(
        width: frameWidth,
        height: frameHeight,
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned(
              top: avatarTop,
              left: avatarLeft,
              child: _buildAvatar(avatarSize),
            ),

            Positioned(
              top: namePlateTop,
              left: 0,
              right: 0,
              child: Center(
                child: _buildNamePlate(
                  usernameFontSize,
                  stackFontSize,
                  avatarSize,
                ),
              ),
            ),

            // Cards are rendered in front of the avatar with a slight fan.
            if (_hasCards)
              ...List.generate(player!.holeCards!.length, (index) {
                final card = player!.holeCards![index];
                final totalCards = player!.holeCards!.length;

                return Positioned(
                  left: _SeatVisualLayout.cardLeftInFrame(
                    index,
                    totalCards,
                    frameWidth,
                    cardWidth,
                  ),
                  top: cardsTop,
                  child: Opacity(
                    opacity: isDealingCards ? 0.0 : 1.0,
                    child: Transform.rotate(
                      angle: _SeatVisualLayout.cardFanRotation(
                        index,
                        totalCards,
                      ),
                      child: CardWidget(
                        card: card,
                        width: cardWidth,
                        height: cardHeight,
                        isShowdown: showingDown,
                      ),
                    ),
                  ),
                );
              }),

            if (isDealer || isSmallBlind || isBigBlind)
              Positioned(
                top: avatarTop + (avatarSize * 0.06),
                left: avatarLeft + avatarSize * 0.9,
                child: _buildDealerBlindBadge(badgeFontSize),
              ),

            if (player!.isBot)
              Positioned(
                top: avatarTop - (avatarSize * 0.18),
                left: avatarLeft + avatarSize * 0.78,
                child: Container(
                  padding: EdgeInsets.all(seatSize * 0.025),
                  decoration: BoxDecoration(
                    color: Colors.grey[800],
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white54, width: 1),
                  ),
                  child: Icon(
                    Icons.smart_toy,
                    size: seatSize * 0.13,
                    color: Colors.white70,
                  ),
                ),
              ),

            if (lastAction != null && lastAction!.isNotEmpty)
              Positioned(
                top: namePlateTop - (seatSize * 0.22),
                left: 0,
                right: 0,
                child: Center(child: _buildLastActionBadge(badgeFontSize)),
              ),

            if (player!.isWinner && showingDown)
              Positioned(
                top: avatarTop + avatarSize * 0.2,
                left: 0,
                right: 0,
                child: Center(child: _buildWinnerBadge(badgeFontSize)),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptySeat(
    double frameWidth,
    double frameHeight,
    double avatarSize,
    double avatarTop,
    double fontSize,
  ) {
    final avatarLeft = (frameWidth - avatarSize) / 2;

    return GestureDetector(
      onTap: onTakeSeat,
      child: SizedBox(
        width: frameWidth,
        height: frameHeight,
        child: Stack(
          children: [
            Positioned(
              top: avatarTop,
              left: avatarLeft,
              child: Container(
                width: avatarSize,
                height: avatarSize,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white24, width: 1.5),
                ),
                child: Center(
                  child: Icon(
                    Icons.add,
                    color: Colors.white30,
                    size: avatarSize * 0.4,
                  ),
                ),
              ),
            ),
            Positioned(
              top: avatarTop + avatarSize + (seatSize * 0.08),
              left: 0,
              right: 0,
              child: Center(
                child: Text(
                  'Seat ${seatNumber + 1}',
                  style: TextStyle(
                    color: Colors.white30,
                    fontSize: fontSize * 0.9,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAvatar(double size) {
    return GestureDetector(
      onLongPress: (player!.isBot && onRemoveBot != null) ? onRemoveBot : null,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: const Color(0xFF2a3a5c),
          border: Border.all(
            color: isMe
                ? const Color(0xFF4caf50)
                : (isCurrentTurn ? Colors.amber : Colors.white38),
            width: isMe ? 2.5 : (isCurrentTurn ? 2.5 : 1.5),
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
          child: Text(
            player!.username.isNotEmpty
                ? player!.username[0].toUpperCase()
                : '?',
            style: TextStyle(
              color: Colors.white,
              fontSize: size * 0.4,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNamePlate(
    double usernameFontSize,
    double stackFontSize,
    double avatarSize,
  ) {
    return Container(
      constraints: BoxConstraints(maxWidth: seatSize * 1.25),
      padding: EdgeInsets.only(
        left: seatSize * 0.1,
        right: seatSize * 0.1,
        top: avatarSize * 0.16,
        bottom: seatSize * 0.05,
      ),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(seatSize * 0.08),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            player!.username,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: usernameFontSize,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 2),
          Text(
            '\$${player!.stack}',
            style: TextStyle(
              color: const Color(0xFF66bb6a),
              fontWeight: FontWeight.bold,
              fontSize: stackFontSize,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDealerBlindBadge(double fontSize) {
    final Color bgColor;
    final Color borderColor;
    final Color textColor;
    final String label;

    if (isDealer) {
      bgColor = Colors.white;
      borderColor = Colors.black54;
      textColor = Colors.black;
      label = 'D';
    } else if (isSmallBlind) {
      bgColor = Colors.blue[700]!;
      borderColor = Colors.blue[900]!;
      textColor = Colors.white;
      label = 'SB';
    } else {
      bgColor = Colors.orange[700]!;
      borderColor = Colors.orange[900]!;
      textColor = Colors.white;
      label = 'BB';
    }

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: seatSize * 0.06,
        vertical: seatSize * 0.025,
      ),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(seatSize * 0.1),
        border: Border.all(color: borderColor, width: 1),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: textColor,
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildLastActionBadge(double fontSize) {
    final action = lastAction!.toLowerCase();
    final Color bgColor;

    if (action.contains('fold')) {
      bgColor = Colors.red[700]!;
    } else if (action.contains('call')) {
      bgColor = Colors.orange[700]!;
    } else if (action.contains('raise') || action.contains('bet')) {
      bgColor = Colors.green[700]!;
    } else if (action.contains('check')) {
      bgColor = Colors.blue[600]!;
    } else if (action.contains('all')) {
      bgColor = Colors.purple[600]!;
    } else {
      bgColor = Colors.grey[700]!;
    }

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: seatSize * 0.07,
        vertical: seatSize * 0.02,
      ),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(seatSize * 0.06),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.4),
            blurRadius: 3,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Text(
        lastAction!,
        style: TextStyle(
          color: Colors.white,
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildWinnerBadge(double fontSize) {
    return Container(
      constraints: BoxConstraints(maxWidth: seatSize * 1.15),
      padding: EdgeInsets.symmetric(
        horizontal: seatSize * 0.075,
        vertical: seatSize * 0.025,
      ),
      decoration: BoxDecoration(
        color: Colors.amber[600],
        borderRadius: BorderRadius.circular(seatSize * 0.1),
        border: Border.all(color: Colors.amber[900]!, width: 1.5),
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
          Text(
            'WINNER',
            style: TextStyle(
              color: Colors.black,
              fontSize: fontSize,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (player!.winningHand != null && player!.winningHand!.isNotEmpty)
            Text(
              player!.winningHand!,
              style: TextStyle(
                color: Colors.black87,
                fontSize: fontSize * 0.9,
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
        ],
      ),
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
  final Map<String, int> _winnerPots = {}; // Map of userId to pot amount won

  // Card dealing animation
  final Map<String, int> _dealingCardsTo =
      {}; // Map of player ID to card number (1 or 2) being dealt
  final Map<String, bool> _cardAnimationStarted =
      {}; // Track if animation has started for player's current card
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
                  _cardAnimationStarted[player.userId] =
                      false; // Not started yet
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
                Future.delayed(
                  Duration(milliseconds: cardAnimDuration + 10),
                  () {
                    if (mounted) {
                      setState(() {
                        _dealingCardsTo.remove(player.userId);
                        _cardAnimationStarted.remove(player.userId);
                      });
                    }
                  },
                );
              }
            },
          );
        }
      });
    }

    _lastPhase = widget.gamePhase;

    // Don't trigger pot animation if one is already running
    if (_animatingPot) {
      // Table composition may change between frames (for example during
      // tournament table consolidation), so drop stale winner entries.
      _winnerPots.removeWhere(
        (winnerId, _) => !widget.players.any((p) => p.userId == winnerId),
      );
      if (_winnerPots.isEmpty) {
        _animatingPot = false;
      }
      return;
    }

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
          final winner = widget.players
              .where((p) => p.userId == entry.key)
              .firstOrNull;
          if (winner != null) {
            print('  - ${winner.username}: \$${entry.value}');
          }
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
        final centerOffset = Offset(
          tableWidth / 2,
          tableHeight / 2 - potTopOffset,
        );

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
              // Table surface — outer rim + inner felt
              Center(
                child: Container(
                  width: tableWidth * 0.78,
                  height: tableHeight * 0.78,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(
                      (tableHeight * 0.78) / 2,
                    ),
                    gradient: const LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Color(0xFF3a3a3a),
                        Color(0xFF2a2a2a),
                        Color(0xFF1e1e1e),
                      ],
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.5),
                        blurRadius: 20,
                        spreadRadius: 8,
                      ),
                    ],
                  ),
                  padding: EdgeInsets.all(borderWidth),
                  child: Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(
                        (tableHeight * 0.78 - borderWidth * 2) / 2,
                      ),
                      gradient: const RadialGradient(
                        center: Alignment(0.0, -0.1),
                        radius: 0.9,
                        colors: [
                          Color(0xFF1a7a3a), // lighter center
                          Color(0xFF0d5e2e), // mid green
                          Color(0xFF084420), // darker edges
                        ],
                        stops: [0.0, 0.5, 1.0],
                      ),
                      border: Border.all(
                        color: const Color(0xFF0a3d1a),
                        width: 2,
                      ),
                    ),
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
                  final winner = widget.players
                      .where((p) => p.userId == winnerId)
                      .firstOrNull;
                  if (winner == null) {
                    return const SizedBox.shrink();
                  }

                  // Calculate winner seat position
                  final angle =
                      (2 * pi * winner.seat / widget.maxSeats) - pi / 2;
                  final radiusX = tableWidth * 0.48;
                  final radiusY = tableHeight * 0.49;
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
    final fontSize = (tableWidth * 0.025).clamp(10.0, 16.0);

    // Build pot text
    String potText;
    if (pots.length <= 1) {
      potText = 'Pot: \$${widget.potTotal}';
    } else {
      final parts = <String>[];
      for (int i = 0; i < pots.length; i++) {
        if (pots[i].amount > 0) {
          parts.add(
            i == 0 ? 'Main: \$${pots[i].amount}' : 'Side: \$${pots[i].amount}',
          );
        }
      }
      potText = parts.join('  |  ');
    }

    return [
      Positioned(
        left: centerOffset.dx,
        top: centerOffset.dy,
        child: FractionalTranslation(
          translation: const Offset(-0.5, -0.5),
          child: Container(
            padding: EdgeInsets.symmetric(
              horizontal: tableWidth * 0.025,
              vertical: tableWidth * 0.01,
            ),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.65),
              borderRadius: BorderRadius.circular(tableWidth * 0.015),
              border: Border.all(
                color: const Color(0xFF4caf50).withValues(alpha: 0.6),
                width: 1.5,
              ),
            ),
            child: Text(
              potText,
              style: TextStyle(
                color: Colors.lightGreenAccent,
                fontSize: fontSize,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
      ),
    ];
  }

  _SeatGeometry _seatGeometryForIndex(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    final angle = (2 * pi * seatIndex / widget.maxSeats) - pi / 2;
    final radiusX = (tableWidth / 2) * 0.96;
    final radiusY = (tableHeight / 2) * 0.98;
    final x = radiusX * cos(angle);
    final y = radiusY * sin(angle);

    final seatSize = _SeatVisualLayout.seatSizeForTable(tableWidth);
    final cardWidth = _SeatVisualLayout.cardWidthForTable(tableWidth);
    final cardHeight = _SeatVisualLayout.cardHeightForWidth(cardWidth);
    final frameWidth = _SeatVisualLayout.frameWidth(seatSize, cardWidth);
    final frameHeight = _SeatVisualLayout.frameHeight(seatSize, cardHeight);

    return _SeatGeometry(
      seatSize: seatSize,
      cardWidth: cardWidth,
      cardHeight: cardHeight,
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      left: (tableWidth / 2) + x - (frameWidth / 2),
      top:
          (tableHeight / 2) +
          y -
          _SeatVisualLayout.avatarCenterYInFrame(seatSize) -
          _SeatVisualLayout.seatVerticalOffset(tableHeight),
    );
  }

  Widget _buildSeatAtPosition(
    int seatIndex,
    double tableWidth,
    double tableHeight,
  ) {
    final geometry = _seatGeometryForIndex(seatIndex, tableWidth, tableHeight);

    // Find player at this seat - exclude eliminated players (they're invisible)
    final player = widget.players
        .where((p) => p.seat == seatIndex && !p.isEliminated)
        .firstOrNull;
    final isMe = player?.userId == widget.myUserId;
    final isCurrentTurn =
        widget.gamePhase.toLowerCase() != 'waiting' &&
        widget.currentPlayerSeat == seatIndex;

    return Positioned(
      left: geometry.left,
      top: geometry.top,
      child: SizedBox(
        width: geometry.frameWidth,
        height: geometry.frameHeight,
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
          isDealer: widget.dealerSeat == seatIndex,
          isSmallBlind: widget.smallBlindSeat == seatIndex,
          isBigBlind: widget.bigBlindSeat == seatIndex,
          smallBlind: widget.smallBlind,
          isDealingCards:
              player != null && _dealingCardsTo.containsKey(player.userId),
          seatSize: geometry.seatSize,
          cardWidth: geometry.cardWidth,
          cardHeight: geometry.cardHeight,
          lastAction: player?.lastAction,
          avatarUrl: player?.avatarUrl,
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
    final chipRadiusX = ((tableWidth / 2) * 0.96) * 0.6;
    final chipRadiusY = ((tableHeight / 2) * 0.98) * 0.6;

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

    final geometry = _seatGeometryForIndex(seatIndex, tableWidth, tableHeight);
    final totalCards = player.holeCards?.length ?? 2;
    final cardIndex = (cardNumber - 1).clamp(0, totalCards - 1).toInt();
    final targetCardLeft =
        geometry.left +
        _SeatVisualLayout.cardLeftInFrame(
          cardIndex,
          totalCards,
          geometry.frameWidth,
          geometry.cardWidth,
        );
    final targetCardTop =
        geometry.top +
        _SeatVisualLayout.cardsTop(
          geometry.seatSize,
          _SeatVisualLayout.avatarSize(geometry.seatSize),
          geometry.cardHeight,
        );
    final targetRotation = _SeatVisualLayout.cardFanRotation(
      cardIndex,
      totalCards,
    );

    // Check if animation has started (determines position)
    final bool animationStarted = _cardAnimationStarted[player.userId] ?? false;

    return AnimatedPositioned(
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeOut,
      left: animationStarted
          ? targetCardLeft
          : (tableWidth / 2) - (geometry.cardWidth / 2),
      top: animationStarted
          ? targetCardTop
          : (tableHeight / 2) - (geometry.cardHeight / 2),
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 100),
        opacity: 1.0,
        child: Transform.rotate(
          angle: targetRotation,
          child: CardWidget(
            card: card,
            width: geometry.cardWidth,
            height: geometry.cardHeight,
            isShowdown: false,
          ),
        ),
      ),
    );
  }
}
