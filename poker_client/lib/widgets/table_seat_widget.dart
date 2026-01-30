import 'dart:math';
import 'package:flutter/material.dart';
import '../models/player.dart';
import '../models/card.dart';
import 'card_widget.dart';

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
  });

  bool get _hasCards {
    return player != null && 
           player!.holeCards != null && 
           player!.holeCards!.isNotEmpty;
  }

  bool get _shouldShowCards {
    if (!_hasCards) return false;

    // Show face-up cards (the backend sends face-up cards for the viewing player
    // and face-down cards for other active players)
    return player!.holeCards!.any((card) => card.faceUp);
  }

  bool get _shouldShowCardBacks {
    if (!_hasCards) return false;

    // Show face-down cards
    return player!.holeCards!.any((card) => !card.faceUp);
  }

  @override
  Widget build(BuildContext context) {
    final isEmpty = player == null;

    return Column(
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
              onLongPress: (!isEmpty && player!.isBot && onRemoveBot != null)
                  ? onRemoveBot
                  : null,
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isEmpty
                      ? Colors.black26
                      : (isCurrentTurn ? Colors.amber[700] : Colors.blue[900]),
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
                  padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
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
                    isDealer ? 'D' : isSmallBlind ? 'SB' : 'BB',
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
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
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
    );
  }
}

class PokerTableWidget extends StatelessWidget {
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
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate table size with fixed aspect ratio
        final maxWidth = constraints.maxWidth * 0.7;
        final maxHeight = constraints.maxHeight * 0.7;

        // Oval table aspect ratio (width:height = 1.6:1)
        double tableWidth, tableHeight;
        if (maxWidth / maxHeight > 1.6) {
          tableHeight = maxHeight;
          tableWidth = tableHeight * 1.6;
        } else {
          tableWidth = maxWidth;
          tableHeight = tableWidth / 1.6;
        }

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
              for (int i = 0; i < maxSeats; i++)
                _buildSeatAtPosition(i, tableWidth, tableHeight),
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
    final angle = (2 * pi * seatIndex / maxSeats) - pi / 2; // Start from top

    // Oval dimensions - seats positioned outside the table surface
    // Table surface is 70% of container, so we position seats around the full container
    final radiusX = (tableWidth / 2) - 30;
    final radiusY =
        (tableHeight / 2) +
        10; // Even larger Y radius to move seats away from center

    final x = radiusX * cos(angle);
    final y = radiusY * sin(angle);

    // Find player at this seat
    final player = players.where((p) => p.seat == seatIndex).firstOrNull;
    final isMe = player?.userId == myUserId;
    final isCurrentTurn =
        gamePhase.toLowerCase() != 'waiting' && currentPlayerSeat == seatIndex;

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
        onTakeSeat: player == null && onTakeSeat != null
            ? () => onTakeSeat!(seatIndex)
            : null,
        onRemoveBot: player != null && player.isBot && onRemoveBot != null
            ? () => onRemoveBot!(player.userId)
            : null,
        showingDown: showingDown,
        winningHand: winningHand,
        isDealer: dealerSeat == seatIndex,
        isSmallBlind: smallBlindSeat == seatIndex,
        isBigBlind: bigBlindSeat == seatIndex,
      ),
    );
  }
}
