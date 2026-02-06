import 'package:flutter/material.dart';
import '../models/player.dart';
import 'card_widget.dart';

class PlayerWidget extends StatelessWidget {
  final Player player;
  final bool isMe;
  final bool isCurrentTurn;

  const PlayerWidget({
    super.key,
    required this.player,
    this.isMe = false,
    this.isCurrentTurn = false,
  });

  @override
  Widget build(BuildContext context) {
    final isInactive = player.isDisconnected || player.isSittingOut || player.isFolded || player.isEliminated;
    return Opacity(
      opacity: isInactive ? 0.5 : 1.0,
      child: Container(
        width: 160,
        margin: const EdgeInsets.all(8),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isCurrentTurn ? Colors.amber[700] : Colors.white10,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isMe ? Colors.green : Colors.transparent,
            width: 3,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    isMe ? '${player.username} (You)' : player.username,
                    style: TextStyle(
                      color: isCurrentTurn ? Colors.black : Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (isCurrentTurn)
                  const Icon(Icons.timer, color: Colors.black, size: 16),
                if (player.isDisconnected)
                  const Icon(Icons.wifi_off, color: Colors.red, size: 16),
              ],
            ),
          const SizedBox(height: 4),
          Text(
            'Stack: \$${player.stack}',
            style: TextStyle(
              color: isCurrentTurn ? Colors.black87 : Colors.green,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            'Bet: \$${player.currentBet}',
            style: TextStyle(
              color: isCurrentTurn ? Colors.black87 : Colors.white70,
              fontSize: 12,
            ),
          ),
          Text(
            player.state,
            style: TextStyle(
              color: isCurrentTurn ? Colors.black54 : Colors.white54,
              fontSize: 12,
              fontStyle: FontStyle.italic,
            ),
          ),
          if (player.holeCards != null && player.holeCards!.isNotEmpty) ...[
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: player.holeCards!
                  .map((card) => CardWidget(card: card, width: 40, height: 55))
                  .toList(),
            ),
          ],
        ],
      ),
      ),
    );
  }
}
