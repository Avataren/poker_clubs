import 'package:flutter/material.dart';
import '../models/card.dart';

class CardWidget extends StatelessWidget {
  final PokerCard card;
  final double width;
  final double height;
  final bool isShowdown;

  const CardWidget({
    super.key,
    required this.card,
    this.width = 60,
    this.height = 85,
    this.isShowdown = false,
  });

  @override
  Widget build(BuildContext context) {
    // Determine opacity: during showdown, fade cards that aren't highlighted
    final opacity = isShowdown && !card.highlighted ? 0.3 : 1.0;

    // Debug: print card state during showdown
    if (isShowdown && card.faceUp) {
      print(
        'Card ${card.rankStr}${card.suitStr}: highlighted=${card.highlighted}, opacity=$opacity',
      );
    }

    return Opacity(
      opacity: opacity,
      child: Container(
        width: width,
        height: height,
        margin: const EdgeInsets.symmetric(horizontal: 4),
        decoration: BoxDecoration(
          color: card.faceUp ? Colors.white : Colors.blue[900],
          borderRadius: BorderRadius.circular(8),
          border: card.faceUp
              ? (card.highlighted && isShowdown
                    ? Border.all(color: Colors.amber, width: 3)
                    : null)
              : Border.all(color: Colors.white, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.3),
              blurRadius: 4,
              offset: const Offset(2, 2),
            ),
          ],
        ),
        child: card.faceUp
            ? Center(
                child: Text(
                  card.toString(),
                  style: TextStyle(
                    fontSize: width * 0.5,
                    fontWeight: FontWeight.bold,
                    color: card.isRed ? Colors.red : Colors.black,
                  ),
                ),
              )
            : Center(
                child: Icon(
                  Icons.style,
                  color: Colors.white,
                  size: width * 0.4,
                ),
              ),
      ),
    );
  }
}
