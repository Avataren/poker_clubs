import 'dart:math' as math;
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

  static const double _targetAspectRatio = 1.4;

  Color get _suitColor => card.isRed ? Colors.red[700]! : Colors.black87;

  double get _resolvedWidth {
    final widthFromHeight = height > 0 ? height / _targetAspectRatio : width;
    final resolvedWidth = math.min(width, widthFromHeight);
    return resolvedWidth > 0 ? resolvedWidth : width;
  }

  double get _resolvedHeight => _resolvedWidth * _targetAspectRatio;

  @override
  Widget build(BuildContext context) {
    final opacity = isShowdown && !card.highlighted ? 0.45 : 1.0;
    final cardWidth = _resolvedWidth;
    final cardHeight = _resolvedHeight;
    final borderRadius = BorderRadius.circular(cardWidth * 0.12);

    return Opacity(
      opacity: opacity,
      child: Container(
        width: cardWidth,
        height: cardHeight,
        margin: const EdgeInsets.symmetric(horizontal: 2),
        decoration: BoxDecoration(
          color: card.faceUp ? Colors.white : null,
          gradient: card.faceUp
              ? null
              : const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Color(0xFF1a237e),
                    Color(0xFF283593),
                    Color(0xFF1a237e),
                  ],
                ),
          borderRadius: borderRadius,
          border: card.faceUp
              ? (card.highlighted && isShowdown
                    ? Border.all(color: Colors.amber, width: 2.5)
                    : Border.all(color: Colors.grey[300]!, width: 0.5))
              : Border.all(color: Colors.white70, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.35),
              blurRadius: 4,
              offset: const Offset(1, 2),
            ),
          ],
        ),
        clipBehavior: Clip.antiAlias,
        child: card.faceUp
            ? _buildFaceUp(cardWidth, cardHeight)
            : _buildFaceDown(cardWidth),
      ),
    );
  }

  Widget _buildFaceUp(double cardWidth, double cardHeight) {
    final rankFontSize = (cardWidth * 0.26).clamp(8.0, 18.0);
    final suitFontSize = (cardWidth * 0.22).clamp(7.0, 15.0);
    final centerSuitSize = (cardWidth * 0.45).clamp(14.0, 32.0);

    return Stack(
      children: [
        // Top-left rank + suit
        Positioned(
          top: cardHeight * 0.04,
          left: cardWidth * 0.08,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                card.rankStr,
                style: TextStyle(
                  color: _suitColor,
                  fontSize: rankFontSize,
                  fontWeight: FontWeight.bold,
                  height: 1.0,
                ),
              ),
              Text(
                card.suitStr,
                style: TextStyle(
                  color: _suitColor,
                  fontSize: suitFontSize,
                  height: 1.0,
                ),
              ),
            ],
          ),
        ),
        // Center suit symbol
        Center(
          child: Text(
            card.suitStr,
            style: TextStyle(color: _suitColor, fontSize: centerSuitSize),
          ),
        ),
        // Bottom-right rank + suit (rotated)
        Positioned(
          bottom: cardHeight * 0.04,
          right: cardWidth * 0.08,
          child: Transform.rotate(
            angle: 3.14159,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  card.rankStr,
                  style: TextStyle(
                    color: _suitColor,
                    fontSize: rankFontSize,
                    fontWeight: FontWeight.bold,
                    height: 1.0,
                  ),
                ),
                Text(
                  card.suitStr,
                  style: TextStyle(
                    color: _suitColor,
                    fontSize: suitFontSize,
                    height: 1.0,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildFaceDown(double cardWidth) {
    return Container(
      margin: EdgeInsets.all(cardWidth * 0.08),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(cardWidth * 0.06),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.2),
          width: 0.5,
        ),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF303f9f), Color(0xFF1a237e)],
        ),
      ),
      child: CustomPaint(painter: _DiamondPatternPainter()),
    );
  }
}

class _DiamondPatternPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.08)
      ..strokeWidth = 0.5
      ..style = PaintingStyle.stroke;

    final spacing = size.width * 0.3;
    for (double y = 0; y < size.height; y += spacing) {
      for (double x = 0; x < size.width; x += spacing) {
        final path = Path()
          ..moveTo(x + spacing / 2, y)
          ..lineTo(x + spacing, y + spacing / 2)
          ..lineTo(x + spacing / 2, y + spacing)
          ..lineTo(x, y + spacing / 2)
          ..close();
        canvas.drawPath(path, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
