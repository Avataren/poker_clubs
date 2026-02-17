import 'package:flutter/material.dart';

/// A vertical wedge-shaped slider for bet sizing.
///
/// Narrow at the bottom (min bet), wide at the top (all-in).
/// Drag to select amount; bet fires on drag release only if value > minValue.
/// Simple taps are ignored to prevent accidental bets.
class WedgeSlider extends StatefulWidget {
  final int minValue;
  final int maxValue;
  final int value;
  final ValueChanged<int> onChanged;
  final ValueChanged<int> onBetConfirmed;

  const WedgeSlider({
    super.key,
    required this.minValue,
    required this.maxValue,
    required this.value,
    required this.onChanged,
    required this.onBetConfirmed,
  });

  @override
  State<WedgeSlider> createState() => _WedgeSliderState();
}

class _WedgeSliderState extends State<WedgeSlider> {
  bool _isDragging = false;
  bool _hasMoved = false;

  double _fractionFromValue(int value) {
    if (widget.maxValue <= widget.minValue) return 0.0;
    return ((value - widget.minValue) / (widget.maxValue - widget.minValue))
        .clamp(0.0, 1.0);
  }

  int _valueFromFraction(double fraction) {
    final f = fraction.clamp(0.0, 1.0);
    return (widget.minValue + f * (widget.maxValue - widget.minValue)).round();
  }

  void _handleDragStart(DragStartDetails details) {
    _isDragging = true;
    _hasMoved = false;
  }

  void _handleDragUpdate(
      DragUpdateDetails details, BoxConstraints constraints) {
    _hasMoved = true;
    final height = constraints.maxHeight;
    // Invert: top of widget = max, bottom = min
    final fraction = 1.0 - (details.localPosition.dy / height);
    final newValue = _valueFromFraction(fraction);
    widget.onChanged(newValue);
  }

  void _handleDragEnd(DragEndDetails details) {
    if (_isDragging && _hasMoved && widget.value >= widget.minValue) {
      widget.onBetConfirmed(widget.value);
    }
    _isDragging = false;
    _hasMoved = false;
  }

  @override
  Widget build(BuildContext context) {
    final fraction = _fractionFromValue(widget.value);

    return LayoutBuilder(
      builder: (context, constraints) {
        return GestureDetector(
          onVerticalDragStart: _handleDragStart,
          onVerticalDragUpdate: (d) => _handleDragUpdate(d, constraints),
          onVerticalDragEnd: _handleDragEnd,
          child: CustomPaint(
            size: Size(constraints.maxWidth, constraints.maxHeight),
            painter: _WedgePainter(
              fraction: fraction,
              label: '\$${widget.value}',
            ),
          ),
        );
      },
    );
  }
}

class _WedgePainter extends CustomPainter {
  final double fraction;
  final String label;

  _WedgePainter({required this.fraction, required this.label});

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;

    final bottomHalf = w * 0.15;
    final topHalf = w * 0.5;
    final centerX = w / 2;

    // Track background
    final trackPath = Path()
      ..moveTo(centerX - bottomHalf, h)
      ..lineTo(centerX + bottomHalf, h)
      ..lineTo(centerX + topHalf, 0)
      ..lineTo(centerX - topHalf, 0)
      ..close();

    canvas.drawPath(
      trackPath,
      Paint()
        ..color = const Color(0xFF2A2A3A)
        ..style = PaintingStyle.fill,
    );
    canvas.drawPath(
      trackPath,
      Paint()
        ..color = const Color(0xFF444466)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0,
    );

    // Fill portion
    if (fraction > 0.001) {
      final fillY = h * (1.0 - fraction);
      final fillHalf = bottomHalf + fraction * (topHalf - bottomHalf);

      final fillPath = Path()
        ..moveTo(centerX - bottomHalf, h)
        ..lineTo(centerX + bottomHalf, h)
        ..lineTo(centerX + fillHalf, fillY)
        ..lineTo(centerX - fillHalf, fillY)
        ..close();

      canvas.drawPath(
        fillPath,
        Paint()
          ..shader = const LinearGradient(
            begin: Alignment.bottomCenter,
            end: Alignment.topCenter,
            colors: [Color(0xFF2E7D32), Color(0xFF66BB6A)],
          ).createShader(Rect.fromLTWH(0, fillY, w, h - fillY))
          ..style = PaintingStyle.fill,
      );

      // Thumb line
      final lineHalf = fillHalf + 2;
      canvas.drawLine(
        Offset(centerX - lineHalf, fillY),
        Offset(centerX + lineHalf, fillY),
        Paint()
          ..color = Colors.greenAccent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5,
      );

      // Amount label
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.greenAccent,
            fontSize: 10,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      // Draw label background + text to the right of the wedge
      final labelX = centerX + lineHalf + 4;
      final labelY = fillY - textPainter.height / 2;

      final bgRect = RRect.fromRectAndRadius(
        Rect.fromLTWH(
          labelX - 3,
          labelY - 1,
          textPainter.width + 6,
          textPainter.height + 2,
        ),
        const Radius.circular(3),
      );
      canvas.drawRRect(
        bgRect,
        Paint()..color = const Color(0xDD000000),
      );
      textPainter.paint(canvas, Offset(labelX, labelY));
    }

    // Min label at bottom
    _drawLabel(canvas, '\$${label.isEmpty ? '' : ''}Min', centerX,
        h - 2, size, true);
    // All-In label at top
    _drawLabel(canvas, 'All-In', centerX, 10, size, false);
  }

  void _drawLabel(
      Canvas canvas, String text, double cx, double y, Size size, bool bottom) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.white54,
          fontSize: 8,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final dx = cx - tp.width / 2;
    final dy = bottom ? y - tp.height : y;
    tp.paint(canvas, Offset(dx, dy));
  }

  @override
  bool shouldRepaint(covariant _WedgePainter oldDelegate) {
    return oldDelegate.fraction != fraction || oldDelegate.label != label;
  }
}
