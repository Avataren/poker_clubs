import 'package:flutter/material.dart';
import 'dart:math' as math;

/// A widget that displays poker chips in realistic stacks.
/// Chips are organized by denomination (multiples of small blind).
class ChipStackWidget extends StatelessWidget {
  final int amount;
  final int smallBlind;
  final double scale;
  final bool showAmount;

  const ChipStackWidget({
    super.key,
    required this.amount,
    this.smallBlind = 10,
    this.scale = 1.0,
    this.showAmount = true,
  });

  @override
  Widget build(BuildContext context) {
    if (amount <= 0) return const SizedBox.shrink();

    // Calculate chip breakdown
    final chipBreakdown = _calculateChipBreakdown(amount, smallBlind);

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Chip stacks
        SizedBox(
          height: 40 * scale,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: _buildChipStacks(chipBreakdown),
          ),
        ),
        if (showAmount) ...[
          SizedBox(height: 4 * scale),
          Container(
            padding: EdgeInsets.symmetric(
              horizontal: 6 * scale,
              vertical: 2 * scale,
            ),
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(8 * scale),
              border: Border.all(color: Colors.amber, width: 1),
            ),
            child: Text(
              '\$$amount',
              style: TextStyle(
                color: Colors.amber,
                fontSize: 11 * scale,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ],
    );
  }

  /// Calculate chip breakdown into denominations
  /// Returns a map of denomination -> count
  Map<int, int> _calculateChipBreakdown(int total, int sb) {
    final Map<int, int> breakdown = {};

    // Define chip denominations based on small blind
    // Use: 100xSB, 25xSB, 5xSB, 1xSB
    final denominations = [sb * 100, sb * 25, sb * 5, sb];

    int remaining = total;

    for (final denom in denominations) {
      if (remaining >= denom) {
        final count = remaining ~/ denom;
        breakdown[denom] = count;
        remaining = remaining % denom;
      }
    }

    return breakdown;
  }

  /// Build chip stack widgets for each denomination
  List<Widget> _buildChipStacks(Map<int, int> breakdown) {
    final stacks = <Widget>[];

    // Sort by denomination (highest first)
    final sortedDenoms = breakdown.keys.toList()
      ..sort((a, b) => b.compareTo(a));

    for (final denom in sortedDenoms) {
      final count = breakdown[denom]!;
      stacks.add(_buildChipStack(denom, count));
      stacks.add(SizedBox(width: 2 * scale));
    }

    if (stacks.isNotEmpty) {
      stacks.removeLast(); // Remove last spacer
    }

    return stacks;
  }

  /// Build a single stack of chips of the same denomination
  Widget _buildChipStack(int denomination, int count) {
    // Limit visual stacking to 8 chips per stack
    final displayCount = math.min(count, 8);

    // Choose color based on denomination
    final color = _getChipColor(denomination);

    return SizedBox(
      width: 20 * scale,
      height: 40 * scale,
      child: Stack(
        alignment: Alignment.bottomCenter,
        children: List.generate(
          displayCount,
          (index) => Positioned(
            bottom: index * 3.0 * scale,
            child: _ChipWidget(
              color: color,
              size: 18 * scale,
              value: _formatDenomination(denomination),
            ),
          ),
        ),
      ),
    );
  }

  /// Get chip color based on denomination value
  Color _getChipColor(int denomination) {
    final sb = smallBlind;

    if (denomination >= sb * 100) {
      return Colors.black; // Black for highest
    } else if (denomination >= sb * 25) {
      return Colors.green; // Green for 25x
    } else if (denomination >= sb * 5) {
      return Colors.red; // Red for 5x
    } else {
      return Colors.white; // White for 1x (SB)
    }
  }

  /// Format denomination for display on chip
  String _formatDenomination(int denom) {
    if (denom >= 1000) {
      return '${denom ~/ 1000}K';
    } else if (denom >= 100) {
      return '${denom ~/ 100}H';
    }
    return '$denom';
  }
}

/// Individual chip widget
class _ChipWidget extends StatelessWidget {
  final Color color;
  final double size;
  final String value;

  const _ChipWidget({
    required this.color,
    required this.size,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size * 0.3, // Flattened circle (ellipse) for chip
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(size / 2),
        border: Border.all(color: _getBorderColor(color), width: size * 0.08),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 2,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Center(
        child: Text(
          value,
          style: TextStyle(
            color: _getTextColor(color),
            fontSize: size * 0.25,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }

  Color _getBorderColor(Color chipColor) {
    if (chipColor == Colors.white) {
      return Colors.grey[800]!;
    }
    return chipColor == Colors.black ? Colors.grey[600]! : Colors.white;
  }

  Color _getTextColor(Color chipColor) {
    if (chipColor == Colors.white) {
      return Colors.black;
    }
    return Colors.white;
  }
}

/// Animated chip widget that can move between positions
class AnimatedChipStack extends StatefulWidget {
  final int amount;
  final int smallBlind;
  final Offset startPosition;
  final Offset endPosition;
  final Duration duration;
  final VoidCallback? onComplete;
  final bool showAmount;

  const AnimatedChipStack({
    super.key,
    required this.amount,
    required this.smallBlind,
    required this.startPosition,
    required this.endPosition,
    this.duration = const Duration(milliseconds: 800),
    this.onComplete,
    this.showAmount = true,
  });

  @override
  State<AnimatedChipStack> createState() => _AnimatedChipStackState();
}

class _AnimatedChipStackState extends State<AnimatedChipStack>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _positionAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(duration: widget.duration, vsync: this);

    _positionAnimation = Tween<Offset>(
      begin: widget.startPosition,
      end: widget.endPosition,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));

    _controller.forward().then((_) {
      widget.onComplete?.call();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _positionAnimation,
      builder: (context, child) {
        return Positioned(
          left: _positionAnimation.value.dx,
          top: _positionAnimation.value.dy,
          child: ChipStackWidget(
            amount: widget.amount,
            smallBlind: widget.smallBlind,
            showAmount: widget.showAmount,
          ),
        );
      },
    );
  }
}
