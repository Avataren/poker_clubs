import 'package:flutter/material.dart';
import 'dart:math';

/// Compact vertical popup showing preset bet sizes.
/// Tapping a preset fires immediately. Width stretches to fill parent.
class BetSizingPopup extends StatelessWidget {
  final bool isPreflop;
  final int bigBlind;
  final int potTotal;
  final int currentBet;
  final int playerCurrentBet;
  final int playerStack;
  final int minRaise;
  final ValueChanged<int> onSelect;

  const BetSizingPopup({
    super.key,
    required this.isPreflop,
    required this.bigBlind,
    required this.potTotal,
    required this.currentBet,
    required this.playerCurrentBet,
    required this.playerStack,
    required this.minRaise,
    required this.onSelect,
  });

  int get _minBet {
    final effectiveMinRaise = max(minRaise, bigBlind);
    return max(effectiveMinRaise, bigBlind);
  }

  int get _maxBet {
    final toCall = max(0, currentBet - playerCurrentBet);
    return max(playerStack - toCall, _minBet);
  }

  int get _effectivePot {
    final toCall = max(0, currentBet - playerCurrentBet);
    return potTotal + toCall;
  }

  List<_Preset> get _presets {
    if (isPreflop) {
      return [
        _Preset('4 BB', bigBlind * 4),
        _Preset('3 BB', bigBlind * 3),
        _Preset('2.5 BB', (bigBlind * 2.5).round()),
        _Preset('2 BB', bigBlind * 2),
      ];
    } else {
      final pot = _effectivePot;
      return [
        _Preset('Pot', pot),
        _Preset('75%', (pot * 0.75).round()),
        _Preset('50%', (pot * 0.5).round()),
        _Preset('33%', (pot * 0.33).round()),
      ];
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey[900]!.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          for (final preset in _presets)
            _buildRow(preset.label, preset.amount),
        ],
      ),
    );
  }

  Widget _buildRow(String label, int amount) {
    final clamped = amount.clamp(_minBet, _maxBet);

    return GestureDetector(
      onTap: () => onSelect(clamped),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 7),
        margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
        decoration: BoxDecoration(
          color: Colors.green.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Text(
          label,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _Preset {
  final String label;
  final int amount;
  const _Preset(this.label, this.amount);
}
