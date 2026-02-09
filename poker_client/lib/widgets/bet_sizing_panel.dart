import 'package:flutter/material.dart';
import 'dart:math';

class BetSizingPanel extends StatefulWidget {
  final bool isPreflop;
  final int bigBlind;
  final int potTotal;
  final int currentBet;
  final int playerCurrentBet;
  final int playerStack;
  final int minRaise;
  final ValueChanged<int> onConfirm;
  final VoidCallback onCancel;

  const BetSizingPanel({
    super.key,
    required this.isPreflop,
    required this.bigBlind,
    required this.potTotal,
    required this.currentBet,
    required this.playerCurrentBet,
    required this.playerStack,
    required this.minRaise,
    required this.onConfirm,
    required this.onCancel,
  });

  @override
  State<BetSizingPanel> createState() => _BetSizingPanelState();
}

class _BetSizingPanelState extends State<BetSizingPanel> {
  late int _selectedAmount;
  late int _minBet;
  late int _maxBet;

  @override
  void initState() {
    super.initState();
    _computeLimits();
    _selectedAmount = _minBet;
  }

  void _computeLimits() {
    final toCall = max(0, widget.currentBet - widget.playerCurrentBet);
    // Min raise = current bet + min raise increment, but at least one big blind
    final effectiveMinRaise = max(widget.minRaise, widget.bigBlind);
    _minBet = max(effectiveMinRaise, widget.bigBlind);
    // Max is entire stack (which would be an all-in)
    _maxBet = max(widget.playerStack - toCall, _minBet);
  }

  /// The effective pot for pot-percentage calculations:
  /// pot + all current bets on table + what we'd need to call
  int get _effectivePot {
    final toCall = max(0, widget.currentBet - widget.playerCurrentBet);
    return widget.potTotal + toCall;
  }

  List<_PresetButton> get _presets {
    if (widget.isPreflop) {
      return [
        _PresetButton('2BB', widget.bigBlind * 2),
        _PresetButton('2.5BB', (widget.bigBlind * 2.5).round()),
        _PresetButton('3BB', widget.bigBlind * 3),
        _PresetButton('4BB', widget.bigBlind * 4),
      ];
    } else {
      final pot = _effectivePot;
      return [
        _PresetButton('33%', (pot * 0.33).round()),
        _PresetButton('50%', (pot * 0.5).round()),
        _PresetButton('75%', (pot * 0.75).round()),
        _PresetButton('100%', pot),
      ];
    }
  }

  void _setAmount(int amount) {
    setState(() {
      _selectedAmount = amount.clamp(_minBet, _maxBet);
    });
  }

  @override
  Widget build(BuildContext context) {
    final toCall = max(0, widget.currentBet - widget.playerCurrentBet);
    final totalCost = toCall + _selectedAmount;
    final isAllIn = totalCost >= widget.playerStack;

    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
      decoration: BoxDecoration(
        color: Colors.grey[900]!.withValues(alpha: 0.95),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Amount display
          Text(
            isAllIn ? 'ALL IN' : '\$$_selectedAmount',
            style: TextStyle(
              color: isAllIn ? Colors.amber : Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          // Preset buttons row
          Row(
            children: [
              // Presets
              Expanded(
                child: Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  alignment: WrapAlignment.center,
                  children: [
                    for (final preset in _presets)
                      _buildPresetChip(preset),
                    _buildPresetChip(
                      _PresetButton('ALL IN', _maxBet),
                      isAllIn: true,
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          // Slider
          Row(
            children: [
              Text(
                '\$$_minBet',
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
              Expanded(
                child: SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    activeTrackColor: Colors.green,
                    inactiveTrackColor: Colors.white24,
                    thumbColor: Colors.green,
                    overlayColor: Colors.green.withValues(alpha: 0.2),
                    trackHeight: 6,
                    thumbShape: const RoundSliderThumbShape(
                      enabledThumbRadius: 10,
                    ),
                  ),
                  child: Slider(
                    min: _minBet.toDouble(),
                    max: _maxBet.toDouble(),
                    value: _selectedAmount.toDouble().clamp(
                      _minBet.toDouble(),
                      _maxBet.toDouble(),
                    ),
                    onChanged: (v) => _setAmount(v.round()),
                  ),
                ),
              ),
              Text(
                '\$$_maxBet',
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(height: 6),
          // Confirm / Cancel row
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: widget.onCancel,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.grey[700],
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                  child: const Text('Cancel'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 2,
                child: ElevatedButton(
                  onPressed: () => widget.onConfirm(_selectedAmount),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    textStyle: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  child: Text(
                    isAllIn
                        ? 'All In (\$${widget.playerStack})'
                        : widget.currentBet > 0
                            ? 'Raise to \$${widget.currentBet + _selectedAmount}'
                            : 'Bet \$$_selectedAmount',
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPresetChip(_PresetButton preset, {bool isAllIn = false}) {
    final isSelected = _selectedAmount == preset.amount.clamp(_minBet, _maxBet);
    final effectiveAmount = preset.amount.clamp(_minBet, _maxBet);

    return GestureDetector(
      onTap: () => _setAmount(effectiveAmount),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: isAllIn
              ? (isSelected ? Colors.purple : Colors.purple.withValues(alpha: 0.3))
              : (isSelected ? Colors.green : Colors.white.withValues(alpha: 0.1)),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected
                ? (isAllIn ? Colors.purple : Colors.green)
                : Colors.white30,
          ),
        ),
        child: Text(
          preset.label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.white70,
            fontSize: 13,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

class _PresetButton {
  final String label;
  final int amount;
  const _PresetButton(this.label, this.amount);
}
