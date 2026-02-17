import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:math';

/// Full betting action panel shown when it's the player's turn.
///
/// Single horizontal row of fixed-size buttons:
/// [Fold] [Check/Call] [2BB] [3BB] [½Pot] [Pot] [$input][Raise]   [All In]
class BetActionPanel extends StatefulWidget {
  final bool isPreflop;
  final int bigBlind;
  final int potTotal;
  final int currentBet;
  final int playerCurrentBet;
  final int playerStack;
  final int minRaise;
  final bool canCheck;
  final int toCall;
  final void Function(String action, {int? amount}) onAction;

  const BetActionPanel({
    super.key,
    required this.isPreflop,
    required this.bigBlind,
    required this.potTotal,
    required this.currentBet,
    required this.playerCurrentBet,
    required this.playerStack,
    required this.minRaise,
    required this.canCheck,
    required this.toCall,
    required this.onAction,
  });

  @override
  State<BetActionPanel> createState() => _BetActionPanelState();
}

class _BetActionPanelState extends State<BetActionPanel> {
  late TextEditingController _amountController;

  int get _effectiveMinRaise => max(widget.minRaise, widget.bigBlind);

  int get _maxBet {
    final toCall = max(0, widget.currentBet - widget.playerCurrentBet);
    return max(widget.playerStack - toCall, _effectiveMinRaise);
  }

  int get _effectivePot {
    final toCall = max(0, widget.currentBet - widget.playerCurrentBet);
    return widget.potTotal + toCall;
  }

  bool get _canRaise => widget.playerStack > widget.toCall;

  @override
  void initState() {
    super.initState();
    _amountController = TextEditingController(text: '$_effectiveMinRaise');
  }

  @override
  void didUpdateWidget(BetActionPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.minRaise != widget.minRaise ||
        oldWidget.bigBlind != widget.bigBlind) {
      _amountController.text = '$_effectiveMinRaise';
    }
  }

  @override
  void dispose() {
    _amountController.dispose();
    super.dispose();
  }

  List<_Preset> get _presets {
    final raw = <_Preset>[];
    if (widget.isPreflop) {
      raw.addAll([
        _Preset('2BB', widget.bigBlind * 2),
        _Preset('3BB', widget.bigBlind * 3),
        _Preset('4BB', widget.bigBlind * 4),
        _Preset('\u00bdPot', (_effectivePot * 0.5).round()),
        _Preset('Pot', _effectivePot),
      ]);
    } else {
      final pot = _effectivePot;
      raw.addAll([
        _Preset('33%', (pot * 0.33).round()),
        _Preset('\u00bdPot', (pot * 0.5).round()),
        _Preset('\u00bePot', (pot * 0.75).round()),
        _Preset('Pot', pot),
        _Preset('2\u00d7Pot', pot * 2),
      ]);
    }

    final valid = <_Preset>[];
    final seen = <int>{};
    for (final p in raw) {
      final clamped = p.amount.clamp(_effectiveMinRaise, _maxBet);
      if (clamped >= _maxBet) continue;
      if (clamped < _effectiveMinRaise) continue;
      if (seen.contains(clamped)) continue;
      seen.add(clamped);
      valid.add(_Preset(p.label, clamped));
    }
    return valid;
  }

  void _firePreset(int amount) {
    widget.onAction('Raise', amount: amount);
  }

  void _fireManualRaise() {
    final parsed = int.tryParse(_amountController.text) ?? _effectiveMinRaise;
    final clamped = parsed.clamp(_effectiveMinRaise, _maxBet);
    widget.onAction('Raise', amount: clamped);
  }

  ButtonStyle _buttonStyle(Color color, {bool enabled = true}) {
    final bg = enabled ? color : Colors.grey[800]!;
    return ElevatedButton.styleFrom(
      backgroundColor: bg,
      foregroundColor: enabled ? Colors.black : Colors.white38,
      textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
      padding: const EdgeInsets.symmetric(horizontal: 26, vertical: 12),
      minimumSize: const Size(0, 44),
      fixedSize: const Size.fromHeight(44),
      elevation: enabled ? 3 : 0,
      shadowColor: Colors.black87,
      side: BorderSide(
        color: Colors.black.withValues(alpha: enabled ? 0.4 : 0.2),
        width: 1.5,
      ),
    );
  }

  ButtonStyle _presetStyle() {
    return ElevatedButton.styleFrom(
      backgroundColor: const Color(0xFF1B5E20),
      foregroundColor: Colors.greenAccent,
      textStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      minimumSize: const Size(0, 40),
      fixedSize: const Size.fromHeight(40),
      elevation: 2,
      shadowColor: Colors.black87,
      side: BorderSide(
        color: Colors.greenAccent.withValues(alpha: 0.3),
        width: 1,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 8, 8, 12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.black.withValues(alpha: 0.0),
            Colors.black.withValues(alpha: 0.85),
          ],
        ),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Fold
            ElevatedButton(
              onPressed: () => widget.onAction('Fold'),
              style: _buttonStyle(Colors.red),
              child: const Text('Fold'),
            ),
            const SizedBox(width: 6),
            // Check or Call
            if (widget.canCheck)
              ElevatedButton(
                onPressed: () => widget.onAction('Check'),
                style: _buttonStyle(Colors.blue),
                child: const Text('Check'),
              )
            else
              ElevatedButton(
                onPressed: widget.playerStack >= widget.toCall
                    ? () => widget.onAction('Call')
                    : null,
                style: _buttonStyle(
                  Colors.orange,
                  enabled: widget.playerStack >= widget.toCall,
                ),
                child: Text('Call \$${widget.toCall}'),
              ),
            // Preset bet sizes (only valid, affordable ones)
            if (_canRaise && _presets.isNotEmpty) ...[
              const SizedBox(width: 10),
              for (final preset in _presets) ...[
                ElevatedButton(
                  onPressed: () => _firePreset(preset.amount),
                  style: _presetStyle(),
                  child: Text(preset.label),
                ),
                const SizedBox(width: 4),
              ],
            ],
            // Manual input + Raise button
            if (_canRaise) ...[
              const SizedBox(width: 6),
              SizedBox(
                width: 80,
                height: 36,
                child: TextField(
                  controller: _amountController,
                  keyboardType: TextInputType.number,
                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  ),
                  decoration: InputDecoration(
                    prefixText: '\$',
                    prefixStyle: const TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                    filled: true,
                    fillColor: Colors.black54,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 6),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(6),
                      borderSide: const BorderSide(color: Colors.white24),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(6),
                      borderSide: const BorderSide(color: Colors.white24),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(6),
                      borderSide: const BorderSide(color: Colors.greenAccent),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 4),
              ElevatedButton(
                onPressed: _fireManualRaise,
                style: _buttonStyle(Colors.green),
                child: Text(widget.currentBet > 0 ? 'Raise' : 'Bet'),
              ),
            ],
            // All-In - spaced out from the rest
            const SizedBox(width: 20),
            ElevatedButton(
              onPressed: () => widget.onAction('AllIn'),
              style: _buttonStyle(Colors.purple),
              child: const Text('All In'),
            ),
          ],
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
