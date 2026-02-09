import 'package:flutter/material.dart';

/// Shared status badge for tournament status display.
///
/// Use [compact] for shorter labels (e.g., in list cards).
class TournamentStatusBadge extends StatelessWidget {
  final String status;
  final bool compact;

  const TournamentStatusBadge({
    super.key,
    required this.status,
    this.compact = false,
  });

  @override
  Widget build(BuildContext context) {
    final (color, label) = _resolve(status, compact);

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 12, vertical: compact ? 4 : 6),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  static (Color, String) _resolve(String status, bool compact) {
    switch (status) {
      case 'registering':
        return (Colors.blue, compact ? 'Open' : 'Registration Open');
      case 'seating':
        return (Colors.teal, 'Seating');
      case 'running':
        return (Colors.orange, compact ? 'Running' : 'In Progress');
      case 'finished':
        return (Colors.grey, 'Finished');
      case 'cancelled':
        return (Colors.red, 'Cancelled');
      default:
        return (Colors.grey, status);
    }
  }
}
