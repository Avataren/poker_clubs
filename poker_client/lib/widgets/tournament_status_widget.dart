import 'package:flutter/material.dart';
import '../models/tournament.dart';

class TournamentStatusWidget extends StatelessWidget {
  final Tournament tournament;
  final int registeredCount;

  const TournamentStatusWidget({
    super.key,
    required this.tournament,
    required this.registeredCount,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Tournament Info',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                _buildStatusBadge(tournament.status),
              ],
            ),
            const SizedBox(height: 16),
            _buildInfoRow('Type', tournament.tournamentType.toUpperCase()),
            _buildInfoRow('Buy-in', '\$${_formatChips(tournament.buyIn)}'),
            _buildInfoRow(
              'Players',
              '$registeredCount / ${tournament.maxPlayers}',
            ),
            _buildInfoRow(
              'Starting Stack',
              '${_formatChips(tournament.startingStack)} chips',
            ),
            _buildInfoRow(
              'Level Duration',
              '${tournament.levelDurationMins} minutes',
            ),
            if (tournament.scheduledStart != null)
              _buildInfoRow(
                'Scheduled Start',
                _formatDateTime(tournament.scheduledStart!),
              ),
            if (tournament.actualStart != null)
              _buildInfoRow(
                'Started At',
                _formatDateTime(tournament.actualStart!),
              ),
            if (tournament.finishedAt != null)
              _buildInfoRow(
                'Finished At',
                _formatDateTime(tournament.finishedAt!),
              ),
            const SizedBox(height: 12),
            _buildProgressBar(registeredCount, tournament.maxPlayers),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusBadge(String status) {
    Color color;
    String label;

    switch (status) {
      case 'registering':
        color = Colors.blue;
        label = 'Registration Open';
        break;
      case 'running':
        color = Colors.orange;
        label = 'In Progress';
        break;
      case 'finished':
        color = Colors.grey;
        label = 'Finished';
        break;
      case 'cancelled':
        color = Colors.red;
        label = 'Cancelled';
        break;
      default:
        color = Colors.grey;
        label = status;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
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

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 14, color: Colors.grey[600])),
          Text(
            value,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressBar(int current, int max) {
    final progress = max > 0 ? current / max : 0.0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Registration Progress',
          style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: progress,
            minHeight: 8,
            backgroundColor: Colors.grey[300],
            valueColor: AlwaysStoppedAnimation<Color>(
              progress >= 1.0 ? Colors.red : Colors.green,
            ),
          ),
        ),
      ],
    );
  }

  String _formatChips(int chips) {
    if (chips >= 1000000) {
      return '${(chips / 1000000).toStringAsFixed(1)}M';
    } else if (chips >= 1000) {
      return '${(chips / 1000).toStringAsFixed(1)}K';
    }
    return chips.toString();
  }

  String _formatDateTime(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
