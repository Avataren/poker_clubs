import 'package:flutter/material.dart';
import '../models/tournament.dart';
import '../utils/format_utils.dart';

class TournamentCardWidget extends StatelessWidget {
  final TournamentWithStats tournamentStats;
  final VoidCallback? onTap;

  const TournamentCardWidget({
    super.key,
    required this.tournamentStats,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final tournament = tournamentStats.tournament;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      tournament.name,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  _buildStatusBadge(tournament.status),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  _buildInfoChip(
                    tournament.tournamentType.toUpperCase(),
                    Colors.blue,
                  ),
                  const SizedBox(width: 8),
                  _buildInfoChip(
                    'Buy-in: \$${FormatUtils.formatChips(tournament.buyIn)}',
                    Colors.green,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildInfoRow(
                    Icons.people,
                    '${tournamentStats.registeredCount}/${tournament.maxPlayers}',
                  ),
                  _buildInfoRow(
                    Icons.casino,
                    '${FormatUtils.formatChips(tournament.startingStack)} chips',
                  ),
                  _buildInfoRow(
                    Icons.timer,
                    '${tournament.levelDurationMins}m levels',
                  ),
                ],
              ),
              if (tournamentStats.isRegistered) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.green.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.check_circle, size: 16, color: Colors.green),
                      SizedBox(width: 4),
                      Text(
                        'Registered',
                        style: TextStyle(
                          color: Colors.green,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              if (tournament.scheduledStart != null) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Icon(Icons.schedule, size: 16),
                    const SizedBox(width: 4),
                    Text(
                      'Starts: ${FormatUtils.formatCountdown(tournament.scheduledStart!)}',
                      style: const TextStyle(fontSize: 12),
                    ),
                  ],
                ),
              ],
            ],
          ),
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
        label = 'Open';
        break;
      case 'running':
        color = Colors.orange;
        label = 'Running';
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
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
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

  Widget _buildInfoChip(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String text) {
    return Row(
      children: [
        Icon(icon, size: 16, color: Colors.grey),
        const SizedBox(width: 4),
        Text(text, style: const TextStyle(fontSize: 12)),
      ],
    );
  }

}
