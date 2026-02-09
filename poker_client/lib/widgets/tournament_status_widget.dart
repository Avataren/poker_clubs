import 'dart:async';
import 'package:flutter/material.dart';
import '../models/tournament.dart';
import '../utils/format_utils.dart';

class TournamentStatusWidget extends StatefulWidget {
  final Tournament tournament;
  final int registeredCount;

  const TournamentStatusWidget({
    super.key,
    required this.tournament,
    required this.registeredCount,
  });

  @override
  State<TournamentStatusWidget> createState() =>
      _TournamentStatusWidgetState();
}

class _TournamentStatusWidgetState extends State<TournamentStatusWidget> {
  Timer? _countdownTimer;
  Duration _remaining = Duration.zero;

  bool get _isPreStart =>
      widget.tournament.status == 'registering' ||
      widget.tournament.status == 'seating';

  @override
  void initState() {
    super.initState();
    _startCountdownIfNeeded();
  }

  @override
  void didUpdateWidget(covariant TournamentStatusWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.tournament.status != widget.tournament.status ||
        oldWidget.tournament.scheduledStart != widget.tournament.scheduledStart) {
      _startCountdownIfNeeded();
    }
  }

  void _startCountdownIfNeeded() {
    _countdownTimer?.cancel();
    _countdownTimer = null;

    if (!_isPreStart || widget.tournament.scheduledStart == null) return;

    _updateRemaining();
    _countdownTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) => _updateRemaining(),
    );
  }

  void _updateRemaining() {
    final diff =
        widget.tournament.scheduledStart!.difference(DateTime.now().toUtc());
    setState(() {
      _remaining = diff.isNegative ? Duration.zero : diff;
    });
    if (diff.isNegative) {
      _countdownTimer?.cancel();
      _countdownTimer = null;
    }
  }

  @override
  void dispose() {
    _countdownTimer?.cancel();
    super.dispose();
  }

  String _formatCountdown(Duration d) {
    final hours = d.inHours;
    final minutes = d.inMinutes.remainder(60);
    final seconds = d.inSeconds.remainder(60);
    if (hours > 0) {
      return '${hours}h ${minutes}m ${seconds}s';
    } else if (minutes > 0) {
      return '${minutes}m ${seconds}s';
    }
    return '${seconds}s';
  }

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
                _buildStatusBadge(widget.tournament.status),
              ],
            ),
            const SizedBox(height: 16),
            _buildInfoRow(
                'Type', widget.tournament.tournamentType.toUpperCase()),
            _buildInfoRow(
                'Buy-in', '\$${FormatUtils.formatChips(widget.tournament.buyIn)}'),
            _buildInfoRow(
              'Prize Pool',
              '\$${FormatUtils.formatChips(widget.tournament.prizePool)}',
            ),
            _buildInfoRow(
              'Players',
              '${widget.registeredCount} / ${widget.tournament.maxPlayers}',
            ),
            _buildInfoRow(
              'Starting Stack',
              '${FormatUtils.formatChips(widget.tournament.startingStack)} chips',
            ),
            _buildInfoRow(
              'Level Duration',
              '${widget.tournament.levelDurationMins} minutes',
            ),
            if (widget.tournament.scheduledStart != null && _isPreStart)
              _buildCountdownRow()
            else if (widget.tournament.scheduledStart != null)
              _buildInfoRow(
                'Scheduled Start',
                FormatUtils.formatAbsolute(widget.tournament.scheduledStart!),
              ),
            if (widget.tournament.actualStart != null)
              _buildInfoRow(
                'Started At',
                FormatUtils.formatAbsolute(widget.tournament.actualStart!),
              ),
            if (widget.tournament.finishedAt != null)
              _buildInfoRow(
                'Finished At',
                FormatUtils.formatAbsolute(widget.tournament.finishedAt!),
              ),
            const SizedBox(height: 12),
            _buildProgressBar(
                widget.registeredCount, widget.tournament.maxPlayers),
          ],
        ),
      ),
    );
  }

  Widget _buildCountdownRow() {
    final isUrgent = _remaining.inMinutes < 5;
    final color = _remaining == Duration.zero
        ? Colors.grey
        : isUrgent
            ? Colors.red
            : Colors.orange;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('Starts In',
              style: TextStyle(fontSize: 14, color: Colors.grey[600])),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              _remaining == Duration.zero
                  ? 'Starting...'
                  : _formatCountdown(_remaining),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
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
      case 'seating':
        color = Colors.teal;
        label = 'Seating';
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

}
