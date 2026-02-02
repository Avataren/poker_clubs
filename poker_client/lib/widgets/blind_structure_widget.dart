import 'package:flutter/material.dart';
import '../models/tournament.dart';

class BlindStructureWidget extends StatelessWidget {
  final List<TournamentBlindLevel> blindLevels;
  final int? currentLevel;

  const BlindStructureWidget({
    super.key,
    required this.blindLevels,
    this.currentLevel,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              'Blind Structure',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
          const Divider(height: 1),
          ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: blindLevels.length,
            separatorBuilder: (context, index) => const Divider(height: 1),
            itemBuilder: (context, index) {
              final level = blindLevels[index];
              final isCurrentLevel =
                  currentLevel != null && level.level == currentLevel;

              return Container(
                color: isCurrentLevel ? Colors.green.withOpacity(0.1) : null,
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 60,
                        child: Text(
                          'Level ${level.level}',
                          style: TextStyle(
                            fontWeight: isCurrentLevel
                                ? FontWeight.bold
                                : FontWeight.normal,
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Row(
                          children: [
                            _buildBlindInfo(
                              'SB',
                              _formatChips(level.smallBlind),
                              isCurrentLevel,
                            ),
                            const SizedBox(width: 16),
                            _buildBlindInfo(
                              'BB',
                              _formatChips(level.bigBlind),
                              isCurrentLevel,
                            ),
                            if (level.ante > 0) ...[
                              const SizedBox(width: 16),
                              _buildBlindInfo(
                                'Ante',
                                _formatChips(level.ante),
                                isCurrentLevel,
                              ),
                            ],
                          ],
                        ),
                      ),
                      if (isCurrentLevel)
                        const Icon(
                          Icons.arrow_forward_ios,
                          size: 16,
                          color: Colors.green,
                        ),
                    ],
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildBlindInfo(String label, String value, bool isCurrent) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(fontSize: 10, color: Colors.grey[600])),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: isCurrent ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ],
    );
  }

  String _formatChips(int chips) {
    if (chips >= 1000000) {
      return '${(chips / 1000000).toStringAsFixed(1)}M';
    } else if (chips >= 1000) {
      return '${(chips / 1000).toStringAsFixed(0)}K';
    }
    return chips.toString();
  }
}
