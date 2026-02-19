import 'package:flutter/material.dart';
import '../models/card.dart';
import '../models/hand_history.dart';

class HandHistorySheet extends StatelessWidget {
  final List<HandHistorySummary> hands;
  final VoidCallback? onLoadMore;

  const HandHistorySheet({
    super.key,
    required this.hands,
    this.onLoadMore,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SizedBox(
        height: MediaQuery.of(context).size.height * 0.65,
        child: Column(
          children: [
            const SizedBox(height: 8),
            // Drag handle
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.white24,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'Hand History',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            const Divider(color: Colors.white24, height: 1),
            Expanded(
              child: hands.isEmpty
                  ? const Center(
                      child: Text(
                        'No hands played yet',
                        style: TextStyle(color: Colors.white38),
                      ),
                    )
                  : ListView.separated(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      itemCount: hands.length,
                      separatorBuilder: (_, __) =>
                          const Divider(color: Colors.white12, height: 1),
                      itemBuilder: (context, index) {
                        return _HandRow(hand: hands[index]);
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HandRow extends StatefulWidget {
  final HandHistorySummary hand;

  const _HandRow({required this.hand});

  @override
  State<_HandRow> createState() => _HandRowState();
}

class _HandRowState extends State<_HandRow> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final hand = widget.hand;
    final winners = hand.players.where((p) => p.isWinner).toList();
    final winnerText = winners.map((w) {
      final desc = w.winningHandDesc != null ? ' (${w.winningHandDesc})' : '';
      return '${w.username} wins \$${w.potWon}$desc';
    }).join(', ');

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        InkWell(
          onTap: () => setState(() => _expanded = !_expanded),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Row(
              children: [
                Text(
                  '#${hand.handNumber}',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(width: 12),
                // Community cards mini
                ...hand.communityCards.map(
                  (c) => Padding(
                    padding: const EdgeInsets.only(right: 2),
                    child: _MiniCard(card: c),
                  ),
                ),
                if (hand.communityCards.isEmpty)
                  const Text(
                    'No board',
                    style: TextStyle(color: Colors.white24, fontSize: 11),
                  ),
                const Spacer(),
                Text(
                  'Pot \$${hand.potTotal}',
                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                ),
                const SizedBox(width: 8),
                Icon(
                  _expanded ? Icons.expand_less : Icons.expand_more,
                  color: Colors.white38,
                  size: 20,
                ),
              ],
            ),
          ),
        ),
        if (winnerText.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, bottom: 4),
            child: Text(
              winnerText,
              style: const TextStyle(color: Colors.greenAccent, fontSize: 11),
            ),
          ),
        if (_expanded) _HandDetail(hand: hand),
      ],
    );
  }
}

class _HandDetail extends StatelessWidget {
  final HandHistorySummary hand;

  const _HandDetail({required this.hand});

  @override
  Widget build(BuildContext context) {
    final streets = ['preflop', 'flop', 'turn', 'river'];
    final actionsByStreet = hand.actionsByStreet;
    final communityByStreet = hand.communityByStreet;
    final activeStreets = streets
        .where((s) =>
            actionsByStreet.containsKey(s) || communityByStreet.containsKey(s))
        .toList();

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _PlayersSection(players: hand.players),
          const SizedBox(height: 8),
          // Streets as horizontal columns
          IntrinsicHeight(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                for (int i = 0; i < activeStreets.length; i++) ...[
                  if (i > 0)
                    Container(
                      width: 1,
                      color: Colors.white12,
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                    ),
                  Expanded(
                    child: _StreetColumn(
                      street: activeStreets[i],
                      actions: actionsByStreet[activeStreets[i]] ?? [],
                      communityCards:
                          communityByStreet[activeStreets[i]] ?? [],
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PlayersSection extends StatelessWidget {
  final List<HandHistoryPlayer> players;

  const _PlayersSection({required this.players});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (final p in players)
          Padding(
            padding: const EdgeInsets.only(bottom: 2),
            child: Row(
              children: [
                SizedBox(
                  width: 80,
                  child: Text(
                    p.username,
                    style: TextStyle(
                      color: p.isWinner ? Colors.greenAccent : Colors.white60,
                      fontSize: 12,
                      fontWeight:
                          p.isWinner ? FontWeight.bold : FontWeight.normal,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '\$${p.startingStack}',
                  style:
                      const TextStyle(color: Colors.white38, fontSize: 11),
                ),
                const SizedBox(width: 8),
                if (p.holeCards != null)
                  ...p.holeCards!.map(
                    (c) => Padding(
                      padding: const EdgeInsets.only(right: 2),
                      child: _MiniCard(card: c),
                    ),
                  )
                else if (!p.folded)
                  const Text('??',
                      style: TextStyle(color: Colors.white24, fontSize: 11))
                else
                  const Text('folded',
                      style: TextStyle(
                          color: Colors.white24,
                          fontSize: 11,
                          fontStyle: FontStyle.italic)),
                if (p.isWinner && p.winningHandDesc != null) ...[
                  const SizedBox(width: 6),
                  Text(
                    p.winningHandDesc!,
                    style: const TextStyle(
                        color: Colors.amber, fontSize: 10),
                  ),
                ],
              ],
            ),
          ),
      ],
    );
  }
}

class _StreetColumn extends StatelessWidget {
  final String street;
  final List<HandHistoryAction> actions;
  final List<PokerCard> communityCards;

  const _StreetColumn({
    required this.street,
    required this.actions,
    required this.communityCards,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          street.toUpperCase(),
          style: const TextStyle(
            color: Colors.white54,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            letterSpacing: 1,
          ),
        ),
        if (communityCards.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 2, bottom: 4),
            child: Row(
              children: communityCards
                  .map((c) => Padding(
                        padding: const EdgeInsets.only(right: 2),
                        child: _MiniCard(card: c),
                      ))
                  .toList(),
            ),
          ),
        if (communityCards.isEmpty) const SizedBox(height: 4),
        for (final a in actions)
          Padding(
            padding: const EdgeInsets.only(bottom: 1),
            child: Text(
              _shortAction(a),
              style: TextStyle(
                color: _actionColor(a.actionType),
                fontSize: 10,
              ),
            ),
          ),
      ],
    );
  }

  String _shortAction(HandHistoryAction a) {
    switch (a.actionType) {
      case 'fold':
        return '${a.playerName} folds';
      case 'check':
        return '${a.playerName} ✓';
      case 'call':
        return '${a.playerName} calls \$${a.amount}';
      case 'raise':
        return '${a.playerName} ↑\$${a.amount}';
      case 'allin':
        return '${a.playerName} ALL-IN \$${a.amount}';
      case 'post_sb':
        return '${a.playerName} SB \$${a.amount}';
      case 'post_bb':
        return '${a.playerName} BB \$${a.amount}';
      default:
        return '${a.playerName} ${a.actionType}';
    }
  }

  Color _actionColor(String type) {
    switch (type) {
      case 'fold':
        return Colors.white30;
      case 'check':
        return Colors.white54;
      case 'call':
        return Colors.lightBlueAccent;
      case 'raise':
        return Colors.orangeAccent;
      case 'allin':
        return Colors.redAccent;
      default:
        return Colors.white38;
    }
  }
}

class _MiniCard extends StatelessWidget {
  final PokerCard card;

  const _MiniCard({required this.card});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(2),
      ),
      child: Text(
        '${card.rankStr}${card.suitStr}',
        style: TextStyle(
          color: card.isRed ? Colors.red[700] : Colors.black87,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}
