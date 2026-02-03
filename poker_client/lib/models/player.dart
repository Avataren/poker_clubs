import 'card.dart';

class Player {
  final String userId;
  final String username;
  final int seat;
  final int stack;
  final int currentBet;
  final String state;
  final List<PokerCard>? holeCards;
  final bool isWinner;
  final String? lastAction;
  final int potWon;

  Player({
    required this.userId,
    required this.username,
    required this.seat,
    required this.stack,
    required this.currentBet,
    required this.state,
    this.holeCards,
    this.isWinner = false,
    this.lastAction,
    this.potWon = 0,
  });

  factory Player.fromJson(Map<String, dynamic> json) {
    List<PokerCard>? cards;
    if (json['hole_cards'] != null) {
      cards = (json['hole_cards'] as List)
          .map((c) => PokerCard.fromJson(c))
          .toList();
    }

    return Player(
      userId: json['user_id'] as String,
      username: json['username'] as String,
      seat: json['seat'] as int,
      stack: json['stack'] as int,
      currentBet: json['current_bet'] as int,
      state: json['state'] as String,
      holeCards: cards,
      isWinner: json['is_winner'] as bool? ?? false,
      lastAction: json['last_action'] as String?,
      potWon: json['pot_won'] as int? ?? 0,
    );
  }

  bool get isActive => state == 'Active';
  bool get isFolded => state == 'Folded';
  bool get isAllIn => state == 'AllIn';
  bool get isEliminated => state == 'Eliminated';
  bool get isBot => userId.startsWith('bot_');
}
