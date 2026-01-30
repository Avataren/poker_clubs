import 'card.dart';
import 'player.dart';

class GameState {
  final String tableId;
  final String name;
  final String phase;
  final List<PokerCard> communityCards;
  final int potTotal;
  final int currentBet;
  final int currentPlayerSeat;
  final List<Player> players;
  final int maxSeats;
  final String? lastWinnerMessage;
  final String? winningHand;

  GameState({
    required this.tableId,
    required this.name,
    required this.phase,
    required this.communityCards,
    required this.potTotal,
    required this.currentBet,
    required this.currentPlayerSeat,
    required this.players,
    required this.maxSeats,
    this.lastWinnerMessage,
    this.winningHand,
  });

  factory GameState.fromJson(Map<String, dynamic> json) {
    return GameState(
      tableId: json['table_id'] as String,
      name: json['name'] as String,
      phase: json['phase'] as String,
      communityCards: (json['community_cards'] as List)
          .map((c) => PokerCard.fromJson(c))
          .toList(),
      potTotal: json['pot_total'] as int,
      currentBet: json['current_bet'] as int,
      currentPlayerSeat: json['current_player_seat'] as int,
      players: (json['players'] as List)
          .map((p) => Player.fromJson(p))
          .toList(),
      maxSeats: json['max_seats'] as int? ?? 9,
      lastWinnerMessage: json['last_winner_message'] as String?,
      winningHand: json['winning_hand'] as String?,
    );
  }

  Player? get currentPlayer {
    // currentPlayerSeat is a seat number, not an array index
    // Find the player with that seat number
    try {
      return players.firstWhere((player) => player.seat == currentPlayerSeat);
    } catch (e) {
      return null;
    }
  }
}
