import 'card.dart';
import 'player.dart';

class GameState {
  final String tableId;
  final String name;
  final String variantId;
  final String variantName;
  final String formatId;
  final String formatName;
  final bool canCashOut;
  final bool canTopUp;
  final String phase;
  final List<PokerCard> communityCards;
  final int potTotal;
  final int currentBet;
  final int currentPlayerSeat;
  final List<Player> players;
  final int maxSeats;
  final String? lastWinnerMessage;
  final String? winningHand;
  final int? dealerSeat;
  final int? smallBlindSeat;
  final int? bigBlindSeat;

  GameState({
    required this.tableId,
    required this.name,
    required this.variantId,
    required this.variantName,
    required this.formatId,
    required this.formatName,
    required this.canCashOut,
    required this.canTopUp,
    required this.phase,
    required this.communityCards,
    required this.potTotal,
    required this.currentBet,
    required this.currentPlayerSeat,
    required this.players,
    required this.maxSeats,
    this.lastWinnerMessage,
    this.winningHand,
    this.dealerSeat,
    this.smallBlindSeat,
    this.bigBlindSeat,
  });

  factory GameState.fromJson(Map<String, dynamic> json) {
    return GameState(
      tableId: json['table_id'] as String,
      name: json['name'] as String,
      variantId: json['variant_id'] as String? ?? 'holdem',
      variantName: json['variant_name'] as String? ?? 'Texas Hold\'em',
      formatId: json['format_id'] as String? ?? 'cash',
      formatName: json['format_name'] as String? ?? 'Cash Game',
      canCashOut: json['can_cash_out'] as bool? ?? true,
      canTopUp: json['can_top_up'] as bool? ?? true,
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
      dealerSeat: json['dealer_seat'] as int?,
      smallBlindSeat: json['small_blind_seat'] as int?,
      bigBlindSeat: json['big_blind_seat'] as int?,
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

  /// Returns a human-readable game type description
  String get gameTypeDescription => '$variantName - $formatName';
}
