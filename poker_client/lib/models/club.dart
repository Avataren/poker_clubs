class Club {
  final String id;
  final String name;
  final String adminId;
  final int balance;
  final bool isAdmin;

  Club({
    required this.id,
    required this.name,
    required this.adminId,
    required this.balance,
    required this.isAdmin,
  });

  factory Club.fromJson(Map<String, dynamic> json) {
    final club = json['club'] as Map<String, dynamic>;
    return Club(
      id: club['id'] as String,
      name: club['name'] as String,
      adminId: club['admin_id'] as String,
      balance: json['balance'] as int,
      isAdmin: json['is_admin'] as bool,
    );
  }

  String get balanceFormatted => '\$${(balance / 100).toStringAsFixed(2)}';
}

class PokerTable {
  final String id;
  final String clubId;
  final String name;
  final int smallBlind;
  final int bigBlind;
  final int minBuyin;
  final int maxBuyin;
  final int maxPlayers;

  PokerTable({
    required this.id,
    required this.clubId,
    required this.name,
    required this.smallBlind,
    required this.bigBlind,
    required this.minBuyin,
    required this.maxBuyin,
    required this.maxPlayers,
  });

  factory PokerTable.fromJson(Map<String, dynamic> json) {
    return PokerTable(
      id: json['id'] as String,
      clubId: json['club_id'] as String,
      name: json['name'] as String,
      smallBlind: json['small_blind'] as int,
      bigBlind: json['big_blind'] as int,
      minBuyin: json['min_buyin'] as int,
      maxBuyin: json['max_buyin'] as int,
      maxPlayers: json['max_players'] as int,
    );
  }

  String get blindsStr => '\$$smallBlind/\$$bigBlind';
}
