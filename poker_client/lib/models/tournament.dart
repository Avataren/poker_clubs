class Tournament {
  final String id;
  final String clubId;
  final String name;
  final String formatId; // 'sng' or 'mtt'
  final String variantId; // 'holdem', etc.
  final int buyIn;
  final int maxPlayers;
  final int minPlayers;
  final int startingStack;
  final int prizePool;
  final int registeredPlayers;
  final int remainingPlayers;
  final int currentBlindLevel;
  final int levelDurationSecs;
  final String? levelStartTime;
  final String
  status; // 'registering', 'seating', 'running', 'paused', 'finished', 'cancelled'
  final int preSeatSecs;
  final DateTime createdAt;
  final DateTime? scheduledStart;
  final DateTime? actualStart;
  final DateTime? finishedAt;
  final String? cancelReason;

  Tournament({
    required this.id,
    required this.clubId,
    required this.name,
    required this.formatId,
    required this.variantId,
    required this.buyIn,
    required this.maxPlayers,
    required this.minPlayers,
    required this.startingStack,
    required this.prizePool,
    required this.registeredPlayers,
    required this.remainingPlayers,
    required this.currentBlindLevel,
    required this.levelDurationSecs,
    this.levelStartTime,
    required this.status,
    required this.preSeatSecs,
    required this.createdAt,
    this.scheduledStart,
    this.actualStart,
    this.finishedAt,
    this.cancelReason,
  });

  // Convenience getter for level duration in minutes
  int get levelDurationMins => (levelDurationSecs / 60).round();

  // Convenience getter for tournament type (alias for formatId)
  String get tournamentType => formatId;

  factory Tournament.fromJson(Map<String, dynamic> json) {
    return Tournament(
      id: json['id'] as String,
      clubId: json['club_id'] as String,
      name: json['name'] as String,
      formatId: json['format_id'] as String,
      variantId: json['variant_id'] as String,
      buyIn: json['buy_in'] as int,
      maxPlayers: json['max_players'] as int,
      minPlayers: json['min_players'] as int,
      startingStack: json['starting_stack'] as int,
      prizePool: json['prize_pool'] as int,
      registeredPlayers: json['registered_players'] as int,
      remainingPlayers: json['remaining_players'] as int,
      currentBlindLevel: json['current_blind_level'] as int,
      levelDurationSecs: json['level_duration_secs'] as int,
      levelStartTime: json['level_start_time'] as String?,
      status: json['status'] as String,
      preSeatSecs: json['pre_seat_secs'] as int,
      createdAt: DateTime.parse(json['created_at'] as String),
      scheduledStart: json['scheduled_start'] != null
          ? DateTime.parse(json['scheduled_start'] as String)
          : null,
      actualStart: json['actual_start'] != null
          ? DateTime.parse(json['actual_start'] as String)
          : null,
      finishedAt: json['finished_at'] != null
          ? DateTime.parse(json['finished_at'] as String)
          : null,
      cancelReason: json['cancel_reason'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'club_id': clubId,
      'name': name,
      'format_id': formatId,
      'variant_id': variantId,
      'buy_in': buyIn,
      'max_players': maxPlayers,
      'min_players': minPlayers,
      'starting_stack': startingStack,
      'prize_pool': prizePool,
      'registered_players': registeredPlayers,
      'remaining_players': remainingPlayers,
      'current_blind_level': currentBlindLevel,
      'level_duration_secs': levelDurationSecs,
      'level_start_time': levelStartTime,
      'status': status,
      'pre_seat_secs': preSeatSecs,
      'created_at': createdAt.toIso8601String(),
      'scheduled_start': scheduledStart?.toIso8601String(),
      'actual_start': actualStart?.toIso8601String(),
      'finished_at': finishedAt?.toIso8601String(),
      'cancel_reason': cancelReason,
    };
  }
}

class TournamentBlindLevel {
  final String tournamentId;
  final int levelNumber;
  final int smallBlind;
  final int bigBlind;
  final int ante;

  TournamentBlindLevel({
    required this.tournamentId,
    required this.levelNumber,
    required this.smallBlind,
    required this.bigBlind,
    required this.ante,
  });

  factory TournamentBlindLevel.fromJson(Map<String, dynamic> json) {
    return TournamentBlindLevel(
      tournamentId: json['tournament_id'] as String,
      levelNumber: json['level_number'] as int,
      smallBlind: json['small_blind'] as int,
      bigBlind: json['big_blind'] as int,
      ante: json['ante'] as int,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'tournament_id': tournamentId,
      'level_number': levelNumber,
      'small_blind': smallBlind,
      'big_blind': bigBlind,
      'ante': ante,
    };
  }
}

class TournamentRegistration {
  final String userId;
  final String username;
  final DateTime registeredAt;
  final int? finishPosition;
  final int prizeAmount;

  TournamentRegistration({
    required this.userId,
    required this.username,
    required this.registeredAt,
    this.finishPosition,
    required this.prizeAmount,
  });

  factory TournamentRegistration.fromJson(Map<String, dynamic> json) {
    return TournamentRegistration(
      userId: json['user_id'] as String,
      username: json['username'] as String,
      registeredAt: DateTime.parse(json['registered_at'] as String),
      finishPosition: json['finish_position'] as int?,
      prizeAmount: json['prize_amount'] as int,
    );
  }
}

class TournamentWithStats {
  final Tournament tournament;
  final int registeredCount;
  final bool isRegistered;

  TournamentWithStats({
    required this.tournament,
    required this.registeredCount,
    required this.isRegistered,
  });

  factory TournamentWithStats.fromJson(Map<String, dynamic> json) {
    return TournamentWithStats(
      tournament: Tournament.fromJson(
        json['tournament'] as Map<String, dynamic>,
      ),
      registeredCount: json['registered_count'] as int,
      isRegistered: json['is_registered'] as bool,
    );
  }
}

class TournamentDetail {
  final Tournament tournament;
  final List<TournamentBlindLevel> blindLevels;
  final List<TournamentRegistration> registrations;
  final bool isRegistered;
  final bool canRegister;

  TournamentDetail({
    required this.tournament,
    required this.blindLevels,
    required this.registrations,
    required this.isRegistered,
    required this.canRegister,
  });

  factory TournamentDetail.fromJson(Map<String, dynamic> json) {
    return TournamentDetail(
      tournament: Tournament.fromJson(
        json['tournament'] as Map<String, dynamic>,
      ),
      blindLevels: (json['blind_levels'] as List)
          .map((e) => TournamentBlindLevel.fromJson(e as Map<String, dynamic>))
          .toList(),
      registrations: (json['registrations'] as List)
          .map(
            (e) => TournamentRegistration.fromJson(e as Map<String, dynamic>),
          )
          .toList(),
      isRegistered: json['is_registered'] as bool,
      canRegister: json['can_register'] as bool,
    );
  }
}

class TournamentResult {
  final String userId;
  final String username;
  final int finishPosition;
  final int prizeAmount;
  final DateTime? eliminatedAt;

  TournamentResult({
    required this.userId,
    required this.username,
    required this.finishPosition,
    required this.prizeAmount,
    this.eliminatedAt,
  });

  factory TournamentResult.fromJson(Map<String, dynamic> json) {
    return TournamentResult(
      userId: json['user_id'] as String,
      username: json['username'] as String,
      finishPosition: json['finish_position'] as int,
      prizeAmount: json['prize_amount'] as int,
      eliminatedAt: json['eliminated_at'] != null
          ? DateTime.parse(json['eliminated_at'] as String)
          : null,
    );
  }
}

class TournamentTableInfo {
  final String tableId;
  final int tableNumber;
  final String tableName;
  final int playerCount;

  TournamentTableInfo({
    required this.tableId,
    required this.tableNumber,
    required this.tableName,
    required this.playerCount,
  });

  factory TournamentTableInfo.fromJson(Map<String, dynamic> json) {
    return TournamentTableInfo(
      tableId: json['table_id'] as String,
      tableNumber: json['table_number'] as int,
      tableName: json['table_name'] as String,
      playerCount: json['player_count'] as int,
    );
  }
}

class TournamentWinner {
  final String username;
  final int position;
  final int prize;

  TournamentWinner({
    required this.username,
    required this.position,
    required this.prize,
  });

  factory TournamentWinner.fromJson(Map<String, dynamic> json) {
    return TournamentWinner(
      username: json['username'] as String,
      position: json['position'] as int,
      prize: json['prize'] as int,
    );
  }
}
