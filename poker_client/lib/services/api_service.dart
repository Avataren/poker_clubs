import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config.dart';
import '../models/club.dart';
import '../models/tournament.dart';

class ApiService {
  static const String baseUrl = AppConfig.httpBaseUrl;
  String? _token;
  String? _userId;

  String? get token => _token;
  String? get userId => _userId;
  bool get isAuthenticated => _token != null;

  void logout() {
    _token = null;
    _userId = null;
  }

  Future<Map<String, dynamic>> register(
    String username,
    String email,
    String password,
  ) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'username': username,
        'email': email,
        'password': password,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      _token = data['token'];
      _userId = data['user']['id'];
      return data;
    } else {
      throw Exception('Registration failed: ${response.body}');
    }
  }

  Future<Map<String, dynamic>> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      _token = data['token'];
      _userId = data['user']['id'];
      return data;
    } else {
      throw Exception('Login failed: ${response.body}');
    }
  }

  Future<List<Club>> getMyClubs() async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.get(
      Uri.parse('$baseUrl/api/clubs/my'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((json) => Club.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load clubs');
    }
  }

  Future<List<Club>> getAllClubs() async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.get(
      Uri.parse('$baseUrl/api/clubs/all'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((json) => Club.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load clubs');
    }
  }

  Future<Club> createClub(String name) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/clubs'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $_token',
      },
      body: jsonEncode({'name': name}),
    );

    if (response.statusCode == 200) {
      return Club.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to create club');
    }
  }

  Future<Club> joinClub(String clubId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/clubs/join'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $_token',
      },
      body: jsonEncode({'club_id': clubId}),
    );

    if (response.statusCode == 200) {
      return Club.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to join club');
    }
  }

  Future<List<PokerTable>> getClubTables(String clubId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.get(
      Uri.parse('$baseUrl/api/tables/club/$clubId'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final List<dynamic> tables = data['tables'];
      return tables.map((json) => PokerTable.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load tables');
    }
  }

  Future<PokerTable> createTable(
    String clubId,
    String name,
    int smallBlind,
    int bigBlind, {
    String? variantId,
    String? formatId,
  }) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final body = {
      'club_id': clubId,
      'name': name,
      'small_blind': smallBlind,
      'big_blind': bigBlind,
    };

    if (variantId != null) {
      body['variant_id'] = variantId;
    }
    if (formatId != null) {
      body['format_id'] = formatId;
    }

    final response = await http.post(
      Uri.parse('$baseUrl/api/tables'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $_token',
      },
      body: jsonEncode(body),
    );

    if (response.statusCode == 200) {
      return PokerTable.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to create table');
    }
  }

  /// Get all available poker variants
  Future<List<VariantInfo>> getVariants() async {
    final response = await http.get(Uri.parse('$baseUrl/api/tables/variants'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final List<dynamic> variants = data['variants'];
      return variants.map((json) => VariantInfo.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load variants');
    }
  }

  /// Get all available game formats
  Future<List<FormatInfo>> getFormats() async {
    final response = await http.get(Uri.parse('$baseUrl/api/tables/formats'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final List<dynamic> formats = data['formats'];
      return formats.map((json) => FormatInfo.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load formats');
    }
  }

  // ==================== Tournament Methods ====================

  /// Create a Sit-and-Go tournament
  Future<Tournament> createSng({
    required String clubId,
    required String name,
    required int buyIn,
    required int maxPlayers,
    int? minPlayers,
    required int startingStack,
    required int levelDurationMins,
    String? variantId,
  }) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final body = {
      'club_id': clubId,
      'name': name,
      'buy_in': buyIn,
      'max_players': maxPlayers,
      'min_players': minPlayers,
      'starting_stack': startingStack,
      'level_duration_mins': levelDurationMins,
      'variant_id': variantId,
    };

    final response = await http.post(
      Uri.parse('$baseUrl/api/tournaments/sng'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $_token',
      },
      body: jsonEncode(body),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return Tournament.fromJson(data['tournament']);
    } else {
      throw Exception('Failed to create SNG: ${response.body}');
    }
  }

  /// Create a Multi-Table Tournament
  Future<Tournament> createMtt({
    required String clubId,
    required String name,
    required int buyIn,
    required int maxPlayers,
    int? minPlayers,
    required int startingStack,
    required int levelDurationMins,
    DateTime? scheduledStart,
    String? variantId,
  }) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final body = {
      'club_id': clubId,
      'name': name,
      'buy_in': buyIn,
      'max_players': maxPlayers,
      'min_players': minPlayers,
      'starting_stack': startingStack,
      'level_duration_mins': levelDurationMins,
      'scheduled_start': scheduledStart?.toIso8601String(),
      'variant_id': variantId,
    };

    final response = await http.post(
      Uri.parse('$baseUrl/api/tournaments/mtt'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $_token',
      },
      body: jsonEncode(body),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return Tournament.fromJson(data['tournament']);
    } else {
      throw Exception('Failed to create MTT: ${response.body}');
    }
  }

  /// Get all tournaments (optionally filtered by club)
  Future<List<TournamentWithStats>> getTournaments({String? clubId}) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    if (clubId == null) {
      throw Exception('Club ID is required');
    }

    final uri = Uri.parse('$baseUrl/api/tournaments/club/$clubId');

    final response = await http.get(
      uri,
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final List<dynamic> tournaments = data['tournaments'];
      return tournaments
          .map((json) => TournamentWithStats.fromJson(json))
          .toList();
    } else {
      throw Exception('Failed to load tournaments');
    }
  }

  /// Get tournament details
  Future<TournamentDetail> getTournamentDetail(String tournamentId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.get(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      return TournamentDetail.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load tournament details');
    }
  }

  /// Get tournament tables
  Future<List<TournamentTableInfo>> getTournamentTables(
    String tournamentId,
  ) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.get(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/tables'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final List<dynamic> tables = data['tables'];
      return tables.map((json) => TournamentTableInfo.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load tournament tables');
    }
  }

  /// Register for a tournament
  Future<void> registerForTournament(String tournamentId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/register'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to register: ${response.body}');
    }
  }

  /// Unregister from a tournament
  Future<void> unregisterFromTournament(String tournamentId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/unregister'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to unregister: ${response.body}');
    }
  }

  /// Start a tournament (admin only)
  Future<void> startTournament(String tournamentId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/start'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to start tournament: ${response.body}');
    }
  }

  /// Cancel a tournament (admin only)
  Future<void> cancelTournament(String tournamentId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.delete(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/cancel'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to cancel tournament: ${response.body}');
    }
  }

  /// Get tournament results
  Future<List<TournamentResult>> getTournamentResults(
    String tournamentId,
  ) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.get(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/results'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final List<dynamic> results = data['results'];
      return results.map((json) => TournamentResult.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load tournament results');
    }
  }

  /// Fill tournament with bots (admin only)
  Future<TournamentDetail> fillTournamentWithBots(String tournamentId) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/tournaments/$tournamentId/fill-bots'),
      headers: {'Authorization': 'Bearer $_token'},
    );

    if (response.statusCode == 200) {
      return TournamentDetail.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to fill with bots: ${response.body}');
    }
  }
}

/// Poker variant information
class VariantInfo {
  final String id;
  final String name;

  VariantInfo({required this.id, required this.name});

  factory VariantInfo.fromJson(Map<String, dynamic> json) {
    return VariantInfo(id: json['id'], name: json['name']);
  }
}

/// Game format information
class FormatInfo {
  final String id;
  final String name;

  FormatInfo({required this.id, required this.name});

  factory FormatInfo.fromJson(Map<String, dynamic> json) {
    return FormatInfo(id: json['id'], name: json['name']);
  }
}
