import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/club.dart';

class ApiService {
  static const String baseUrl = 'http://127.0.0.1:3000';
  String? _token;
  String? _userId;

  String? get token => _token;
  String? get userId => _userId;
  bool get isAuthenticated => _token != null;

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
      body: jsonEncode({
        'username': username,
        'password': password,
      }),
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
    int bigBlind,
  ) async {
    if (!isAuthenticated) throw Exception('Not authenticated');

    final response = await http.post(
      Uri.parse('$baseUrl/api/tables'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $_token',
      },
      body: jsonEncode({
        'club_id': clubId,
        'name': name,
        'small_blind': smallBlind,
        'big_blind': bigBlind,
      }),
    );

    if (response.statusCode == 200) {
      return PokerTable.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to create table');
    }
  }
}
