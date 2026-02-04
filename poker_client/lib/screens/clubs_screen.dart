import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/club.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import 'tables_screen.dart';

class ClubsScreen extends StatefulWidget {
  const ClubsScreen({super.key});

  @override
  State<ClubsScreen> createState() => _ClubsScreenState();
}

class _ClubsScreenState extends State<ClubsScreen> with WidgetsBindingObserver {
  List<Club> _clubs = [];
  bool _isLoading = true;
  final _clubNameController = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadClubs();

    // Subscribe to global broadcasts for new clubs
    final wsService = context.read<WebSocketService>();
    wsService.onGlobalUpdate = () {
      print('Global broadcast received - refreshing clubs list');
      _loadClubs();
    };
    wsService.viewingClubsList();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _clubNameController.dispose();

    // Unsubscribe from global broadcasts
    final wsService = context.read<WebSocketService>();
    wsService.onGlobalUpdate = null;
    wsService.leavingView();

    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Refresh clubs when app comes back to foreground
    if (state == AppLifecycleState.resumed) {
      _loadClubs();
    }
  }

  Future<void> _loadClubs() async {
    try {
      final clubs = await context.read<ApiService>().getAllClubs();
      setState(() {
        _clubs = clubs;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  Future<void> _joinClub(String clubId) async {
    try {
      await context.read<ApiService>().joinClub(clubId);
      _loadClubs();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Joined club successfully!')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  Future<void> _createClub() async {
    if (_clubNameController.text.isEmpty) return;

    try {
      await context.read<ApiService>().createClub(_clubNameController.text);
      _clubNameController.clear();
      _loadClubs();
    } catch (e) {
      if (mounted) {
        final errorMsg = e.toString();
        // Check if it's an authentication error
        if (errorMsg.contains('no longer exists') ||
            errorMsg.contains('Unauthorized') ||
            errorMsg.contains('401')) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Session expired. Please log in again.'),
              backgroundColor: Colors.red,
            ),
          );
          // Log out and return to login screen
          context.read<ApiService>().logout();
          if (mounted) {
            Navigator.of(context).pushReplacementNamed('/login');
          }
        } else {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('Error: $e')));
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isCompact = MediaQuery.of(context).size.width < 600;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Poker Clubs'),
        backgroundColor: Colors.green,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() => _isLoading = true);
              _loadClubs();
            },
            tooltip: 'Refresh clubs',
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              context.read<ApiService>().logout();
              context.read<WebSocketService>().disconnect();
              Navigator.of(context).pushReplacementNamed('/login');
            },
            tooltip: 'Logout',
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: isCompact
                ? Column(
                    children: [
                      TextField(
                        controller: _clubNameController,
                        decoration: const InputDecoration(
                          labelText: 'New Club Name',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: _createClub,
                          child: const Text('Create'),
                        ),
                      ),
                    ],
                  )
                : Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _clubNameController,
                          decoration: const InputDecoration(
                            labelText: 'New Club Name',
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      ElevatedButton(
                        onPressed: _createClub,
                        child: const Text('Create'),
                      ),
                    ],
                  ),
          ),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : ListView.builder(
                    itemCount: _clubs.length,
                    itemBuilder: (context, index) {
                      final club = _clubs[index];
                      return Card(
                        margin: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        child: ListTile(
                          title: Text(
                            club.name,
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          subtitle: club.balance > 0
                              ? Text('Balance: ${club.balanceFormatted}')
                              : const Text('Not a member'),
                          trailing: club.balance > 0
                              ? ElevatedButton(
                                  onPressed: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (_) =>
                                            TablesScreen(club: club),
                                      ),
                                    );
                                  },
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.green,
                                  ),
                                  child: const Text('Select'),
                                )
                              : ElevatedButton(
                                  onPressed: () => _joinClub(club.id),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.blue,
                                  ),
                                  child: const Text('Join'),
                                ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
