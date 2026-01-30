import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/club.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import 'game_screen.dart';

class TablesScreen extends StatefulWidget {
  final Club club;

  const TablesScreen({super.key, required this.club});

  @override
  State<TablesScreen> createState() => _TablesScreenState();
}

class _TablesScreenState extends State<TablesScreen> {
  List<PokerTable> _tables = [];
  bool _isLoading = true;
  final _tableNameController = TextEditingController();
  final _smallBlindController = TextEditingController(text: '50');
  final _bigBlindController = TextEditingController(text: '100');

  @override
  void initState() {
    super.initState();
    _loadTables();

    // Subscribe to club broadcasts for new tables
    final wsService = context.read<WebSocketService>();
    wsService.onClubUpdate = () {
      print('Club broadcast received - refreshing tables list');
      _loadTables();
    };
    wsService.viewingClub(widget.club.id);
  }

  Future<void> _loadTables() async {
    try {
      final tables = await context.read<ApiService>().getClubTables(widget.club.id);
      setState(() {
        _tables = tables;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _createTable() async {
    if (_tableNameController.text.isEmpty) return;

    try {
      await context.read<ApiService>().createTable(
            widget.club.id,
            _tableNameController.text,
            int.parse(_smallBlindController.text),
            int.parse(_bigBlindController.text),
          );
      _tableNameController.clear();
      _loadTables();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.club.name} - Tables'),
        backgroundColor: Colors.green,
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _tableNameController,
                  decoration: const InputDecoration(
                    labelText: 'Table Name',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _smallBlindController,
                        decoration: const InputDecoration(
                          labelText: 'Small Blind',
                          border: OutlineInputBorder(),
                        ),
                        keyboardType: TextInputType.number,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        controller: _bigBlindController,
                        decoration: const InputDecoration(
                          labelText: 'Big Blind',
                          border: OutlineInputBorder(),
                        ),
                        keyboardType: TextInputType.number,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _createTable,
                    child: const Text('Create Table'),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : ListView.builder(
                    itemCount: _tables.length,
                    itemBuilder: (context, index) {
                      final table = _tables[index];
                      return Card(
                        margin: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        child: ListTile(
                          title: Text(
                            table.name,
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          subtitle: Text('Blinds: ${table.blindsStr}'),
                          trailing: ElevatedButton(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) => GameScreen(table: table),
                                ),
                              );
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.orange,
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

  @override
  void dispose() {
    _tableNameController.dispose();
    _smallBlindController.dispose();
    _bigBlindController.dispose();

    // Unsubscribe from club broadcasts
    final wsService = context.read<WebSocketService>();
    wsService.onClubUpdate = null;
    wsService.leavingView();

    super.dispose();
  }
}
