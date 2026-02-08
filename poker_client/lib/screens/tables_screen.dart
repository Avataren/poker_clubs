import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/club.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import 'game_screen.dart';
import 'tournaments_screen.dart';

class TablesScreen extends StatefulWidget {
  final Club club;

  const TablesScreen({super.key, required this.club});

  @override
  State<TablesScreen> createState() => _TablesScreenState();
}

class _TablesScreenState extends State<TablesScreen>
    with SingleTickerProviderStateMixin {
  List<PokerTable> _tables = [];
  List<VariantInfo> _variants = [];
  List<FormatInfo> _formats = [];
  bool _isLoading = true;
  final _tableNameController = TextEditingController();
  final _smallBlindController = TextEditingController(text: '50');
  final _bigBlindController = TextEditingController(text: '100');
  String _selectedVariantId = 'holdem';
  String _selectedFormatId = 'cash';
  WebSocketService? _wsService;
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _loadData();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();

    // Set up WebSocketService subscription if not already done
    if (_wsService == null) {
      _wsService = context.read<WebSocketService>();
      _wsService!.onClubUpdate = () {
        print('Club broadcast received - refreshing tables list');
        _loadTables();
      };
      _wsService!.viewingClub(widget.club.id);
    }
  }

  Future<void> _loadData() async {
    try {
      final apiService = context.read<ApiService>();
      final results = await Future.wait([
        apiService.getClubTables(widget.club.id),
        apiService.getVariants(),
        apiService.getFormats(),
      ]);
      setState(() {
        _tables = results[0] as List<PokerTable>;
        _variants = results[1] as List<VariantInfo>;
        _formats = results[2] as List<FormatInfo>;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _loadTables() async {
    try {
      final tables = await context.read<ApiService>().getClubTables(
        widget.club.id,
      );
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
      print(
        'Creating table: ${_tableNameController.text}, variant: $_selectedVariantId, format: $_selectedFormatId',
      );
      final table = await context.read<ApiService>().createTable(
        widget.club.id,
        _tableNameController.text,
        int.parse(_smallBlindController.text),
        int.parse(_bigBlindController.text),
        variantId: _selectedVariantId,
        formatId: _selectedFormatId,
      );
      print('Table created successfully: ${table.id}');
      _tableNameController.clear();
      await _loadTables();
      if (mounted) {
        debugPrint('Table "${table.name}" created');
      }
    } catch (e, stackTrace) {
      print('Error creating table: $e');
      print('Stack trace: $stackTrace');
      if (mounted) {
        debugPrint('Create table error: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.club.name),
        backgroundColor: Colors.grey[850],
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.table_chart), text: 'Tables'),
            Tab(icon: Icon(Icons.emoji_events), text: 'Tournaments'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildTablesTab(),
          TournamentsScreen(
            apiService: context.read<ApiService>(),
            websocketService: context.read<WebSocketService>(),
            clubId: widget.club.id,
          ),
        ],
      ),
    );
  }

  Widget _buildTablesTab() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final isCompact = constraints.maxWidth < 700;
        return Column(
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
                  isCompact
                      ? Column(
                          children: [
                            TextField(
                              controller: _smallBlindController,
                              decoration: const InputDecoration(
                                labelText: 'Small Blind',
                                border: OutlineInputBorder(),
                              ),
                              keyboardType: TextInputType.number,
                            ),
                            const SizedBox(height: 8),
                            TextField(
                              controller: _bigBlindController,
                              decoration: const InputDecoration(
                                labelText: 'Big Blind',
                                border: OutlineInputBorder(),
                              ),
                              keyboardType: TextInputType.number,
                            ),
                          ],
                        )
                      : Row(
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
                  // Variant and Format selection
                  isCompact
                      ? Column(
                          children: [
                            DropdownButtonFormField<String>(
                              initialValue: _selectedVariantId,
                              decoration: const InputDecoration(
                                labelText: 'Game Variant',
                                border: OutlineInputBorder(),
                              ),
                              items: _variants.isEmpty
                                  ? [
                                      const DropdownMenuItem(
                                        value: 'holdem',
                                        child: Text('Texas Hold\'em'),
                                      ),
                                      const DropdownMenuItem(
                                        value: 'plo',
                                        child: Text('Pot Limit Omaha'),
                                      ),
                                    ]
                                  : _variants
                                        .map(
                                          (v) => DropdownMenuItem(
                                            value: v.id,
                                            child: Text(v.name),
                                          ),
                                        )
                                        .toList(),
                              onChanged: (value) {
                                if (value != null) {
                                  setState(() => _selectedVariantId = value);
                                }
                              },
                            ),
                            const SizedBox(height: 8),
                            DropdownButtonFormField<String>(
                              initialValue: _selectedFormatId,
                              decoration: const InputDecoration(
                                labelText: 'Game Format',
                                border: OutlineInputBorder(),
                              ),
                              items: _formats.isEmpty
                                  ? [
                                      const DropdownMenuItem(
                                        value: 'cash',
                                        child: Text('Cash Game'),
                                      ),
                                    ]
                                  : _formats
                                        .map(
                                          (f) => DropdownMenuItem(
                                            value: f.id,
                                            child: Text(f.name),
                                          ),
                                        )
                                        .toList(),
                              onChanged: (value) {
                                if (value != null) {
                                  setState(() => _selectedFormatId = value);
                                }
                              },
                            ),
                          ],
                        )
                      : Row(
                          children: [
                            Expanded(
                              child: DropdownButtonFormField<String>(
                                initialValue: _selectedVariantId,
                                decoration: const InputDecoration(
                                  labelText: 'Game Variant',
                                  border: OutlineInputBorder(),
                                ),
                                items: _variants.isEmpty
                                    ? [
                                        const DropdownMenuItem(
                                          value: 'holdem',
                                          child: Text('Texas Hold\'em'),
                                        ),
                                        const DropdownMenuItem(
                                          value: 'plo',
                                          child: Text('Pot Limit Omaha'),
                                        ),
                                      ]
                                    : _variants
                                          .map(
                                            (v) => DropdownMenuItem(
                                              value: v.id,
                                              child: Text(v.name),
                                            ),
                                          )
                                          .toList(),
                                onChanged: (value) {
                                  if (value != null) {
                                    setState(() => _selectedVariantId = value);
                                  }
                                },
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: DropdownButtonFormField<String>(
                                initialValue: _selectedFormatId,
                                decoration: const InputDecoration(
                                  labelText: 'Game Format',
                                  border: OutlineInputBorder(),
                                ),
                                items: _formats.isEmpty
                                    ? [
                                        const DropdownMenuItem(
                                          value: 'cash',
                                          child: Text('Cash Game'),
                                        ),
                                      ]
                                    : _formats
                                          .map(
                                            (f) => DropdownMenuItem(
                                              value: f.id,
                                              child: Text(f.name),
                                            ),
                                          )
                                          .toList(),
                                onChanged: (value) {
                                  if (value != null) {
                                    setState(() => _selectedFormatId = value);
                                  }
                                },
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
        );
      },
    );
  }

  @override
  void dispose() {
    _tabController.dispose();
    _tableNameController.dispose();
    _smallBlindController.dispose();
    _bigBlindController.dispose();

    // Unsubscribe from club broadcasts using stored reference
    if (_wsService != null) {
      _wsService!.onClubUpdate = null;
      _wsService!.leavingView();
    }

    super.dispose();
  }
}
