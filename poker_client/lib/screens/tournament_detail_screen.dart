import 'package:flutter/material.dart';
import '../models/tournament.dart';
import '../models/club.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import '../widgets/blind_structure_widget.dart';
import '../widgets/tournament_status_widget.dart';

class TournamentDetailScreen extends StatefulWidget {
  final ApiService apiService;
  final WebSocketService websocketService;
  final String tournamentId;

  const TournamentDetailScreen({
    super.key,
    required this.apiService,
    required this.websocketService,
    required this.tournamentId,
  });

  @override
  State<TournamentDetailScreen> createState() => _TournamentDetailScreenState();
}

class _TournamentDetailScreenState extends State<TournamentDetailScreen>
    with SingleTickerProviderStateMixin {
  TournamentDetail? _detail;
  bool _isLoading = true;
  String? _error;
  bool _isProcessing = false;
  int? _currentBlindLevel;
  late TabController _tabController;
  List<TournamentTableInfo> _tables = [];
  bool _loadingTables = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
    _loadDetail();

    // Listen for tournament events
    widget.websocketService.onTournamentStarted = _onTournamentStarted;
    widget.websocketService.onTournamentBlindLevelIncreased =
        _onBlindLevelIncreased;
    widget.websocketService.onTournamentPlayerEliminated = _onPlayerEliminated;
    widget.websocketService.onTournamentFinished = _onTournamentFinished;
    widget.websocketService.onTournamentCancelled = _onTournamentCancelled;

    // Load tables when switching to tables tab
    _tabController.addListener(() {
      if (_tabController.index == 2 && _tables.isEmpty) {
        _loadTables();
      }
    });
  }

  @override
  void dispose() {
    widget.websocketService.onTournamentStarted = null;
    widget.websocketService.onTournamentBlindLevelIncreased = null;
    widget.websocketService.onTournamentPlayerEliminated = null;
    widget.websocketService.onTournamentFinished = null;
    widget.websocketService.onTournamentCancelled = null;
    _tabController.dispose();
    super.dispose();
  }

  void _onTournamentStarted(
    String tournamentId,
    String tournamentName,
    String? tableId,
  ) {
    if (tournamentId == widget.tournamentId) {
      _showSnackBar('Tournament has started!');
      _loadDetail();
      _loadTables(); // Refresh tables when tournament starts
    }
  }

  void _onBlindLevelIncreased(
    String tournamentId,
    int level,
    int smallBlind,
    int bigBlind,
    int ante,
  ) {
    if (tournamentId == widget.tournamentId) {
      setState(() {
        _currentBlindLevel = level;
      });
      _showSnackBar('Blinds increased to $smallBlind/$bigBlind');
    }
  }

  void _onPlayerEliminated(
    String tournamentId,
    String username,
    int position,
    int prize,
  ) {
    if (tournamentId == widget.tournamentId) {
      _showSnackBar('$username eliminated in position $position');
      _loadDetail();
    }
  }

  void _onTournamentFinished(
    String tournamentId,
    String tournamentName,
    List<TournamentWinner> winners,
  ) {
    if (tournamentId == widget.tournamentId) {
      _showSnackBar('Tournament has finished!');
      _loadDetail();
    }
  }

  void _onTournamentCancelled(
    String tournamentId,
    String tournamentName,
    String reason,
  ) {
    if (tournamentId == widget.tournamentId) {
      _showSnackBar('Tournament cancelled: $reason');
      _loadDetail();
    }
  }

  void _showSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(message)));
    }
  }

  Future<void> _loadDetail() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final detail = await widget.apiService.getTournamentDetail(
        widget.tournamentId,
      );

      setState(() {
        _detail = detail;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _loadTables() async {
    if (_loadingTables) return;

    setState(() => _loadingTables = true);

    try {
      final tables = await widget.apiService.getTournamentTables(
        widget.tournamentId,
      );

      setState(() {
        _tables = tables;
        _loadingTables = false;
      });
    } catch (e) {
      setState(() => _loadingTables = false);
      _showSnackBar('Failed to load tables: $e');
    }
  }

  Future<void> _register() async {
    setState(() => _isProcessing = true);

    try {
      await widget.apiService.registerForTournament(widget.tournamentId);
      _showSnackBar('Successfully registered!');
      await _loadDetail();
    } catch (e) {
      _showSnackBar('Error: $e');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _unregister() async {
    setState(() => _isProcessing = true);

    try {
      await widget.apiService.unregisterFromTournament(widget.tournamentId);
      _showSnackBar('Successfully unregistered');
      await _loadDetail();
    } catch (e) {
      _showSnackBar('Error: $e');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _startTournament() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Start Tournament'),
        content: const Text('Are you sure you want to start this tournament?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Start'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    setState(() => _isProcessing = true);

    try {
      await widget.apiService.startTournament(widget.tournamentId);
      _showSnackBar('Tournament started!');
      await _loadDetail();
    } catch (e) {
      _showSnackBar('Error: $e');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _cancelTournament() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Cancel Tournament'),
        content: const Text(
          'Are you sure you want to cancel this tournament? This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('No'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
            ),
            child: const Text('Yes, Cancel Tournament'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    setState(() => _isProcessing = true);

    try {
      await widget.apiService.cancelTournament(widget.tournamentId);
      _showSnackBar('Tournament cancelled');
      await _loadDetail();
    } catch (e) {
      _showSnackBar('Error: $e');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _fillWithBots() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Fill with Bots'),
        content: const Text(
          'This will fill all remaining seats with bot players. Continue?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Fill'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    setState(() => _isProcessing = true);

    try {
      final updatedDetail = await widget.apiService.fillTournamentWithBots(
        widget.tournamentId,
      );
      setState(() {
        _detail = updatedDetail;
      });
      _showSnackBar('Tournament filled with bots!');
    } catch (e) {
      _showSnackBar('Error: $e');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_detail?.tournament.name ?? 'Tournament'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Info', icon: Icon(Icons.info)),
            Tab(text: 'Players', icon: Icon(Icons.people)),
            Tab(text: 'Tables', icon: Icon(Icons.table_chart)),
            Tab(text: 'Blinds', icon: Icon(Icons.monetization_on)),
          ],
        ),
      ),
      body: _buildBody(),
      bottomNavigationBar: _buildBottomBar(),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Error: $_error'),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _loadDetail, child: const Text('Retry')),
          ],
        ),
      );
    }

    if (_detail == null) {
      return const Center(child: Text('No data'));
    }

    return TabBarView(
      controller: _tabController,
      children: [
        _buildInfoTab(),
        _buildPlayersTab(),
        _buildTablesTab(),
        _buildBlindsTab(),
      ],
    );
  }

  Widget _buildInfoTab() {
    return SingleChildScrollView(
      child: TournamentStatusWidget(
        tournament: _detail!.tournament,
        registeredCount: _detail!.registrations.length,
      ),
    );
  }

  Widget _buildPlayersTab() {
    final registrations = _detail!.registrations;

    if (registrations.isEmpty) {
      return const Center(child: Text('No players registered yet'));
    }

    // Sort by finish position (if finished) or registration time
    final sortedPlayers = List<TournamentRegistration>.from(registrations);
    sortedPlayers.sort((a, b) {
      if (a.finishPosition != null && b.finishPosition != null) {
        return a.finishPosition!.compareTo(b.finishPosition!);
      } else if (a.finishPosition != null) {
        return -1;
      } else if (b.finishPosition != null) {
        return 1;
      }
      return a.registeredAt.compareTo(b.registeredAt);
    });

    return ListView.separated(
      itemCount: sortedPlayers.length,
      separatorBuilder: (context, index) => const Divider(height: 1),
      itemBuilder: (context, index) {
        final player = sortedPlayers[index];
        final isFinished = player.finishPosition != null;

        return ListTile(
          leading: CircleAvatar(
            backgroundColor: isFinished ? Colors.grey : Colors.blue,
            child: Text(
              isFinished ? '#${player.finishPosition}' : '${index + 1}',
              style: const TextStyle(color: Colors.white),
            ),
          ),
          title: Text(
            player.username,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              decoration: isFinished ? TextDecoration.lineThrough : null,
            ),
          ),
          subtitle: Text(
            isFinished
                ? 'Eliminated - Position ${player.finishPosition}'
                : 'Registered ${_formatDateTime(player.registeredAt)}',
          ),
          trailing: player.prizeAmount > 0
              ? Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.green,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '\$${_formatChips(player.prizeAmount)}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                )
              : null,
        );
      },
    );
  }

  Widget _buildBlindsTab() {
    return SingleChildScrollView(
      child: BlindStructureWidget(
        blindLevels: _detail!.blindLevels,
        currentLevel: _currentBlindLevel,
      ),
    );
  }

  Widget _buildTablesTab() {
    if (_loadingTables) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_tables.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.table_chart, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            const Text('No tables created yet'),
            const SizedBox(height: 8),
            const Text(
              'Tables will be created when the tournament starts',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _loadTables,
              icon: const Icon(Icons.refresh),
              label: const Text('Refresh'),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadTables,
      child: ListView.builder(
        itemCount: _tables.length,
        itemBuilder: (context, index) {
          final table = _tables[index];
          return Card(
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: ListTile(
              leading: CircleAvatar(
                backgroundColor: Colors.green,
                child: Text(
                  '${table.tableNumber}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              title: Text(
                table.tableName,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              subtitle: Text('${table.playerCount} players'),
              trailing: ElevatedButton.icon(
                onPressed: () {
                  // Create a PokerTable object for tournament table
                  // Tournament tables are in-memory and use default values
                  final pokerTable = PokerTable(
                    id: table.tableId,
                    clubId: _detail!.tournament.clubId,
                    name: table.tableName.isEmpty
                        ? 'Table ${table.tableNumber}'
                        : table.tableName,
                    smallBlind: 50, // Default values for tournament
                    bigBlind: 100,
                    minBuyin: 0,
                    maxBuyin: 0,
                    maxPlayers: 9,
                  );

                  Navigator.pushNamed(
                    context,
                    '/table',
                    arguments: {'table': pokerTable},
                  );
                },
                icon: const Icon(Icons.visibility, size: 18),
                label: const Text('Watch'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget? _buildBottomBar() {
    if (_detail == null || _isProcessing) return null;

    final tournament = _detail!.tournament;
    final isRegistered = _detail!.isRegistered;
    final canRegister = _detail!.canRegister;

    if (tournament.status == 'finished' || tournament.status == 'cancelled') {
      return null;
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: SafeArea(
        child: Row(
          spacing: 8,
          children: [
            // Show Fill with Bots button if tournament is in registration and not full
            if (tournament.status == 'registering' &&
                _detail!.registrations.length < tournament.maxPlayers)
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isProcessing ? null : _fillWithBots,
                  icon: const Icon(Icons.smart_toy, size: 18),
                  label: Text(
                    'Fill ${tournament.maxPlayers - _detail!.registrations.length} with Bots',
                    style: const TextStyle(fontSize: 13),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            // Show Cancel button if tournament hasn't finished
            if (tournament.status != 'finished' &&
                tournament.status != 'cancelled')
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isProcessing ? null : _cancelTournament,
                  icon: const Icon(Icons.cancel, size: 18),
                  label: const Text(
                    'Cancel',
                    style: TextStyle(fontSize: 13),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade700,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            Expanded(
              child: _buildActionButton(
                tournament,
                isRegistered,
                canRegister,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton(
    Tournament tournament,
    bool isRegistered,
    bool canRegister,
  ) {
    if (tournament.status == 'registering') {
      if (isRegistered) {
        return ElevatedButton.icon(
          onPressed: _isProcessing ? null : _unregister,
          icon: const Icon(Icons.exit_to_app, size: 18),
          label: const Text(
            'Unregister',
            style: TextStyle(fontSize: 13),
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.red,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 12),
          ),
        );
      } else if (canRegister) {
        return ElevatedButton.icon(
          onPressed: _isProcessing ? null : _register,
          icon: const Icon(Icons.how_to_reg, size: 18),
          label: Text(
            'Register - \$${_formatChips(tournament.buyIn)}',
            style: const TextStyle(fontSize: 13),
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.green,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 12),
          ),
        );
      } else {
        return ElevatedButton.icon(
          onPressed: null,
          icon: const Icon(Icons.block, size: 18),
          label: const Text(
            'Full',
            style: TextStyle(fontSize: 13),
          ),
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 12),
          ),
        );
      }
    } else if (tournament.status == 'running') {
      return ElevatedButton.icon(
        onPressed: null,
        icon: const Icon(Icons.play_arrow, size: 18),
        label: const Text(
          'In Progress',
          style: TextStyle(fontSize: 13),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.orange,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 12),
        ),
      );
    }

    return const SizedBox.shrink();
  }

  String _formatChips(int chips) {
    if (chips >= 1000000) {
      return '${(chips / 1000000).toStringAsFixed(1)}M';
    } else if (chips >= 1000) {
      return '${(chips / 1000).toStringAsFixed(1)}K';
    }
    return chips.toString();
  }

  String _formatDateTime(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);

    if (diff.inDays > 0) {
      return '${diff.inDays}d ago';
    } else if (diff.inHours > 0) {
      return '${diff.inHours}h ago';
    } else if (diff.inMinutes > 0) {
      return '${diff.inMinutes}m ago';
    } else {
      return 'Just now';
    }
  }
}
