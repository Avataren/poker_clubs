import 'package:flutter/material.dart';
import '../models/tournament.dart';
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

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _loadDetail();

    // Listen for tournament events
    widget.websocketService.onTournamentStarted = _onTournamentStarted;
    widget.websocketService.onTournamentBlindLevelIncreased =
        _onBlindLevelIncreased;
    widget.websocketService.onTournamentPlayerEliminated = _onPlayerEliminated;
    widget.websocketService.onTournamentFinished = _onTournamentFinished;
  }

  @override
  void dispose() {
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
      children: [_buildInfoTab(), _buildPlayersTab(), _buildBlindsTab()],
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
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Show Fill with Bots button if tournament is in registration and not full
            if (tournament.status == 'registration' &&
                _detail!.registrations.length < tournament.maxPlayers) ...[
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _isProcessing ? null : _fillWithBots,
                  icon: const Icon(Icons.smart_toy),
                  label: Text(
                    'Fill ${tournament.maxPlayers - _detail!.registrations.length} Remaining Seats with Bots',
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
              const SizedBox(height: 8),
            ],
            Row(
              children: [
                Expanded(
                  child: _buildActionButton(
                    tournament,
                    isRegistered,
                    canRegister,
                  ),
                ),
              ],
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
    if (tournament.status == 'registration') {
      if (isRegistered) {
        return ElevatedButton.icon(
          onPressed: _isProcessing ? null : _unregister,
          icon: const Icon(Icons.exit_to_app),
          label: const Text('Unregister'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.red,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
        );
      } else if (canRegister) {
        return ElevatedButton.icon(
          onPressed: _isProcessing ? null : _register,
          icon: const Icon(Icons.how_to_reg),
          label: Text('Register - \$${_formatChips(tournament.buyIn)}'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.green,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
        );
      } else {
        return ElevatedButton.icon(
          onPressed: null,
          icon: const Icon(Icons.block),
          label: const Text('Registration Full'),
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
        );
      }
    } else if (tournament.status == 'running') {
      return ElevatedButton.icon(
        onPressed: null,
        icon: const Icon(Icons.play_arrow),
        label: const Text('In Progress'),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.orange,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
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
