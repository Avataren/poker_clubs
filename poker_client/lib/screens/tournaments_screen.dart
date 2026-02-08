import 'package:flutter/material.dart';
import '../models/tournament.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import '../widgets/tournament_card_widget.dart';
import 'tournament_detail_screen.dart';

class TournamentsScreen extends StatefulWidget {
  final ApiService apiService;
  final WebSocketService websocketService;
  final String? clubId;

  const TournamentsScreen({
    super.key,
    required this.apiService,
    required this.websocketService,
    this.clubId,
  });

  @override
  State<TournamentsScreen> createState() => _TournamentsScreenState();
}

class _TournamentsScreenState extends State<TournamentsScreen> {
  List<TournamentWithStats> _allTournaments = [];
  List<TournamentWithStats> _filteredTournaments = [];
  bool _isLoading = true;
  String? _error;
  String _filter = 'all'; // 'all', 'registration', 'running', 'finished'

  @override
  void initState() {
    super.initState();
    _loadTournaments();

    // Listen for tournament events
    widget.websocketService.onTournamentStarted = _onTournamentStarted;
    widget.websocketService.onTournamentFinished = _onTournamentFinished;
  }

  void _onTournamentStarted(
    String tournamentId,
    String tournamentName,
    String? tableId,
  ) {
    _showSnackBar('Tournament "$tournamentName" has started!');
    _loadTournaments();
  }

  void _onTournamentFinished(
    String tournamentId,
    String tournamentName,
    List<TournamentWinner> winners,
  ) {
    _showSnackBar('Tournament "$tournamentName" has finished!');
    _loadTournaments();
  }

  void _showSnackBar(String message) {
    debugPrint('Tournaments notice: $message');
  }

  Future<void> _loadTournaments() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final tournaments = await widget.apiService.getTournaments(
        clubId: widget.clubId,
      );

      setState(() {
        _allTournaments = tournaments;
        _applyFilter();
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  void _applyFilter() {
    if (_filter == 'all') {
      _filteredTournaments = _allTournaments;
    } else {
      _filteredTournaments = _allTournaments
          .where((t) => t.tournament.status == _filter)
          .toList();
    }
  }

  void _setFilter(String filter) {
    setState(() {
      _filter = filter;
      _applyFilter();
    });
  }

  void _navigateToDetail(TournamentWithStats tournamentStats) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => TournamentDetailScreen(
          apiService: widget.apiService,
          websocketService: widget.websocketService,
          tournamentId: tournamentStats.tournament.id,
        ),
      ),
    ).then((_) => _loadTournaments()); // Refresh on return
  }

  void _showCreateDialog() {
    showDialog(
      context: context,
      builder: (context) => _CreateTournamentDialog(
        apiService: widget.apiService,
        clubId: widget.clubId ?? '',
        onCreated: _loadTournaments,
      ),
    );
  }

  @override
  void dispose() {
    widget.websocketService.onTournamentStarted = null;
    widget.websocketService.onTournamentFinished = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          widget.clubId != null ? 'Club Tournaments' : 'All Tournaments',
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadTournaments,
          ),
        ],
      ),
      body: Column(
        children: [
          _buildFilterBar(),
          Expanded(child: _buildBody()),
        ],
      ),
      floatingActionButton: widget.clubId != null
          ? FloatingActionButton(
              onPressed: _showCreateDialog,
              child: const Icon(Icons.add),
            )
          : null,
    );
  }

  Widget _buildFilterBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          _buildFilterChip('All', 'all'),
          _buildFilterChip('Open', 'registering'),
          _buildFilterChip('Running', 'running'),
          _buildFilterChip('Finished', 'finished'),
        ],
      ),
    );
  }

  Widget _buildFilterChip(String label, String value) {
    final isSelected = _filter == value;

    return FilterChip(
      label: Text(label),
      selected: isSelected,
      onSelected: (selected) {
        if (selected) _setFilter(value);
      },
      selectedColor: Colors.blue,
      labelStyle: TextStyle(color: isSelected ? Colors.white : Colors.black),
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
            ElevatedButton(
              onPressed: _loadTournaments,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_filteredTournaments.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.emoji_events, size: 64, color: Colors.grey[400]),
            const SizedBox(height: 16),
            Text(
              'No tournaments found',
              style: TextStyle(fontSize: 18, color: Colors.grey[600]),
            ),
            if (widget.clubId != null) ...[
              const SizedBox(height: 8),
              ElevatedButton.icon(
                onPressed: _showCreateDialog,
                icon: const Icon(Icons.add),
                label: const Text('Create Tournament'),
              ),
            ],
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadTournaments,
      child: ListView.builder(
        itemCount: _filteredTournaments.length,
        itemBuilder: (context, index) {
          final tournamentStats = _filteredTournaments[index];
          return TournamentCardWidget(
            tournamentStats: tournamentStats,
            onTap: () => _navigateToDetail(tournamentStats),
          );
        },
      ),
    );
  }
}

class _CreateTournamentDialog extends StatefulWidget {
  final ApiService apiService;
  final String clubId;
  final VoidCallback onCreated;

  const _CreateTournamentDialog({
    required this.apiService,
    required this.clubId,
    required this.onCreated,
  });

  @override
  State<_CreateTournamentDialog> createState() =>
      _CreateTournamentDialogState();
}

class _CreateTournamentDialogState extends State<_CreateTournamentDialog> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _buyInController = TextEditingController(text: '1000');
  final _maxPlayersController = TextEditingController(text: '9');
  final _startingStackController = TextEditingController(text: '10000');
  final _levelDurationController = TextEditingController(text: '10');

  String _tournamentType = 'sng';
  DateTime? _scheduledStart;
  bool _isCreating = false;

  @override
  void dispose() {
    _nameController.dispose();
    _buyInController.dispose();
    _maxPlayersController.dispose();
    _startingStackController.dispose();
    _levelDurationController.dispose();
    super.dispose();
  }

  Future<void> _pickDateTime() async {
    final now = DateTime.now();
    final date = await showDatePicker(
      context: context,
      initialDate: _scheduledStart ?? now.add(const Duration(hours: 1)),
      firstDate: now,
      lastDate: now.add(const Duration(days: 365)),
    );
    if (date == null || !mounted) return;

    final time = await showTimePicker(
      context: context,
      initialTime: _scheduledStart != null
          ? TimeOfDay.fromDateTime(_scheduledStart!)
          : TimeOfDay.fromDateTime(now.add(const Duration(hours: 1))),
    );
    if (time == null || !mounted) return;

    setState(() {
      _scheduledStart = DateTime(
        date.year,
        date.month,
        date.day,
        time.hour,
        time.minute,
      );
    });
  }

  Future<void> _createTournament() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isCreating = true);

    try {
      if (_tournamentType == 'sng') {
        await widget.apiService.createSng(
          clubId: widget.clubId,
          name: _nameController.text,
          buyIn: int.parse(_buyInController.text),
          maxPlayers: int.parse(_maxPlayersController.text),
          startingStack: int.parse(_startingStackController.text),
          levelDurationMins: int.parse(_levelDurationController.text),
        );
      } else {
        await widget.apiService.createMtt(
          clubId: widget.clubId,
          name: _nameController.text,
          buyIn: int.parse(_buyInController.text),
          maxPlayers: int.parse(_maxPlayersController.text),
          startingStack: int.parse(_startingStackController.text),
          levelDurationMins: int.parse(_levelDurationController.text),
          scheduledStart: _scheduledStart,
        );
      }

      if (mounted) {
        Navigator.pop(context);
        widget.onCreated();
        debugPrint('Tournament created successfully');
      }
    } catch (e) {
      if (mounted) {
        debugPrint('Tournament create error: $e');
      }
    } finally {
      if (mounted) {
        setState(() => _isCreating = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Create Tournament'),
      content: SingleChildScrollView(
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              DropdownButtonFormField<String>(
                value: _tournamentType,
                decoration: const InputDecoration(labelText: 'Type'),
                items: const [
                  DropdownMenuItem(value: 'sng', child: Text('Sit & Go')),
                  DropdownMenuItem(value: 'mtt', child: Text('Multi-Table')),
                ],
                onChanged: (value) {
                  setState(() => _tournamentType = value!);
                },
              ),
              TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(labelText: 'Name'),
                validator: (value) =>
                    value?.isEmpty ?? true ? 'Required' : null,
              ),
              TextFormField(
                controller: _buyInController,
                decoration: const InputDecoration(labelText: 'Buy-in'),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    value?.isEmpty ?? true ? 'Required' : null,
              ),
              TextFormField(
                controller: _maxPlayersController,
                decoration: const InputDecoration(labelText: 'Max Players'),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    value?.isEmpty ?? true ? 'Required' : null,
              ),
              TextFormField(
                controller: _startingStackController,
                decoration: const InputDecoration(labelText: 'Starting Stack'),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    value?.isEmpty ?? true ? 'Required' : null,
              ),
              TextFormField(
                controller: _levelDurationController,
                decoration: const InputDecoration(
                  labelText: 'Level Duration (minutes)',
                ),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    value?.isEmpty ?? true ? 'Required' : null,
              ),
              if (_tournamentType == 'mtt') ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        _scheduledStart != null
                            ? 'Start: ${_scheduledStart!.year}-${_scheduledStart!.month.toString().padLeft(2, '0')}-${_scheduledStart!.day.toString().padLeft(2, '0')} ${_scheduledStart!.hour.toString().padLeft(2, '0')}:${_scheduledStart!.minute.toString().padLeft(2, '0')}'
                            : 'No scheduled start',
                        style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.schedule),
                      onPressed: _pickDateTime,
                      tooltip: 'Pick date & time',
                    ),
                    if (_scheduledStart != null)
                      IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () => setState(() => _scheduledStart = null),
                        tooltip: 'Clear',
                      ),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: _isCreating ? null : () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: _isCreating ? null : _createTournament,
          child: _isCreating
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Text('Create'),
        ),
      ],
    );
  }
}
