import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../widgets/avatar_sprite.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _loading = true;
  bool _saving = false;
  String _username = '';
  int _avatarIndex = 0;
  String _deckStyle = 'classic';

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    try {
      final profile = await context.read<ApiService>().getMyProfile();
      if (!mounted) return;
      setState(() {
        _username = profile.username;
        _avatarIndex = profile.avatarIndex;
        _deckStyle = profile.deckStyle;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to load settings: $e')));
    }
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      final profile = await context.read<ApiService>().updateMyProfile(
        avatarIndex: _avatarIndex,
        deckStyle: _deckStyle,
      );
      if (!mounted) return;
      setState(() {
        _username = profile.username;
        _avatarIndex = profile.avatarIndex;
        _deckStyle = profile.deckStyle;
        _saving = false;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Settings saved')));
    } catch (e) {
      if (!mounted) return;
      setState(() => _saving = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to save settings: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final isCompact = MediaQuery.of(context).size.width < 520;
    final avatarImageSize = isCompact ? 80.0 : 96.0;
    final avatarTileSize = isCompact ? 84.0 : 100.0;
    const avatarGridSpacing = 8.0;
    final avatarGridWidth =
        (avatarTileSize * AvatarSprite.cols) +
        (avatarGridSpacing * (AvatarSprite.cols - 1));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        actions: [
          TextButton.icon(
            onPressed: _saving || _loading ? null : _save,
            icon: _saving
                ? const SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.save),
            label: const Text('Save'),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _username.isEmpty
                        ? 'Player Settings'
                        : 'Player: $_username',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 16),

                  Text(
                    'Deck style',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    initialValue: _deckStyle,
                    items: const [
                      DropdownMenuItem(
                        value: 'classic',
                        child: Text('Classic'),
                      ),
                      DropdownMenuItem(
                        value: 'multi_color',
                        child: Text('Multi-color'),
                      ),
                    ],
                    onChanged: (value) {
                      if (value == null) return;
                      setState(() => _deckStyle = value);
                    },
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                    ),
                  ),

                  const SizedBox(height: 24),
                  Text(
                    'Profile picture',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Select one portrait from the sprite sheet.',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  const SizedBox(height: 12),

                  Center(
                    child: Container(
                      width: avatarTileSize,
                      height: avatarTileSize,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.greenAccent, width: 2),
                      ),
                      child: AvatarSprite(
                        avatarIndex: _avatarIndex,
                        size: avatarImageSize,
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),
                  Center(
                    child: SizedBox(
                      width: avatarGridWidth,
                      child: GridView.builder(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: AvatarSprite.totalAvatars,
                        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: AvatarSprite.cols,
                          crossAxisSpacing: avatarGridSpacing,
                          mainAxisSpacing: avatarGridSpacing,
                          mainAxisExtent: avatarTileSize,
                        ),
                        itemBuilder: (context, index) {
                          final selected = index == _avatarIndex;
                          return InkWell(
                            onTap: () => setState(() => _avatarIndex = index),
                            borderRadius: BorderRadius.circular(999),
                            child: Container(
                              width: avatarTileSize,
                              height: avatarTileSize,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color: selected
                                      ? Colors.greenAccent
                                      : Colors.white24,
                                  width: selected ? 2.5 : 1,
                                ),
                              ),
                              padding: const EdgeInsets.all(2),
                              child: AvatarSprite(
                                avatarIndex: index,
                                size: avatarImageSize,
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}
