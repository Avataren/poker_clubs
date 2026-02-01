import 'package:audioplayers/audioplayers.dart';

class SoundService {
  static final SoundService _instance = SoundService._internal();
  factory SoundService() => _instance;
  SoundService._internal();

  final AudioPlayer _player = AudioPlayer();
  bool _enabled = true;

  bool get enabled => _enabled;
  set enabled(bool value) => _enabled = value;

  Future<void> playChipBet(int amount) async {
    if (!_enabled) return;

    // Use single chip sound for small bets, multiple chips for larger bets
    final soundFile = amount <= 50
        ? 'sounds/poker_chips_dropping.ogg'
        : 'sounds/multiple_chips_dropping.ogg';

    await _player.play(AssetSource(soundFile));
  }

  Future<void> playAllIn() async {
    if (!_enabled) return;
    await _player.play(AssetSource('sounds/all_in.ogg'));
  }

  Future<void> playFold() async {
    if (!_enabled) return;
    await _player.play(AssetSource('sounds/fold.ogg'));
  }

  Future<void> playShuffle() async {
    if (!_enabled) return;
    await _player.play(AssetSource('sounds/shuffle.ogg'));
  }

  void dispose() {
    _player.dispose();
  }
}
