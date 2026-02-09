/// Live tournament info state received via periodic broadcasts.
class TournamentInfoState {
  String? tournamentId;
  String? serverTime;
  int? level;
  int? smallBlind;
  int? bigBlind;
  int? ante;
  String? levelStartTime;
  int? levelDurationSecs;
  int? levelTimeRemainingSecs;
  int? nextSmallBlind;
  int? nextBigBlind;

  void update({
    required String tournamentId,
    required String serverTime,
    required int level,
    required int smallBlind,
    required int bigBlind,
    required int ante,
    required String levelStartTime,
    required int levelDurationSecs,
    required int levelTimeRemainingSecs,
    int? nextSmallBlind,
    int? nextBigBlind,
  }) {
    this.tournamentId = tournamentId;
    this.serverTime = serverTime;
    this.level = level;
    this.smallBlind = smallBlind;
    this.bigBlind = bigBlind;
    this.ante = ante;
    this.levelStartTime = levelStartTime;
    this.levelDurationSecs = levelDurationSecs;
    this.levelTimeRemainingSecs = levelTimeRemainingSecs;
    this.nextSmallBlind = nextSmallBlind;
    this.nextBigBlind = nextBigBlind;
  }
}
