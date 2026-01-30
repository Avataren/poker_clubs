class PokerCard {
  final int rank; // 2-14 (11=J, 12=Q, 13=K, 14=A)
  final int suit; // 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
  final bool faceUp;
  final bool highlighted; // Whether this card is part of the winning hand

  PokerCard({
    required this.rank,
    required this.suit,
    this.faceUp = true,
    this.highlighted = false,
  });

  factory PokerCard.fromJson(Map<String, dynamic> json) {
    final highlighted = json['highlighted'] as bool? ?? false;
    print(
      'Card fromJson: rank=${json['rank']}, suit=${json['suit']}, highlighted=${json['highlighted']} -> $highlighted',
    );

    return PokerCard(
      rank: json['rank'] as int,
      suit: json['suit'] as int,
      faceUp: json['face_up'] as bool? ?? true,
      highlighted: highlighted,
    );
  }

  // Factory for creating a face-down card
  factory PokerCard.faceDown() {
    return PokerCard(rank: 0, suit: 0, faceUp: false, highlighted: false);
  }

  String get rankStr {
    if (rank == 14) return 'A';
    if (rank == 13) return 'K';
    if (rank == 12) return 'Q';
    if (rank == 11) return 'J';
    return rank.toString();
  }

  String get suitStr {
    const suits = ['♣', '♦', '♥', '♠'];
    return suits[suit];
  }

  bool get isRed => suit == 1 || suit == 2;

  @override
  String toString() => '$rankStr$suitStr';
}
