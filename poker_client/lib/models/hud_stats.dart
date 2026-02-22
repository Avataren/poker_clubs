/// Per-player HUD statistics tracked client-side from observed actions.
/// Mirrors the bot's OpponentTracker logic but accumulated on the client.
class PlayerHudStats {
  int hands = 0;
  int vpipCount = 0;  // preflop voluntary call/raise
  int pfrCount = 0;   // preflop raise/all-in
  int totalRaises = 0; // raises + all-ins across all streets
  int totalCalls = 0;  // calls across all streets

  double get vpip => hands > 0 ? vpipCount / hands : 0.0;
  double get pfr => hands > 0 ? pfrCount / hands : 0.0;
  // AF: raises per call. Undefined if no calls; show 4+ for pure aggression.
  double get af =>
      totalCalls > 0 ? totalRaises / totalCalls : (totalRaises > 0 ? 4.0 : 0.0);
}
