"""State encoding utilities.

The Rust engine produces 569 floats directly. This module provides
utilities for when you need to manipulate or inspect the encoding
on the Python side.
"""

# Feature layout in the 569-float observation vector:
# [0..104)     = Hole cards (2 x 52 one-hot)
# [104..364)   = Community cards (5 x 52 one-hot, zero-padded)
# [364..389)   = Game state (25 floats: phase, ratios, position, etc.)
# [389..517)   = LSTM hidden state placeholder (128 zeros, filled by Python)
# [517..569)   = Hand strength features (52 floats)

HOLE_CARDS_START = 0
HOLE_CARDS_END = 104
COMMUNITY_START = 104
COMMUNITY_END = 364
GAME_STATE_START = 364
GAME_STATE_END = 389
LSTM_PLACEHOLDER_START = 389
LSTM_PLACEHOLDER_END = 517
HAND_STRENGTH_START = 517
HAND_STRENGTH_END = 569

# Total observation size
OBS_SIZE = 569

# Size of features that go into the main network (excluding LSTM placeholder)
# Card features + game state + hand strength = 364 + 25 + 52 = 441
# LSTM output (128) is concatenated separately
STATIC_FEATURE_SIZE = 441
LSTM_OUTPUT_SIZE = 128
