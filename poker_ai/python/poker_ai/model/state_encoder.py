"""State encoding utilities.

The Rust engine produces 590 floats directly. This module provides
utilities for when you need to manipulate or inspect the encoding
on the Python side.
"""

# Feature layout in the 590-float observation vector:
# [0..104)     = Hole cards (2 x 52 one-hot)
# [104..364)   = Community cards (5 x 52 one-hot, zero-padded)
# [364..410)   = Game state (46 floats: phase, ratios, position, new features, opponents)
# [410..538)   = History hidden state placeholder (128 zeros, filled by Python)
# [538..590)   = Hand strength features (52 floats)

HOLE_CARDS_START = 0
HOLE_CARDS_END = 104
COMMUNITY_START = 104
COMMUNITY_END = 364
GAME_STATE_START = 364
GAME_STATE_END = 410
HISTORY_PLACEHOLDER_START = 410
HISTORY_PLACEHOLDER_END = 538
HAND_STRENGTH_START = 538
HAND_STRENGTH_END = 590

# Total observation size
OBS_SIZE = 590

# Size of features that go into the main network (excluding history placeholder)
# Card features + game state + hand strength = 364 + 46 + 52 = 462
# History output (256) is concatenated separately
STATIC_FEATURE_SIZE = 462
HISTORY_OUTPUT_SIZE = 256
