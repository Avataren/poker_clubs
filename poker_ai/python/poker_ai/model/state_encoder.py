"""State encoding utilities.

The Rust engine produces 630 floats directly. This module provides
utilities for when you need to manipulate or inspect the encoding
on the Python side.
"""

# Feature layout in the 630-float observation vector:
# [0..104)     = Hole cards (2 x 52 one-hot)
# [104..364)   = Community cards (5 x 52 one-hot, zero-padded)
# [364..450)   = Game state (86 floats: phase, ratios, position, opponents, opponent stats)
# [450..578)   = History hidden state placeholder (128 zeros, filled by Python)
# [578..630)   = Hand strength features (52 floats)

HOLE_CARDS_START = 0
HOLE_CARDS_END = 104
COMMUNITY_START = 104
COMMUNITY_END = 364
GAME_STATE_START = 364
GAME_STATE_END = 450
HISTORY_PLACEHOLDER_START = 450
HISTORY_PLACEHOLDER_END = 578
HAND_STRENGTH_START = 578
HAND_STRENGTH_END = 630

# Total observation size
OBS_SIZE = 630

# Size of features that go into the main network (excluding history placeholder)
# Card features + game state + hand strength = 364 + 86 + 52 = 502
# History output (256) is concatenated separately
STATIC_FEATURE_SIZE = 502
HISTORY_OUTPUT_SIZE = 256
