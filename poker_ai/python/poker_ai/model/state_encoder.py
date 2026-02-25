"""State encoding utilities.

The Rust engine produces 710 floats directly. This module provides
utilities for when you need to manipulate or inspect the encoding
on the Python side.
"""

# Feature layout in the 710-float observation vector:
# [0..104)     = Hole cards (2 x 52 one-hot)
# [104..364)   = Community cards (5 x 52 one-hot, zero-padded)
# [364..530)   = Game state (166 floats: phase, ratios, position, opponents, opponent stats)
# [530..658)   = History hidden state placeholder (128 zeros, filled by Python)
# [658..710)   = Hand strength features (52 floats):
#   [658]        hand_rank.normalized()       0 preflop; postflop rs_poker rank in [0,1]
#   [659]        preflop_strength(N)          HU equity scaled to N opponents, constant across streets
#   [660..666)   board_texture (6)            flush_draw, straight_draw, paired, trips, high_card, n_cards
#   [666..668)   hero draw potential (2)      flush_draw, straight_draw (hero contributes ≥1 card)
#   [668..677)   hand category one-hot (9)    high_card, one_pair, two_pair, trips, straight,
#                                             flush, full_house, quads, straight_flush  (all 0 preflop)
#   [677]        stack / (200 BB) clamped     explicit short/deep-stack signal
#   [678..710)   reserved (32 zeros)

HOLE_CARDS_START = 0
HOLE_CARDS_END = 104
COMMUNITY_START = 104
COMMUNITY_END = 364
GAME_STATE_START = 364
GAME_STATE_END = 530
HISTORY_PLACEHOLDER_START = 530
HISTORY_PLACEHOLDER_END = 658
HAND_STRENGTH_START = 658
HAND_STRENGTH_END = 710

# Total observation size
OBS_SIZE = 710

# Size of features that go into the main network (excluding history placeholder)
# Card features + game state + hand strength = 364 + 166 + 52 = 582
# History output (256) is concatenated separately
STATIC_FEATURE_SIZE = 582
HISTORY_OUTPUT_SIZE = 256
