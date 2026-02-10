"""Card encoding utilities (used if doing encoding on Python side)."""

import numpy as np


def card_to_onehot(rank: int, suit: int) -> np.ndarray:
    """Encode a single card as 52-dim one-hot vector."""
    vec = np.zeros(52, dtype=np.float32)
    idx = suit * 13 + (rank - 2)
    if 0 <= idx < 52:
        vec[idx] = 1.0
    return vec


def encode_hole_cards(cards: list[tuple[int, int]]) -> np.ndarray:
    """Encode 2 hole cards as 104-dim vector (2 x 52 one-hot)."""
    result = np.zeros(104, dtype=np.float32)
    for i, (rank, suit) in enumerate(cards[:2]):
        idx = suit * 13 + (rank - 2)
        if 0 <= idx < 52:
            result[i * 52 + idx] = 1.0
    return result


def encode_community(cards: list[tuple[int, int]], max_cards: int = 5) -> np.ndarray:
    """Encode community cards as (max_cards x 52) one-hot, zero-padded."""
    result = np.zeros(max_cards * 52, dtype=np.float32)
    for i, (rank, suit) in enumerate(cards[:max_cards]):
        idx = suit * 13 + (rank - 2)
        if 0 <= idx < 52:
            result[i * 52 + idx] = 1.0
    return result
