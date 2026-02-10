"""Reservoir sampling buffer for Average Strategy (supervised) training."""

import numpy as np
from dataclasses import dataclass


@dataclass
class SLTransition:
    obs: np.ndarray           # (441,) static features
    action_history: np.ndarray  # (max_seq, 7) padded
    history_length: int
    action: int
    legal_mask: np.ndarray    # (8,)


class ReservoirBuffer:
    """Reservoir sampling buffer for supervised learning transitions.

    Maintains a uniform random sample of all transitions ever seen,
    even as the number of transitions exceeds the buffer capacity.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[SLTransition | None] = [None] * capacity
        self.size = 0
        self.total_seen = 0

    def push(self, transition: SLTransition):
        self.total_seen += 1
        if self.size < self.capacity:
            self.buffer[self.size] = transition
            self.size += 1
        else:
            # Reservoir sampling: replace with probability capacity/total_seen
            idx = np.random.randint(0, self.total_seen)
            if idx < self.capacity:
                self.buffer[idx] = transition

    def sample(self, batch_size: int) -> list[SLTransition]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return self.size
