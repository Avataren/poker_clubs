"""Circular replay buffer for Best Response (DQN) training."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Transition:
    obs: np.ndarray           # (441,) static features
    action_history: np.ndarray  # (max_seq, 7) padded
    history_length: int
    action: int
    reward: float
    next_obs: np.ndarray      # (441,)
    next_action_history: np.ndarray
    next_history_length: int
    next_legal_mask: np.ndarray  # (8,)
    done: bool
    legal_mask: np.ndarray    # (8,)


class CircularBuffer:
    """Fixed-size circular replay buffer for RL transitions."""

    def __init__(self, capacity: int, max_seq_len: int = 200):
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.buffer: list[Transition | None] = [None] * capacity
        self.position = 0
        self.size = 0

    def push(self, transition: Transition):
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return self.size
