"""Reservoir sampling buffer for Average Strategy (supervised) training.

Uses pre-allocated numpy arrays for zero-allocation batch operations.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SLTransition:
    """Single SL transition (kept for API compatibility)."""
    obs: np.ndarray
    action_history: np.ndarray
    history_length: int
    action: int
    legal_mask: np.ndarray


class ReservoirBuffer:
    """Reservoir sampling buffer using pre-allocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int = 441, max_seq_len: int = 30, num_actions: int = 8):
        self.capacity = capacity
        self.size = 0
        self.total_seen = 0

        # Pre-allocated arrays
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_history = np.zeros((capacity, max_seq_len, 7), dtype=np.float32)
        self.history_length = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.legal_mask = np.zeros((capacity, num_actions), dtype=bool)

    def push(self, transition: SLTransition):
        self.total_seen += 1
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            idx = np.random.randint(0, self.total_seen)
            if idx >= self.capacity:
                return
            i = idx

        self.obs[i] = transition.obs
        self.action_history[i] = transition.action_history
        self.history_length[i] = transition.history_length
        self.actions[i] = transition.action
        self.legal_mask[i] = transition.legal_mask

    def push_batch(
        self,
        obs: np.ndarray,
        action_history: np.ndarray,
        history_length: np.ndarray,
        actions: np.ndarray,
        legal_mask: np.ndarray,
    ):
        """Push a batch using reservoir sampling."""
        n = len(obs)
        if n == 0:
            return

        for j in range(n):
            self.total_seen += 1
            if self.size < self.capacity:
                i = self.size
                self.size += 1
            else:
                idx = np.random.randint(0, self.total_seen)
                if idx >= self.capacity:
                    continue
                i = idx

            self.obs[i] = obs[j]
            self.action_history[i] = action_history[j]
            self.history_length[i] = history_length[j]
            self.actions[i] = actions[j]
            self.legal_mask[i] = legal_mask[j]

    def sample_arrays(self, batch_size: int) -> tuple:
        """Sample and return raw numpy arrays.

        Returns: (obs, ah, ah_len, actions, masks)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[indices],
            self.action_history[indices],
            self.history_length[indices],
            self.actions[indices],
            self.legal_mask[indices],
        )

    def sample(self, batch_size: int) -> list[SLTransition]:
        """Sample transitions (legacy API)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return [
            SLTransition(
                obs=self.obs[i],
                action_history=self.action_history[i],
                history_length=int(self.history_length[i]),
                action=int(self.actions[i]),
                legal_mask=self.legal_mask[i],
            )
            for i in indices
        ]

    def __len__(self) -> int:
        return self.size
