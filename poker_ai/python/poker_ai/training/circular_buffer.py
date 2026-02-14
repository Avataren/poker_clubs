"""Circular replay buffer for Best Response (DQN) training.

Uses pre-allocated numpy arrays (Structure of Arrays) for zero-allocation
batch insert and fast random sampling.
"""

import threading

import numpy as np
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition (kept for API compatibility with evaluation code)."""
    obs: np.ndarray
    action_history: np.ndarray
    history_length: int
    action: int
    reward: float
    next_obs: np.ndarray
    next_action_history: np.ndarray
    next_history_length: int
    next_legal_mask: np.ndarray
    done: bool
    legal_mask: np.ndarray


class CircularBuffer:
    """Fixed-size circular replay buffer using pre-allocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int = 462, max_seq_len: int = 30, num_actions: int = 9, history_dim: int = 11):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_seq_len = max_seq_len
        self.num_actions = num_actions
        self.position = 0
        self.size = 0
        self._lock = threading.Lock()

        # Pre-allocated arrays
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_history = np.zeros((capacity, max_seq_len, history_dim), dtype=np.float32)
        self.history_length = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_action_history = np.zeros((capacity, max_seq_len, history_dim), dtype=np.float32)
        self.next_history_length = np.zeros(capacity, dtype=np.int64)
        self.next_legal_mask = np.zeros((capacity, num_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.legal_mask = np.zeros((capacity, num_actions), dtype=bool)

    def push(self, transition: Transition):
        i = self.position
        self.obs[i] = transition.obs
        self.action_history[i] = transition.action_history
        self.history_length[i] = transition.history_length
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.next_obs[i] = transition.next_obs
        self.next_action_history[i] = transition.next_action_history
        self.next_history_length[i] = transition.next_history_length
        self.next_legal_mask[i] = transition.next_legal_mask
        self.dones[i] = float(transition.done)
        self.legal_mask[i] = transition.legal_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(
        self,
        obs: np.ndarray,
        action_history: np.ndarray,
        history_length: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        next_action_history: np.ndarray,
        next_history_length: np.ndarray,
        next_legal_mask: np.ndarray,
        dones: np.ndarray,
        legal_mask: np.ndarray,
    ):
        """Push a batch of transitions. All arrays have first dim = batch_size."""
        n = len(obs)
        if n == 0:
            return

        with self._lock:
            pos = self.position
            # Check if we wrap around
            if pos + n <= self.capacity:
                s = slice(pos, pos + n)
                self.obs[s] = obs
                self.action_history[s] = action_history
                self.history_length[s] = history_length
                self.actions[s] = actions
                self.rewards[s] = rewards
                self.next_obs[s] = next_obs
                self.next_action_history[s] = next_action_history
                self.next_history_length[s] = next_history_length
                self.next_legal_mask[s] = next_legal_mask
                self.dones[s] = dones
                self.legal_mask[s] = legal_mask
            else:
                # Split across wrap boundary
                first = self.capacity - pos
                self.obs[pos:] = obs[:first]
                self.obs[:n - first] = obs[first:]
                self.action_history[pos:] = action_history[:first]
                self.action_history[:n - first] = action_history[first:]
                self.history_length[pos:] = history_length[:first]
                self.history_length[:n - first] = history_length[first:]
                self.actions[pos:] = actions[:first]
                self.actions[:n - first] = actions[first:]
                self.rewards[pos:] = rewards[:first]
                self.rewards[:n - first] = rewards[first:]
                self.next_obs[pos:] = next_obs[:first]
                self.next_obs[:n - first] = next_obs[first:]
                self.next_action_history[pos:] = next_action_history[:first]
                self.next_action_history[:n - first] = next_action_history[first:]
                self.next_history_length[pos:] = next_history_length[:first]
                self.next_history_length[:n - first] = next_history_length[first:]
                self.next_legal_mask[pos:] = next_legal_mask[:first]
                self.next_legal_mask[:n - first] = next_legal_mask[first:]
                self.dones[pos:] = dones[:first]
                self.dones[:n - first] = dones[first:]
                self.legal_mask[pos:] = legal_mask[:first]
                self.legal_mask[:n - first] = legal_mask[first:]

            self.position = (pos + n) % self.capacity
            self.size = min(self.size + n, self.capacity)

    def sample_arrays(self, batch_size: int) -> tuple:
        """Sample a batch and return raw numpy arrays (no Transition objects).

        Returns: (obs, ah, ah_len, actions, rewards, next_obs, next_ah,
                  next_ah_len, next_mask, dones, masks)
        """
        with self._lock:
            indices = np.random.randint(0, self.size, size=batch_size)
            return (
                self.obs[indices].copy(),
                self.action_history[indices].copy(),
                self.history_length[indices].copy(),
                self.actions[indices].copy(),
                self.rewards[indices].copy(),
                self.next_obs[indices].copy(),
                self.next_action_history[indices].copy(),
                self.next_history_length[indices].copy(),
                self.next_legal_mask[indices].copy(),
                self.dones[indices].copy(),
                self.legal_mask[indices].copy(),
            )

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample transitions (legacy API)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return [
            Transition(
                obs=self.obs[i],
                action_history=self.action_history[i],
                history_length=int(self.history_length[i]),
                action=int(self.actions[i]),
                reward=float(self.rewards[i]),
                next_obs=self.next_obs[i],
                next_action_history=self.next_action_history[i],
                next_history_length=int(self.next_history_length[i]),
                next_legal_mask=self.next_legal_mask[i],
                done=bool(self.dones[i]),
                legal_mask=self.legal_mask[i],
            )
            for i in indices
        ]

    def __len__(self) -> int:
        with self._lock:
            return self.size

    def save(self, path: str) -> None:
        """Save buffer contents to a compressed .npz file (float16 for large arrays)."""
        with self._lock:
            n = self.size
            if n == 0:
                return
            # Snapshot under lock, compress/write outside
            snapshot = dict(
                obs=self.obs[:n].astype(np.float16),
                action_history=self.action_history[:n].astype(np.float16),
                history_length=self.history_length[:n].copy(),
                actions=self.actions[:n].copy(),
                rewards=self.rewards[:n].astype(np.float16),
                next_obs=self.next_obs[:n].astype(np.float16),
                next_action_history=self.next_action_history[:n].astype(np.float16),
                next_history_length=self.next_history_length[:n].copy(),
                next_legal_mask=self.next_legal_mask[:n].copy(),
                dones=self.dones[:n].astype(np.float16),
                legal_mask=self.legal_mask[:n].copy(),
                position=np.array([self.position]),
                size=np.array([n]),
            )
        np.savez_compressed(path, **snapshot)

    def load(self, path: str) -> None:
        """Load buffer contents from a .npz file."""
        data = np.load(path)
        n = int(data["size"][0])
        if n == 0:
            return
        n = min(n, self.capacity)
        with self._lock:
            self.obs[:n] = data["obs"][:n].astype(np.float32)
            self.action_history[:n] = data["action_history"][:n].astype(np.float32)
            self.history_length[:n] = data["history_length"][:n]
            self.actions[:n] = data["actions"][:n]
            self.rewards[:n] = data["rewards"][:n].astype(np.float32)
            self.next_obs[:n] = data["next_obs"][:n].astype(np.float32)
            self.next_action_history[:n] = data["next_action_history"][:n].astype(np.float32)
            self.next_history_length[:n] = data["next_history_length"][:n]
            self.next_legal_mask[:n] = data["next_legal_mask"][:n]
            self.dones[:n] = data["dones"][:n].astype(np.float32)
            self.legal_mask[:n] = data["legal_mask"][:n]
            self.position = int(data["position"][0]) % self.capacity
            self.size = n
