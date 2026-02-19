"""Gym-like wrapper around the Rust PokerEnv."""

from typing import Optional
import numpy as np

try:
    from poker_ai.engine import PokerEnv as RustPokerEnv, BatchPokerEnv as RustBatchPokerEnv
except ImportError:
    RustPokerEnv = None
    RustBatchPokerEnv = None

from poker_ai.model.state_encoder import OBS_SIZE
from poker_ai.env.action_space import NUM_ACTIONS


class PokerEnv:
    """Single-table poker environment wrapping the Rust engine."""

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        seed: Optional[int] = None,
    ):
        if RustPokerEnv is None:
            raise ImportError(
                "poker_ai_engine not built. Run: cd poker_ai/python && maturin develop --release"
            )
        self.env = RustPokerEnv(num_players, starting_stack, small_blind, big_blind, seed)
        self.num_players = num_players
        self.big_blind = big_blind

    def reset(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Reset and deal a new hand.

        Returns:
            (current_player, observation, legal_actions_mask)
        """
        player, obs, mask = self.env.reset()
        return player, np.array(obs, dtype=np.float32), np.array(mask, dtype=bool)

    def step(self, action: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Take an action.

        Returns:
            (current_player, observation, legal_mask, rewards, done)
        """
        player, obs, mask, rewards, done = self.env.step(action)
        return (
            player,
            np.array(obs, dtype=np.float32),
            np.array(mask, dtype=bool),
            np.array(rewards, dtype=np.float64),
            done,
        )

    def get_observation(self, seat: int) -> np.ndarray:
        return np.array(self.env.get_observation(seat), dtype=np.float32)

    def get_action_history(self) -> list[np.ndarray]:
        return [np.array(a, dtype=np.float32) for a in self.env.get_action_history()]

    @property
    def current_player(self) -> int:
        return self.env.current_player()

    @property
    def pot(self) -> int:
        return self.env.pot()

    @property
    def stacks(self) -> list[int]:
        return self.env.stacks()

    @property
    def is_done(self) -> bool:
        return self.env.is_done()


class BatchPokerEnv:
    """Batched environment for parallel self-play."""

    def __init__(
        self,
        num_envs: int = 64,
        num_players: int = 6,
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        base_seed: Optional[int] = None,
    ):
        if RustBatchPokerEnv is None:
            raise ImportError(
                "poker_ai_engine not built. Run: cd poker_ai/python && maturin develop --release"
            )
        self.env = RustBatchPokerEnv(
            num_envs, num_players, starting_stack, small_blind, big_blind, base_seed
        )
        self.num_envs = num_envs
        self.num_players = num_players
        self._has_dense_api = (
            hasattr(self.env, "step_batch_dense")
            and hasattr(self.env, "reset_batch_dense")
        )

    def reset_all(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        results = self.env.reset_all()
        return [
            (p, np.array(o, dtype=np.float32), np.array(m, dtype=bool))
            for p, o, m in results
        ]

    def step(self, env_idx: int, action: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, bool]:
        p, o, m, r, d = self.env.step(env_idx, action)
        return (
            p,
            np.array(o, dtype=np.float32),
            np.array(m, dtype=bool),
            np.array(r, dtype=np.float64),
            d,
        )

    def reset_env(self, env_idx: int) -> tuple[int, np.ndarray, np.ndarray]:
        p, o, m = self.env.reset_env(env_idx)
        return p, np.array(o, dtype=np.float32), np.array(m, dtype=bool)

    def step_batch(
        self, actions: list[tuple[int, int]]
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        """Step multiple envs in one FFI call."""
        players, obs_flat, masks_flat, rewards_flat, dones = self.env.step_batch(actions)
        n = len(actions)
        obs = np.array(obs_flat, dtype=np.float32).reshape(n, OBS_SIZE)
        masks = np.array(masks_flat, dtype=bool).reshape(n, NUM_ACTIONS)
        rewards = np.array(rewards_flat, dtype=np.float64).reshape(n, self.num_players)
        return players, obs, masks, rewards, dones

    def step_batch_dense(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dense self-play stepping path.

        Args:
            actions: action indices for all envs, shape (num_envs,)

        Returns:
            (players, obs, masks, rewards, dones)
        """
        actions_arr = np.asarray(actions, dtype=np.int64)
        n = int(actions_arr.shape[0])
        if self._has_dense_api:
            players, obs_bytes, masks_bytes, rewards_bytes, dones_bytes = self.env.step_batch_dense(
                actions_arr.tolist()
            )
            players_arr = np.asarray(players, dtype=np.intp)
            obs = np.frombuffer(obs_bytes, dtype=np.float32).reshape(n, OBS_SIZE)
            masks = (
                np.frombuffer(masks_bytes, dtype=np.uint8)
                .reshape(n, NUM_ACTIONS)
                .astype(bool, copy=False)
            )
            rewards = np.frombuffer(rewards_bytes, dtype=np.float32).reshape(
                n, self.num_players
            )
            dones = np.frombuffer(dones_bytes, dtype=np.uint8).astype(bool, copy=False)
            return players_arr, obs, masks, rewards, dones

        action_pairs = [(i, int(actions_arr[i])) for i in range(n)]
        players, obs, masks, rewards, dones = self.step_batch(action_pairs)
        return (
            np.asarray(players, dtype=np.intp),
            obs,
            masks,
            rewards.astype(np.float32, copy=False),
            np.asarray(dones, dtype=bool),
        )

    def reset_batch(
        self, env_indices: list[int]
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        """Reset multiple envs in one FFI call."""
        players, obs_flat, masks_flat = self.env.reset_batch(env_indices)
        n = len(env_indices)
        obs = np.array(obs_flat, dtype=np.float32).reshape(n, OBS_SIZE)
        masks = np.array(masks_flat, dtype=bool).reshape(n, NUM_ACTIONS)
        return players, obs, masks

    def reset_batch_dense(
        self, env_indices: np.ndarray | list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense self-play reset path."""
        env_idx_arr = np.asarray(env_indices, dtype=np.intp)
        n = int(env_idx_arr.shape[0])
        if self._has_dense_api:
            players, obs_bytes, masks_bytes = self.env.reset_batch_dense(
                env_idx_arr.tolist()
            )
            players_arr = np.asarray(players, dtype=np.intp)
            obs = np.frombuffer(obs_bytes, dtype=np.float32).reshape(n, OBS_SIZE)
            masks = (
                np.frombuffer(masks_bytes, dtype=np.uint8)
                .reshape(n, NUM_ACTIONS)
                .astype(bool, copy=False)
            )
            return players_arr, obs, masks

        players, obs, masks = self.reset_batch(env_idx_arr.tolist())
        return np.asarray(players, dtype=np.intp), obs, masks

    def reset_player_stats(self, env_indices: list[int], seat_indices: list[int]):
        """Reset HUD stats for specified env/seat pairs."""
        self.env.reset_player_stats(env_indices, seat_indices)
