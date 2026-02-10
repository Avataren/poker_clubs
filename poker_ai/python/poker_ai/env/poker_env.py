"""Gym-like wrapper around the Rust PokerEnv."""

from typing import Optional
import numpy as np

try:
    from poker_ai.engine import PokerEnv as RustPokerEnv, BatchPokerEnv as RustBatchPokerEnv
except ImportError:
    RustPokerEnv = None
    RustBatchPokerEnv = None


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
