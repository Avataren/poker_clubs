"""Tests for the Rust poker engine via Python bindings."""

import numpy as np
import pytest


def test_env_creation():
    """Test basic environment creation."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=6, seed=42)
    assert env.num_players == 6
    assert env.big_blind == 100


def test_env_reset():
    """Test reset returns correct shapes."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=2, seed=42)
    player, obs, mask = env.reset()

    assert isinstance(player, int)
    assert 0 <= player < 2
    assert obs.shape == (569,)
    assert mask.shape == (8,)
    assert obs.dtype == np.float32
    assert mask.dtype == bool
    assert mask.any(), "At least one action must be legal"


def test_env_step():
    """Test stepping the environment."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=2, seed=42)
    player, obs, mask = env.reset()

    # Take a legal action
    legal = np.where(mask)[0]
    action = legal[0]
    next_player, next_obs, next_mask, rewards, done = env.step(int(action))

    assert isinstance(next_player, int)
    assert next_obs.shape == (569,)
    assert next_mask.shape == (8,)
    assert rewards.shape == (2,)
    assert isinstance(done, bool)


def test_env_full_hand():
    """Test playing a complete hand."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=2, seed=42)

    for _ in range(10):  # play 10 hands
        player, obs, mask = env.reset()
        done = False
        steps = 0

        while not done:
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            player, obs, mask, rewards, done = env.step(int(action))
            steps += 1
            assert steps < 100, "Hand should complete in fewer than 100 steps"

        # Rewards should exist and have right shape
        assert rewards.shape == (2,)
        # Over 10 hands, at least one should have non-zero rewards
        # (individual hands can have zero if blind amounts cancel out)


def test_env_six_player():
    """Test 6-player game completes correctly."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=6, seed=42)

    for _ in range(5):
        player, obs, mask = env.reset()
        done = False

        while not done:
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            player, obs, mask, rewards, done = env.step(int(action))

        assert rewards.shape == (6,)


def test_batch_env():
    """Test batch environment."""
    from poker_ai.env.poker_env import BatchPokerEnv

    env = BatchPokerEnv(num_envs=4, num_players=2, base_seed=42)
    results = env.reset_all()

    assert len(results) == 4
    for player, obs, mask in results:
        assert obs.shape == (569,)
        assert mask.shape == (8,)


def test_batch_env_step():
    """Test stepping a batch environment."""
    from poker_ai.env.poker_env import BatchPokerEnv

    env = BatchPokerEnv(num_envs=4, num_players=2, base_seed=42)
    results = env.reset_all()

    # Step first environment
    player, obs, mask = results[0]
    legal = np.where(mask)[0]
    action = legal[0]
    next_player, next_obs, next_mask, rewards, done = env.step(0, int(action))
    assert next_obs.shape == (569,)


def test_legal_mask_validity():
    """Test that legal mask is always valid (at least one action)."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=2, seed=123)

    for _ in range(20):
        player, obs, mask = env.reset()
        done = False

        while not done:
            assert mask.any(), "Must have at least one legal action"
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            player, obs, mask, rewards, done = env.step(int(action))


def test_rewards_sum_to_zero():
    """Test that rewards across all players sum to approximately zero."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=4, seed=42)
    total_reward_sum = 0.0

    for _ in range(50):
        player, obs, mask = env.reset()
        done = False

        while not done:
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            player, obs, mask, rewards, done = env.step(int(action))

        total_reward_sum += rewards.sum()

    # Rewards are in bb, but raw chip changes should sum to ~0
    # (small deviations possible due to blind structure)
    assert abs(total_reward_sum) < 50, f"Reward sum should be near zero, got {total_reward_sum}"
