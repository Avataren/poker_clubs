#!/usr/bin/env python3
"""Benchmark the Rust poker engine speed."""

import time
import argparse
import numpy as np


def benchmark_single(num_hands: int, num_players: int):
    """Benchmark single environment."""
    from poker_ai.env.poker_env import PokerEnv

    env = PokerEnv(num_players=num_players, seed=42)

    start = time.perf_counter()
    for _ in range(num_hands):
        player, obs, mask = env.reset()
        done = False
        while not done:
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            player, obs, mask, rewards, done = env.step(action)
    elapsed = time.perf_counter() - start

    hands_per_sec = num_hands / elapsed
    print(f"Single env ({num_players}p): {num_hands:,} hands in {elapsed:.2f}s = {hands_per_sec:,.0f} hands/sec")
    return hands_per_sec


def benchmark_batch(num_hands: int, num_envs: int, num_players: int):
    """Benchmark batch environment."""
    from poker_ai.env.poker_env import BatchPokerEnv

    env = BatchPokerEnv(num_envs=num_envs, num_players=num_players, base_seed=42)
    results = env.reset_all()

    hands_completed = 0
    start = time.perf_counter()

    while hands_completed < num_hands:
        for env_idx in range(num_envs):
            player, obs, mask = results[env_idx]
            legal = np.where(mask)[0]
            action = np.random.choice(legal) if len(legal) > 0 else 1
            player, obs, mask, rewards, done = env.step(env_idx, action)

            if done:
                hands_completed += 1
                if hands_completed >= num_hands:
                    break
                player, obs, mask = env.reset_env(env_idx)
                results[env_idx] = (player, obs, mask)
            else:
                results[env_idx] = (player, obs, mask)

    elapsed = time.perf_counter() - start
    hands_per_sec = num_hands / elapsed
    print(f"Batch env ({num_envs}x{num_players}p): {num_hands:,} hands in {elapsed:.2f}s = {hands_per_sec:,.0f} hands/sec")
    return hands_per_sec


def main():
    parser = argparse.ArgumentParser(description="Benchmark poker engine")
    parser.add_argument("--hands", type=int, default=100_000, help="Number of hands")
    parser.add_argument("--num-players", type=int, default=6, help="Players per table")
    parser.add_argument("--num-envs", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    print("=== Poker Engine Benchmark ===")
    print()

    # Single env
    for np_ in [2, 6, 9]:
        benchmark_single(args.hands, np_)
    print()

    # Batch env
    benchmark_batch(args.hands, args.num_envs, args.num_players)


if __name__ == "__main__":
    main()
