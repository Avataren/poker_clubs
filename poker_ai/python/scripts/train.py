#!/usr/bin/env python3
"""NFSP training entry point."""

import argparse
import sys

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.training.nfsp import NFSPTrainer


def main():
    parser = argparse.ArgumentParser(description="Train poker AI with NFSP")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--episodes", type=int, default=10_000_000, help="Total episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eta", type=float, default=0.1, help="Anticipatory parameter")
    parser.add_argument("--br-lr", type=float, default=1e-4)
    parser.add_argument("--as-lr", type=float, default=5e-4)
    args = parser.parse_args()

    config = NFSPConfig(
        num_players=args.num_players,
        num_envs=args.num_envs,
        total_episodes=args.episodes,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        eta=args.eta,
        br_lr=args.br_lr,
        as_lr=args.as_lr,
    )

    trainer = NFSPTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
