#!/usr/bin/env python3
"""NFSP training entry point."""

import argparse

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.training.nfsp import NFSPTrainer


def main():
    parser = argparse.ArgumentParser(description="Train poker AI with NFSP")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--num-envs", type=int, default=512, help="Parallel environments")
    parser.add_argument("--episodes", type=int, default=10_000_000, help="Total episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-hands", type=int, default=1_000, help="Hands per evaluation run")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eta", type=float, default=0.1, help="Anticipatory parameter")
    parser.add_argument("--br-lr", type=float, default=1e-4)
    parser.add_argument("--as-lr", type=float, default=5e-4)
    parser.add_argument("--epsilon-start", type=float, default=None, help="Initial BR exploration epsilon")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Final BR exploration epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=None, help="Steps to linearly decay epsilon")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--br-train-steps", type=int, default=None, help="BR gradient steps per rollout")
    parser.add_argument("--as-train-steps", type=int, default=None, help="AS gradient steps per rollout")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    args = parser.parse_args()

    config_kwargs = dict(
        num_players=args.num_players,
        num_envs=args.num_envs,
        total_episodes=args.episodes,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        eval_every=args.eval_every,
        eval_hands=args.eval_hands,
        checkpoint_every=args.checkpoint_every,
        eta=args.eta,
        br_lr=args.br_lr,
        as_lr=args.as_lr,
    )
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.br_train_steps is not None:
        config_kwargs["br_train_steps"] = args.br_train_steps
    if args.as_train_steps is not None:
        config_kwargs["as_train_steps"] = args.as_train_steps
    if args.epsilon_start is not None:
        config_kwargs["epsilon_start"] = args.epsilon_start
    if args.epsilon_end is not None:
        config_kwargs["epsilon_end"] = args.epsilon_end
    if args.epsilon_decay_steps is not None:
        config_kwargs["epsilon_decay_steps"] = args.epsilon_decay_steps
    if args.no_amp:
        config_kwargs["use_amp"] = False

    config = NFSPConfig(**config_kwargs)

    trainer = NFSPTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
