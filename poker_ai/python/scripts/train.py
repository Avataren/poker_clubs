#!/usr/bin/env python3
"""NFSP training entry point."""

import argparse

import torch
torch.set_float32_matmul_precision('high')

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.training.nfsp import NFSPTrainer


def main():
    parser = argparse.ArgumentParser(description="Train poker AI with NFSP")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--num-envs", type=int, default=1024, help="Parallel environments")
    parser.add_argument("--episodes", type=int, default=10_000_000, help="Total episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-hands", type=int, default=1_000, help="Hands per evaluation run")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eta", type=float, default=None, help="Fixed eta (overrides schedule)")
    parser.add_argument("--eta-start", type=float, default=None, help="Eta schedule start")
    parser.add_argument("--eta-end", type=float, default=None, help="Eta schedule end")
    parser.add_argument("--eta-ramp-steps", type=int, default=None, help="Eta linear ramp steps")
    parser.add_argument("--br-lr", type=float, default=1e-4)
    parser.add_argument("--as-lr", type=float, default=1e-4)
    parser.add_argument("--br-buffer-size", type=int, default=None, help="BR circular buffer size")
    parser.add_argument("--as-buffer-size", type=int, default=None, help="AS reservoir buffer size")
    parser.add_argument("--epsilon-start", type=float, default=None, help="Initial BR exploration epsilon")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Final BR exploration epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=None, help="Steps to linearly decay epsilon")
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Linear LR warmup steps")
    parser.add_argument("--lr-min-factor", type=float, default=None, help="Cosine LR decays to lr*factor")
    parser.add_argument("--tau", type=float, default=None, help="Polyak soft target update coefficient")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--br-train-steps", type=int, default=None, help="BR gradient steps per rollout")
    parser.add_argument("--as-train-steps", type=int, default=None, help="AS gradient steps per rollout")
    parser.add_argument("--huber-delta", type=float, default=None, help="Huber loss beta (default 10.0)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="Use async self-play + training (concurrent threads)")
    parser.add_argument("--sync-every", type=int, default=None,
                        help="Sync inference weights every N training rounds (async mode)")
    parser.add_argument("--train-ahead", type=int, default=None,
                        help="Max training rounds ahead of self-play (async mode, default 50)")
    parser.add_argument("--freeze-as", action="store_true",
                        help="Freeze AS network permanently (skip all AS training).")
    parser.add_argument("--as-freeze-duration", type=int, default=None,
                        help="Episodes to freeze AS after resume before unfreezing (e.g. 10000000). "
                             "Lets buffer accumulate diverse BR data before AS trains on it.")
    parser.add_argument("--as-warmup-episodes", type=int, default=None,
                        help="AS LR warmup episodes after unfreeze (ramp from 1%% to 100%%, default 2M)")
    parser.add_argument("--save-buffers", action="store_true",
                        help="Save replay buffers alongside checkpoints (large files, enables perfect resume)")
    parser.add_argument("--bootstrap-as", action="store_true",
                        help="On resume without saved buffers, pre-fill AS buffer by running AS self-play")
    parser.add_argument("--restart-schedules", action="store_true",
                        help="Warm restart: reset LR/epsilon/eta schedules to start values on resume")
    parser.add_argument("--reset-optimizers", action="store_true",
                        help="Reset Adam optimizer state on resume (use with --restart-schedules)")
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
    if args.br_buffer_size is not None:
        config_kwargs["br_buffer_size"] = args.br_buffer_size
    if args.as_buffer_size is not None:
        config_kwargs["as_buffer_size"] = args.as_buffer_size
    if args.lr_warmup_steps is not None:
        config_kwargs["lr_warmup_steps"] = args.lr_warmup_steps
    if args.lr_min_factor is not None:
        config_kwargs["lr_min_factor"] = args.lr_min_factor
    if args.tau is not None:
        config_kwargs["tau"] = args.tau
    if args.eta is not None:
        # Fixed eta: set start=end to disable ramp
        config_kwargs["eta_start"] = args.eta
        config_kwargs["eta_end"] = args.eta
    if args.eta_start is not None:
        config_kwargs["eta_start"] = args.eta_start
    if args.eta_end is not None:
        config_kwargs["eta_end"] = args.eta_end
    if args.eta_ramp_steps is not None:
        config_kwargs["eta_ramp_steps"] = args.eta_ramp_steps
    if args.huber_delta is not None:
        config_kwargs["huber_delta"] = args.huber_delta
    if args.no_amp:
        config_kwargs["use_amp"] = False
    if args.sync_every is not None:
        config_kwargs["sync_every"] = args.sync_every
    if args.train_ahead is not None:
        config_kwargs["train_ahead"] = args.train_ahead
    if args.freeze_as:
        config_kwargs["freeze_as"] = True
    if args.as_freeze_duration is not None:
        config_kwargs["as_freeze_duration"] = args.as_freeze_duration
    if args.as_warmup_episodes is not None:
        config_kwargs["as_warmup_episodes"] = args.as_warmup_episodes
    if args.save_buffers:
        config_kwargs["save_buffers"] = True

    config = NFSPConfig(**config_kwargs)

    if args.async_mode:
        from poker_ai.training.async_trainer import AsyncNFSPTrainer
        trainer = AsyncNFSPTrainer(config)
    else:
        trainer = NFSPTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        if args.restart_schedules:
            trainer.restart_schedules(reset_optimizers=args.reset_optimizers)
        if args.bootstrap_as:
            trainer.bootstrap_as_buffer()

    trainer.train()


if __name__ == "__main__":
    main()
