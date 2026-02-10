#!/usr/bin/env python3
"""Evaluate a trained model against baselines."""

import argparse

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.training.nfsp import NFSPTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained poker AI")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--num-hands", type=int, default=10000, help="Hands to play")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = NFSPConfig(num_players=2, device=args.device)
    trainer = NFSPTrainer(config)
    trainer.load_checkpoint(args.checkpoint)

    print(f"Evaluating over {args.num_hands} hands...")
    print()

    vs_random = trainer._eval_vs_random(num_hands=args.num_hands)
    vs_caller = trainer._eval_vs_caller(num_hands=args.num_hands)
    vs_tag = trainer._eval_vs_tag(num_hands=args.num_hands)

    print(f"Results:")
    print(
        f"  vs Random:  {vs_random.bb100:+.2f} +/- {vs_random.ci95:.2f} bb/100 "
        f"(95% CI, n={vs_random.num_hands})"
    )
    print(
        f"  vs Caller:  {vs_caller.bb100:+.2f} +/- {vs_caller.ci95:.2f} bb/100 "
        f"(95% CI, n={vs_caller.num_hands})"
    )
    print(
        f"  vs TAG:     {vs_tag.bb100:+.2f} +/- {vs_tag.ci95:.2f} bb/100 "
        f"(95% CI, n={vs_tag.num_hands})"
    )


if __name__ == "__main__":
    main()
