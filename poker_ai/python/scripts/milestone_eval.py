#!/usr/bin/env python3
"""Evaluate milestone checkpoints with high-sample confidence intervals.

This script is intended to be re-run during training. It scans checkpoint files,
evaluates milestone episodes (e.g. every 1M), appends results to CSV, and can
promote the best checkpoint by TAG lower confidence bound.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

# Ensure local package imports work when running this file directly.
_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.training.nfsp import NFSPTrainer


_CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")
_CSV_COLUMNS = [
    "timestamp_utc",
    "episode",
    "checkpoint_path",
    "num_hands",
    "random_bb100",
    "random_ci95",
    "random_lb95",
    "caller_bb100",
    "caller_ci95",
    "caller_lb95",
    "tag_bb100",
    "tag_ci95",
    "tag_lb95",
    "tag_seat0_bb100",
    "tag_seat1_bb100",
    "tag_bluff_pct",
    "tag_thin_value_pct",
    "tag_value_bet_pct",
    "exploitability_hands",
    "exploitability_bb100",
    "exploitability_ci95",
    "exploitability_ub95",
    "exploitability_seat0_bb100",
    "exploitability_seat1_bb100",
    "promoted",
]


def _load_done_episodes(csv_path: Path) -> set[int]:
    done: set[int] = set()
    if not csv_path.exists():
        return done

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["episode"]))
            except (KeyError, ValueError):
                continue
    return done


def _discover_milestones(
    checkpoint_dir: Path,
    milestone_every: int,
    min_episode: int,
    max_episode: int | None,
    done_episodes: set[int],
) -> list[tuple[int, Path]]:
    milestones: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_*.pt"):
        m = _CKPT_RE.match(path.name)
        if not m:
            continue
        episode = int(m.group(1))
        if episode < min_episode:
            continue
        if max_episode is not None and episode > max_episode:
            continue
        if episode % milestone_every != 0:
            continue
        if episode in done_episodes:
            continue
        milestones.append((episode, path))
    milestones.sort(key=lambda x: x[0])
    return milestones


def _write_header_if_needed(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)
        return

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        if existing_header == _CSV_COLUMNS:
            return
        rows = list(reader)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {col: row.get(col, "") for col in _CSV_COLUMNS}
            writer.writerow(out)


def _append_row(csv_path: Path, row: list[object]):
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _evaluate_and_record(
    trainer: NFSPTrainer,
    checkpoint_path: Path,
    episode: int,
    num_hands: int,
    exploitability_hands: int,
    csv_path: Path,
    best_path: Path | None,
    min_tag_lb95_for_promotion: float,
) -> bool:
    trainer.load_checkpoint(str(checkpoint_path), load_buffers=False)
    vs_random = trainer._eval_vs_random(num_hands=num_hands)
    vs_caller = trainer._eval_vs_caller(num_hands=num_hands)
    vs_tag = trainer._eval_vs_tag(num_hands=num_hands)
    vs_exploit = trainer.eval_exploitability_proxy(num_hands=exploitability_hands)

    random_lb95 = vs_random.bb100 - vs_random.ci95
    caller_lb95 = vs_caller.bb100 - vs_caller.ci95
    tag_lb95 = vs_tag.bb100 - vs_tag.ci95
    exploit_ub95 = vs_exploit.bb100 + vs_exploit.ci95

    promoted = False
    if best_path is not None and tag_lb95 >= min_tag_lb95_for_promotion:
        if best_path.exists():
            # Evaluate current best for fair comparison.
            trainer.load_checkpoint(str(best_path))
            best_tag = trainer._eval_vs_tag(num_hands=num_hands)
            best_tag_lb95 = best_tag.bb100 - best_tag.ci95
        else:
            best_tag_lb95 = float("-inf")
        if tag_lb95 > best_tag_lb95:
            shutil.copy2(checkpoint_path, best_path)
            promoted = True

    print(
        f"Milestone {episode:,}: "
        f"Random {vs_random.bb100:+.2f} +/- {vs_random.ci95:.2f}, "
        f"Caller {vs_caller.bb100:+.2f} +/- {vs_caller.ci95:.2f}, "
        f"TAG {vs_tag.bb100:+.2f} +/- {vs_tag.ci95:.2f} bb/100 "
        f"(TAG LB95={tag_lb95:+.2f}) | "
        f"Bluff={vs_tag.bluff_pct:.1f}% Thin={vs_tag.thin_value_pct:.1f}% "
        f"Value={vs_tag.value_bet_pct:.1f}% | "
        f"ExploitProxy {vs_exploit.bb100:+.2f} +/- {vs_exploit.ci95:.2f} "
        f"(UB95={exploit_ub95:+.2f}, n={exploitability_hands})"
    )
    if promoted and best_path is not None:
        print(f"  Promoted checkpoint to {best_path}")

    _append_row(
        csv_path,
        [
            dt.datetime.now(dt.UTC).isoformat(),
            episode,
            str(checkpoint_path),
            num_hands,
            f"{vs_random.bb100:.6f}",
            f"{vs_random.ci95:.6f}",
            f"{random_lb95:.6f}",
            f"{vs_caller.bb100:.6f}",
            f"{vs_caller.ci95:.6f}",
            f"{caller_lb95:.6f}",
            f"{vs_tag.bb100:.6f}",
            f"{vs_tag.ci95:.6f}",
            f"{tag_lb95:.6f}",
            f"{vs_tag.seat0_bb100:.6f}",
            f"{vs_tag.seat1_bb100:.6f}",
            f"{vs_tag.bluff_pct:.6f}",
            f"{vs_tag.thin_value_pct:.6f}",
            f"{vs_tag.value_bet_pct:.6f}",
            exploitability_hands,
            f"{vs_exploit.bb100:.6f}",
            f"{vs_exploit.ci95:.6f}",
            f"{exploit_ub95:.6f}",
            f"{vs_exploit.seat0_bb100:.6f}",
            f"{vs_exploit.seat1_bb100:.6f}",
            int(promoted),
        ],
    )
    return promoted


def main():
    parser = argparse.ArgumentParser(description="Evaluate milestone poker checkpoints")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory with checkpoint_*.pt files")
    parser.add_argument("--device", default="cuda", help="Device for evaluation")
    parser.add_argument("--num-hands", type=int, default=100_000, help="Hands per milestone evaluation")
    parser.add_argument(
        "--exploitability-hands",
        type=int,
        default=None,
        help="Hands for exploitability proxy eval (default: same as --num-hands)",
    )
    parser.add_argument(
        "--milestone-every",
        type=int,
        default=1_000_000,
        help="Evaluate checkpoints that are multiples of this episode count",
    )
    parser.add_argument("--min-episode", type=int, default=1_000_000, help="Minimum episode to evaluate")
    parser.add_argument("--max-episode", type=int, default=None, help="Maximum episode to evaluate")
    parser.add_argument("--csv", default="logs/milestones.csv", help="CSV output path")
    parser.add_argument(
        "--best-path",
        default="checkpoints/checkpoint_best.pt",
        help="Path to store promoted best checkpoint (set empty string to disable)",
    )
    parser.add_argument(
        "--min-tag-lb95-for-promotion",
        type=float,
        default=0.0,
        help="Require TAG lower 95%% bound to exceed this before promotion",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP for milestone evaluation")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    csv_path = Path(args.csv)
    best_path = Path(args.best_path) if args.best_path else None
    exploitability_hands = (
        args.exploitability_hands
        if args.exploitability_hands is not None
        else args.num_hands
    )

    done_episodes = _load_done_episodes(csv_path)
    milestones = _discover_milestones(
        checkpoint_dir=checkpoint_dir,
        milestone_every=args.milestone_every,
        min_episode=args.min_episode,
        max_episode=args.max_episode,
        done_episodes=done_episodes,
    )

    if not milestones:
        print("No new milestone checkpoints found.")
        return

    _write_header_if_needed(csv_path)
    config = NFSPConfig(
        num_players=2,
        device=args.device,
        use_amp=False if args.no_amp else None,
    )
    trainer = NFSPTrainer(config)

    for episode, checkpoint_path in milestones:
        try:
            _evaluate_and_record(
                trainer=trainer,
                checkpoint_path=checkpoint_path,
                episode=episode,
                num_hands=args.num_hands,
                exploitability_hands=exploitability_hands,
                csv_path=csv_path,
                best_path=best_path,
                min_tag_lb95_for_promotion=args.min_tag_lb95_for_promotion,
            )
        except Exception as exc:
            print(f"Failed milestone {episode:,} ({checkpoint_path}): {exc}")


if __name__ == "__main__":
    main()
