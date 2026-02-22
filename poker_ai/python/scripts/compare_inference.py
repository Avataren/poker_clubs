#!/usr/bin/env python3
"""Compare PyTorch model inference vs ONNX inference on real game sequences.

Plays a deterministic "shoe" of hands with a fixed seed, collecting observations
from the training engine at each decision point. Both the PyTorch model and the
ONNX model (via onnxruntime) run on the exact same inputs. Any divergence
indicates a bug in the ONNX export or in tract-onnx's execution.

If PyTorch == ONNX: the ONNX pipeline is faithful; high all-in rates etc. are
genuine model behavior (training convergence issue).
If PyTorch != ONNX: there is an export or numerical precision bug to fix.

Usage:
    cd /home/avataren/src/poker/poker_ai/python
    source .venv/bin/activate
    python scripts/compare_inference.py \\
        --checkpoint checkpoints/6max_v2/checkpoint_latest.pt \\
        [--hands 500] [--seed 42] [--players 6] [--tol 1e-4]
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.env.poker_env import PokerEnv
from poker_ai.export.onnx_export import (
    AverageStrategyONNXWrapper,
    _infer_static_feature_size,
    infer_config_from_checkpoint,
)
from poker_ai.model.network import AverageStrategyNet
from poker_ai.model.state_encoder import GAME_STATE_START, GAME_STATE_END
from poker_ai.training.self_play import extract_static_features_batch, pad_action_history

ACTION_NAMES = ["Fold", "Call", "R25%", "R40%", "R60%", "R80%", "RPot", "R150%", "AllIn"]
PHASE_NAMES = ["PreFlop", "Flop", "Turn", "River", "Showdown", "Over"]
# Offset of phase one-hot inside the 582-dim static obs
_PHASE_START = GAME_STATE_START           # 364
_PHASE_END   = GAME_STATE_START + 6       # 370


def _export_temp_onnx(as_net: AverageStrategyNet, config: NFSPConfig) -> str:
    """Export AS model to a temp ONNX file and return its path."""
    wrapper = AverageStrategyONNXWrapper(as_net, max_seq_len=config.max_history_len)
    wrapper.eval()

    obs_d     = torch.randn(1, config.static_feature_size)
    hist_d    = torch.randn(1, config.max_history_len, config.history_input_dim)
    lens_d    = torch.tensor([config.max_history_len], dtype=torch.long)
    mask_d    = torch.ones(1, config.num_actions, dtype=torch.bool)

    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    torch.onnx.export(
        wrapper,
        (obs_d, hist_d, lens_d, mask_d),
        path,
        input_names=["obs", "action_history", "history_lengths", "legal_mask"],
        output_names=["action_probs"],
        dynamic_axes={k: {0: "batch"} for k in
                      ["obs", "action_history", "history_lengths", "legal_mask", "action_probs"]},
        opset_version=17,
        do_constant_folding=True,
    )
    return path


def run_comparison(
    checkpoint_path: str,
    num_hands: int = 500,
    seed: int = 42,
    tolerance: float = 1e-4,
    num_players: int = 6,
) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        return False

    # ── Load checkpoint ─────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = infer_config_from_checkpoint(ckpt, NFSPConfig())
    config.static_feature_size = _infer_static_feature_size(ckpt, config)

    as_net = AverageStrategyNet(config)
    as_net.load_state_dict(ckpt["as_net"])
    as_net.eval()

    # ── Export and load ONNX ─────────────────────────────────────────────────
    print("Exporting fresh ONNX from checkpoint …")
    onnx_path = _export_temp_onnx(as_net, config)
    session = ort.InferenceSession(onnx_path)
    print(f"ONNX session ready  (static_feature_size={config.static_feature_size})")

    # ── Play deterministic hands ─────────────────────────────────────────────
    env = PokerEnv(
        num_players=num_players,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        seed=seed,
    )

    diffs: list[float] = []
    action_disagree = 0
    phase_diffs: dict[int, list[float]] = {i: [] for i in range(6)}
    worst: list[tuple] = []          # (diff, phase_idx, pt_p, onnx_p, mask)

    # Per-phase action frequency counters
    phase_actions_pt   = {i: np.zeros(9) for i in range(4)}
    phase_actions_onnx = {i: np.zeros(9) for i in range(4)}
    phase_counts       = {i: 0 for i in range(4)}

    player, obs, mask = env.reset()
    hand_count = 0
    decision_count = 0

    print(f"Playing {num_hands} hands ({num_players}-player, seed={seed}) …\n")

    while hand_count < num_hands:
        # Training-engine obs is 710 floats; extract 582-dim static slice
        static_obs = extract_static_features_batch(obs[np.newaxis])[0]  # (582,)

        history = env.get_action_history()
        padded_hist, hist_len = pad_action_history(history, max_len=config.max_history_len)

        # ── PyTorch forward ──────────────────────────────────────────────────
        obs_t  = torch.tensor(static_obs,   dtype=torch.float32).unsqueeze(0)
        hist_t = torch.tensor(padded_hist,  dtype=torch.float32).unsqueeze(0)
        lens_t = torch.tensor([hist_len],   dtype=torch.long)
        mask_t = torch.tensor(mask,         dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            pt_p = as_net(obs_t, hist_t, lens_t, mask_t).squeeze(0).numpy()

        # ── ONNX forward ─────────────────────────────────────────────────────
        onnx_p = session.run(None, {
            "obs":             static_obs[np.newaxis].astype(np.float32),
            "action_history":  padded_hist[np.newaxis].astype(np.float32),
            "history_lengths": np.array([hist_len], dtype=np.int64),
            "legal_mask":      mask[np.newaxis],
        })[0][0]

        # ── Record diff ───────────────────────────────────────────────────────
        diff = float(np.abs(pt_p - onnx_p).max())
        diffs.append(diff)

        phase_idx = int(np.argmax(static_obs[_PHASE_START:_PHASE_END]))
        phase_diffs[phase_idx].append(diff)

        if diff > 1e-3:
            worst.append((diff, phase_idx, pt_p.copy(), onnx_p.copy(), mask.copy()))

        # ── Action choices ────────────────────────────────────────────────────
        pt_action   = int(np.argmax(np.where(mask, pt_p,   -1.0)))
        onnx_action = int(np.argmax(np.where(mask, onnx_p, -1.0)))

        if pt_action != onnx_action:
            action_disagree += 1

        if phase_idx < 4:
            phase_actions_pt  [phase_idx][pt_action]   += 1
            phase_actions_onnx[phase_idx][onnx_action] += 1
            phase_counts[phase_idx] += 1

        decision_count += 1

        # Use PT greedy action to advance the game (deterministic trajectory)
        player, obs, mask, rewards, done = env.step(pt_action)
        if done:
            hand_count += 1
            if hand_count % 100 == 0:
                print(f"  … {hand_count}/{num_hands} hands, {decision_count} decisions")
            if hand_count < num_hands:
                player, obs, mask = env.reset()

    os.unlink(onnx_path)

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print("  INFERENCE PARITY REPORT")
    print(sep)
    print(f"  Hands: {hand_count}   Decisions: {decision_count}")
    print(f"  Players: {num_players}   Seed: {seed}")
    print()

    print("PyTorch vs ONNX probability difference (L-inf per decision):")
    print(f"  Max  : {max(diffs):.2e}")
    print(f"  Mean : {np.mean(diffs):.2e}")
    print(f"  p95  : {np.percentile(diffs, 95):.2e}")
    print(f"  p99  : {np.percentile(diffs, 99):.2e}")
    print(f"  >1e-4: {sum(d > 1e-4 for d in diffs):5d}  ({sum(d > 1e-4 for d in diffs)/decision_count*100:.1f}%)")
    print(f"  >1e-3: {sum(d > 1e-3 for d in diffs):5d}  ({sum(d > 1e-3 for d in diffs)/decision_count*100:.1f}%)")
    print(f"  Greedy action disagrees: {action_disagree}/{decision_count} "
          f"({action_disagree/decision_count*100:.1f}%)")
    print()

    print("Per-phase max diff:")
    for pi, pname in enumerate(PHASE_NAMES[:4]):
        pd = phase_diffs[pi]
        if pd:
            print(f"  {pname:8s}: max={max(pd):.2e}  mean={np.mean(pd):.2e}  n={len(pd)}")
    print()

    # Action frequency comparison (PyTorch greedy)
    print("Greedy action frequencies — PyTorch vs ONNX (should match if diff≈0):")
    for pi, pname in enumerate(PHASE_NAMES[:4]):
        n = phase_counts[pi]
        if n == 0:
            continue
        print(f"\n  {pname} ({n} decisions):")
        header = f"    {'Action':8s}  {'PyTorch':>8s}  {'ONNX':>8s}  {'Delta':>8s}"
        print(header)
        for ai, aname in enumerate(ACTION_NAMES):
            pp = phase_actions_pt  [pi][ai] / n * 100
            op = phase_actions_onnx[pi][ai] / n * 100
            if pp > 0.05 or op > 0.05:
                print(f"    {aname:8s}  {pp:7.2f}%  {op:7.2f}%  {op-pp:+7.2f}%")

    if worst:
        worst.sort(reverse=True)
        print(f"\nTop {min(5, len(worst))} largest disagreements:")
        for diff, pidx, pt_p, onnx_p, mk in worst[:5]:
            phase_name = PHASE_NAMES[pidx] if pidx < len(PHASE_NAMES) else "?"
            print(f"\n  diff={diff:.4f}  phase={phase_name}")
            print(f"  {'Action':8s} {'Legal':>6} {'PT':>8} {'ONNX':>8} {'Delta':>8}")
            for ai, aname in enumerate(ACTION_NAMES):
                if mk[ai] or pt_p[ai] > 0.001 or onnx_p[ai] > 0.001:
                    legal = "Y" if mk[ai] else "n"
                    print(f"  {aname:8s} {legal:>6} {pt_p[ai]*100:7.2f}% "
                          f"{onnx_p[ai]*100:7.2f}%  {(onnx_p[ai]-pt_p[ai])*100:+7.2f}%")

    ok = max(diffs) < tolerance
    print(f"\n{'PASS' if ok else 'FAIL'} — max diff {max(diffs):.2e} "
          f"{'<' if ok else '>='} tolerance {tolerance:.0e}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs ONNX inference on deterministic game sequences"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="checkpoints/6max_v2/checkpoint_latest.pt",
        help="Path to .pt checkpoint (default: checkpoints/6max_v2/checkpoint_latest.pt)",
    )
    parser.add_argument("--hands",   type=int,   default=500,  help="Hands to play (default: 500)")
    parser.add_argument("--seed",    type=int,   default=42,   help="RNG seed (default: 42)")
    parser.add_argument("--players", type=int,   default=6,    help="Number of players (default: 6)")
    parser.add_argument("--tol",     type=float, default=1e-4, help="Pass tolerance (default: 1e-4)")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ok = run_comparison(
        checkpoint_path=args.checkpoint,
        num_hands=args.hands,
        seed=args.seed,
        tolerance=args.tol,
        num_players=args.players,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
