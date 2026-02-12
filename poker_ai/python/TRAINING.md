# NFSP Training Procedure

## Overview

Training follows a 3-stage pipeline: heads-up (2p) → 6-max (6p) → full ring (9p).
Each stage fine-tunes from the previous checkpoint. The network architecture is
player-count agnostic (fixed 590-dim observation), so weights transfer directly.

**Important:** v4 fixed critical transition bugs — old checkpoints are incompatible.
Start Stage 1 from scratch.

## Prerequisites

```bash
cd /home/avataren/src/poker/poker_ai/python
source .venv/bin/activate
# Rebuild engine if needed:
cd ../engine && maturin develop --release && cd ../python
```

## Key Hyperparameters

These stabilization parameters are critical for convergence and should be passed
to all training stages:

| Parameter | Value | Why |
|---|---|---|
| `--eta-start` | 0.1 | Anticipatory param start — mostly BR early for stronger best response |
| `--eta-end` | 0.4 | Ramps up AS mixing as training matures |
| `--eta-ramp-steps` | 200000000 | Linear ramp over ~25M episodes (200M env steps) |
| `--as-lr` | 0.0001 | Low AS learning rate prevents oscillation in the average strategy |
| `--br-lr` | 0.0001 | Best response learning rate |
| `--br-buffer-size` | 1000000 | ~125k hands of recent experience; keeps RAM in check |
| `--as-buffer-size` | 4000000 | Large reservoir preserves long-run average, prevents catastrophic forgetting |
| `--huber-delta` | 10.0 | Huber loss beta — squared error for <10 BB, linear above |
| `--batch-size` | 4096 | Fills 24GB VRAM well, stable gradient estimates |
| `--lr-min-factor` | 0.01 | Cosine LR decays to 1% of initial (1e-4 → 1e-6) |
| `--lr-warmup-steps` | 4000000 | Linear warmup over ~500k episodes (4M env steps) |
| `--tau` | 0.005 | Polyak soft target update (every round, replacing hard copy) |
| `--epsilon-start` | 0.10 | Enough exploration while Q-values are still random |
| `--epsilon-end` | 0.003 | Near-deterministic late in training |
| `--epsilon-decay-steps` | 200000000 | Explore first ~25M episodes, exploit remaining ~75M |

## Stage 1: Heads-Up (2 players)

Train from scratch. This builds the foundation: hand values, bet sizing, aggression.

```bash
python scripts/train.py \
  --num-players 2 \
  --device cuda \
  --num-envs 1024 \
  --batch-size 4096 \
  --eta-start 0.1 \
  --eta-end 0.4 \
  --eta-ramp-steps 200000000 \
  --br-lr 0.0001 \
  --as-lr 0.0001 \
  --br-buffer-size 1000000 \
  --as-buffer-size 4000000 \
  --br-train-steps 8 \
  --as-train-steps 4 \
  --epsilon-start 0.10 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 200000000 \
  --huber-delta 10.0 \
  --lr-warmup-steps 4000000 \
  --lr-min-factor 0.01 \
  --tau 0.005 \
  --eval-every 200000 \
  --eval-hands 5000 \
  --checkpoint-every 200000 \
  --episodes 100000000 \
  --checkpoint-dir checkpoints/hu \
  --log-dir logs/hu
```

**What to watch:**
- `vs Caller` should go positive within the first few million episodes
- `vs TAG` trending toward zero or positive
- `vs Random` solidly positive
- `lr` factor in logs — should decay smoothly from 1.0 to 0.01 over the run
- Monitor in TensorBoard: `meta/epsilon`, `meta/eta`, `meta/lr_factor`

**When to stop:** When eval metrics plateau for several million episodes, or at the
episode limit. Pick the best checkpoint using milestone eval:

```bash
./run_eval.sh  # runs in a separate terminal, evaluates milestones
```

## Stage 2: 6-Max (6 players)

Fine-tune from the best heads-up checkpoint. The model already knows poker
fundamentals — it needs to learn multi-way dynamics: tighter ranges, position
importance, pot odds with multiple callers.

```bash
python scripts/train.py \
  --num-players 6 \
  --device cuda \
  --num-envs 256 \
  --batch-size 4096 \
  --eta-start 0.1 \
  --eta-end 0.4 \
  --eta-ramp-steps 100000000 \
  --br-lr 0.0001 \
  --as-lr 0.0001 \
  --br-buffer-size 1000000 \
  --as-buffer-size 4000000 \
  --br-train-steps 8 \
  --as-train-steps 4 \
  --epsilon-start 0.04 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 100000000 \
  --lr-warmup-steps 2000000 \
  --lr-min-factor 0.01 \
  --tau 0.005 \
  --eval-every 200000 \
  --eval-hands 3000 \
  --checkpoint-every 200000 \
  --episodes 50000000 \
  --checkpoint-dir checkpoints/6max \
  --log-dir logs/6max \
  --resume checkpoints/hu/checkpoint_best.pt
```

**Key differences from heads-up:**
- `--num-envs 256`: hands have more steps (more players acting), so fewer envs
  needed to keep GPU busy
- `--epsilon-start 0.04`: lower starting exploration since the model already
  understands basic poker
- `--episodes 50000000`: fewer total episodes needed (fine-tuning, not from scratch)
- `--eval-hands 3000`: each hand takes longer with 6 players
- `--lr-warmup-steps 2000000`: shorter warmup for fine-tuning

**What to watch:**
- The model may initially regress (heads-up habits don't all transfer)
- Should recover and improve within a few million episodes
- Position-dependent play developing (tighter UTG, looser on button)

## Stage 3: Full Ring (9 players)

Fine-tune from the best 6-max checkpoint. Jump from 6 to 9 is smaller than
2 to 6 — the model already knows multi-way play.

```bash
python scripts/train.py \
  --num-players 9 \
  --device cuda \
  --num-envs 128 \
  --batch-size 4096 \
  --eta-start 0.1 \
  --eta-end 0.4 \
  --eta-ramp-steps 60000000 \
  --br-lr 0.0001 \
  --as-lr 0.0001 \
  --br-buffer-size 1000000 \
  --as-buffer-size 4000000 \
  --br-train-steps 8 \
  --as-train-steps 4 \
  --epsilon-start 0.03 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 60000000 \
  --lr-warmup-steps 2000000 \
  --lr-min-factor 0.01 \
  --tau 0.005 \
  --eval-every 200000 \
  --eval-hands 2000 \
  --checkpoint-every 200000 \
  --episodes 30000000 \
  --checkpoint-dir checkpoints/9ring \
  --log-dir logs/9ring \
  --resume checkpoints/6max/checkpoint_best.pt
```

**Key differences:**
- `--num-envs 128`: 9-player hands are long, fewer envs needed
- `--epsilon-start 0.03`: minimal exploration for fine-tuning
- `--episodes 30000000`: fine-tuning pass
- `--eval-hands 2000`: 9-player eval hands are slow

## What Changed (v4)

Critical convergence fixes:

| Change | Before | After | Why |
|---|---|---|---|
| **BR transitions** | One per env step; next_obs from wrong player | Per-player tracking; same-player next_obs | Q-values were bootstrapping from opponent's cards — couldn't converge |
| **Terminal rewards** | Only last-to-act player got reward | All players get terminal transitions | Most players' rewards were lost; ~93% of training signal missing |
| **Card augmentation** | Independent random perm for obs/next_obs | Same permutation for both | Temporal consistency in DQN transitions |
| **Default players** | 6 | 2 | Matches eval; NFSP convergence guarantees are for 2-player |
| **Epsilon schedule** | 0.06 → 0.003 over 400M steps | 0.10 → 0.003 over 200M steps | More exploration early (Q-values start random), faster decay |
| **AS buffer** | 5M | 4M | Saves ~3GB RAM; 4M still preserves long-run average well |
| **step_batch mask** | `.take(8)` (wrong) | `.take(9)` | Off-by-one in non-dense API mask padding |

**Old checkpoints are incompatible** — they trained on corrupted transitions.
Start fresh from Stage 1.

## What Changed (v3)

Key improvements over v2:

| Change | Before | After | Why |
|---|---|---|---|
| **Action space** | 8 actions (overlapping raises) | 9 actions (0.25×–1.5× pot) | Better strategic coverage, cleaner pot-relative sizing |
| **History encoding** | 7-dim (coarse 5-cat one-hot) | 11-dim (9-action one-hot + bet/pot ratio) | Richer action history for pattern recognition |
| **Observation** | 569 floats, 25 game state features | 590 floats, 46 game state features | Pot odds, SPR, street counts, aggressor tracking |
| **Head network** | 256-dim heads | 512-dim heads | More capacity for value/policy heads |
| **Legal mask** | `logits.clamp(min=-1e9)` | `torch.where(mask, logits, -1e9)` | Proper masking — clamp affected legal actions too |
| **Epsilon** | 0.12 → 0.003 over 20M | 0.10 → 0.003 over 200M steps | Sufficient exploration for 9 actions, less BR noise |
| **Eta** | Fixed 0.1 | Linear ramp 0.1 → 0.4 over 200M steps | Mostly BR early (stronger best response), more AS later |

Previous v2 changes (still in effect):

| Change | Before | After | Why |
|---|---|---|---|
| **LR schedule** | Constant 1e-4 | Cosine decay 1e-4 → 1e-6 with warmup | Prevents policy oscillation after convergence |
| **Target updates** | Hard copy every 300 rounds | Polyak soft (tau=0.005) every round | Smoother Q-value targets, less instability |
| **Grad clipping** | None | `clip_grad_norm_ 10.0` on both BR and AS | Prevents gradient spikes from large Q-value errors |
| **br_train_steps** | 12 | 8 | Less overfitting to recent BR buffer data |
| **as_train_steps** | 6 | 4 | Matches reduced BR steps proportionally |

## ONNX Export

After each stage, export the AS network for use in the backend:

```bash
python scripts/export_onnx.py \
  checkpoints/hu/checkpoint_best.pt \
  --output models/poker_as_hu.onnx

python scripts/export_onnx.py \
  checkpoints/6max/checkpoint_best.pt \
  --output models/poker_as_6max.onnx

python scripts/export_onnx.py \
  checkpoints/9ring/checkpoint_best.pt \
  --output models/poker_as_9ring.onnx
```

## Backend Integration

Set the model path for the backend bots:

```bash
export POKER_BOT_MODEL_ONNX=path/to/poker_as_6max.onnx
```

Or register bots with a specific model:

```
strategy = "model:path/to/poker_as_6max.onnx"
```

The backend normalizes all features by the current table's big blind automatically,
so the model works across different blind levels and tournaments.

## Tips

- **Separate checkpoint/log dirs per stage** to avoid overwriting previous work
- **Don't delete earlier checkpoints** — you may want to restart fine-tuning with
  different hyperparameters
- **Monitor with TensorBoard:** `tensorboard --logdir logs/`
- **Milestone eval** can run in parallel with training on the same GPU (it only
  does inference)
- **Resuming interrupted training:** use `--resume checkpoints/hu/checkpoint_latest.pt`
  with the same arguments to continue where you left off
- The reward signal is normalized to big blinds, so models transfer across blind levels
- **RAM usage:** the 4M AS reservoir uses ~10GB, 1M BR circular uses ~6GB (~16GB total).
  With 31GB system RAM this leaves plenty for the OS and Rust engine.
