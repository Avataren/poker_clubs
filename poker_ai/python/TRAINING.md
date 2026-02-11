# NFSP Training Procedure

## Overview

Training follows a 3-stage pipeline: heads-up (2p) → 6-max (6p) → full ring (9p).
Each stage fine-tunes from the previous checkpoint. The network architecture is
player-count agnostic (fixed 569-dim observation), so weights transfer directly.

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
| `--eta` | 0.1 | Anticipatory param — lower values give smoother average strategy |
| `--as-lr` | 0.0001 | Low AS learning rate prevents oscillation in the average strategy |
| `--br-lr` | 0.0001 | Best response learning rate |
| `--as-buffer-size` | 5000000 | Large reservoir preserves long-run average, prevents catastrophic forgetting |
| `--batch-size` | 4096 | Fills 24GB VRAM well, stable gradient estimates |
| `--lr-min-factor` | 0.01 | Cosine LR decays to 1% of initial (1e-4 → 1e-6) |
| `--lr-warmup-steps` | 500000 | Linear warmup over first 500k env steps |
| `--tau` | 0.005 | Polyak soft target update (every round, replacing hard copy) |

## Stage 1: Heads-Up (2 players)

Train from scratch. This builds the foundation: hand values, bet sizing, aggression.

```bash
python scripts/train.py \
  --num-players 2 \
  --device cuda \
  --num-envs 512 \
  --batch-size 4096 \
  --eta 0.1 \
  --br-lr 0.0001 \
  --as-lr 0.0001 \
  --as-buffer-size 5000000 \
  --br-train-steps 8 \
  --as-train-steps 4 \
  --epsilon-start 0.06 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 300000000 \
  --lr-warmup-steps 500000 \
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
- Monitor in TensorBoard: `meta/lr_factor`, `meta/br_lr`

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
  --eta 0.1 \
  --br-lr 0.0001 \
  --as-lr 0.0001 \
  --as-buffer-size 5000000 \
  --br-train-steps 8 \
  --as-train-steps 4 \
  --epsilon-start 0.04 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 200000000 \
  --lr-warmup-steps 200000 \
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
- `--lr-warmup-steps 200000`: shorter warmup for fine-tuning

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
  --eta 0.1 \
  --br-lr 0.0001 \
  --as-lr 0.0001 \
  --as-buffer-size 5000000 \
  --br-train-steps 8 \
  --as-train-steps 4 \
  --epsilon-start 0.03 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 150000000 \
  --lr-warmup-steps 200000 \
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
- `--epsilon-start 0.03`: even less exploration needed
- `--episodes 30000000`: fine-tuning pass
- `--eval-hands 2000`: 9-player eval hands are slow

## What Changed (v2)

Key improvements over the initial training setup:

| Change | Before | After | Why |
|---|---|---|---|
| **LR schedule** | Constant 1e-4 | Cosine decay 1e-4 → 1e-6 with warmup | Prevents policy oscillation after convergence |
| **Target updates** | Hard copy every 300 rounds | Polyak soft (tau=0.005) every round | Smoother Q-value targets, less instability |
| **eta** | 0.2 | 0.1 | More AS play → better average strategy quality |
| **br_train_steps** | 12 | 8 | Less overfitting to recent BR buffer data |
| **as_train_steps** | 6 | 4 | Matches reduced BR steps proportionally |

The LR schedule is the most impactful change. Previous runs showed convergence
to ~-4 bb/100 vs TAG by episode 4-5M with no improvement through 32M — the
constant LR caused the policy to oscillate around the basin instead of settling.

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
- **Resuming interrupted training:** use `--resume checkpoints/6max/checkpoint_latest.pt`
  with the same arguments to continue where you left off
- The reward signal is normalized to big blinds, so models transfer across blind levels
- **RAM usage:** the 5M reservoir buffer uses ~13GB RAM. With 31GB system RAM this
  leaves plenty for the OS and Rust engine. Don't go above 5M without more RAM.
