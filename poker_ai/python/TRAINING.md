# NFSP Training Procedure

## Overview

Training follows a 3-stage pipeline: heads-up (2p) → 6-max (6p) → full ring (9p).
Each stage fine-tunes from the previous checkpoint. The network architecture is
player-count agnostic (fixed 462-dim observation), so weights transfer directly.

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
| `--eta-start` | 0.01 | Anticipatory param start — almost all BR early for strongest best response |
| `--eta-end` | 0.4 | Ramps up AS mixing as training matures |
| `--eta-ramp-steps` | 200000000 | Linear ramp over ~25M episodes (200M env steps) |
| `--as-lr` | 0.0001 | Low AS learning rate prevents oscillation in the average strategy |
| `--br-lr` | 0.0001 | Best response learning rate |
| `--br-buffer-size` | 2000000 | ~250k hands of recent experience; ~30x batch size for diverse sampling |
| `--as-buffer-size` | 3000000 | Large reservoir preserves long-run average, prevents catastrophic forgetting |
| `--huber-delta` | 10.0 | Huber loss beta — squared error for <10 BB, linear above |
| `--batch-size` | 8192 | GPU-saturating batch for async training |
| `--lr-min-factor` | 0.01 | Cosine LR decays to 1% of initial (1e-4 → 1e-6) |
| `--lr-warmup-steps` | 8000000 | Linear warmup over ~1M episodes (8M env steps) |
| `--tau` | 0.005 | Polyak soft target update (every round, replacing hard copy) |
| `--epsilon-start` | 0.10 | Enough exploration while Q-values are still random |
| `--epsilon-end` | 0.003 | Near-deterministic late in training |
| `--epsilon-decay-steps` | 1200000000 | Explore across full training horizon |
| `--br-train-steps` | 24 | BR gradient steps per self-play round — more BR updates = stronger exploits |
| `--as-train-steps` | 12 | AS gradient steps per self-play round — maintains 2:1 ratio with BR |

**Note on train steps:** BR quality directly drives AS quality in NFSP. Fewer BR
updates mean weaker counter-strategies, which leads to a weaker average strategy.
Use at least 8/4 (BR/AS); 24/12 is recommended if your GPU can keep up.

### Async Training Parameters

| Parameter | Value | Why |
|---|---|---|
| `--async` | flag | Concurrent self-play + training threads for ~2-5x throughput |
| `--train-ahead` | 4 | Max training rounds ahead of self-play before sleeping |
| `--sync-every` | 15 | Sync training weights → inference copies every N rounds |
| `--num-envs` | 2048 | Larger inference batches for better GPU utilization |
| `--save-buffers` | flag | Save replay buffers alongside checkpoints (~11GB) |

### Resume & Warm Restart Parameters

| Parameter | Value | Why |
|---|---|---|
| `--freeze-as` | flag | Freeze AS weights permanently on resume |
| `--as-freeze-duration` | 5000000 | Episodes to freeze AS before unfreezing (buffer diversity) |
| `--as-warmup-episodes` | 2000000 | AS LR warmup after unfreeze (ramp from 1% → 100%) |
| `--restart-schedules` | flag | Warm restart: reset LR/epsilon/eta to start values on resume |
| `--reset-optimizers` | flag | Clear Adam momentum on resume (use with `--restart-schedules`) |

## Stage 1: Heads-Up (2 players)

Train from scratch. This builds the foundation: hand values, bet sizing, aggression.

```bash
python scripts/train.py \
  --num-players 2 \
  --device cuda --async --train-ahead 4 --sync-every 15 \
  --num-envs 2048 \
  --batch-size 8192 \
  --eta-start 0.01 \
  --eta-end 0.4 \
  --eta-ramp-steps 200000000 \
  --br-lr 0.0001 --br-train-steps 24 \
  --as-lr 0.0001 --as-train-steps 12 \
  --br-buffer-size 2000000 \
  --as-buffer-size 3000000 \
  --epsilon-start 0.10 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 1200000000 \
  --huber-delta 10.0 \
  --lr-warmup-steps 8000000 \
  --lr-min-factor 0.01 \
  --tau 0.005 \
  --eval-every 1000000 \
  --eval-hands 5000 \
  --checkpoint-every 5000000 \
  --save-buffers \
  --episodes 300000000 \
  --checkpoint-dir checkpoints/hu \
  --log-dir logs/hu
```

### Resuming Heads-Up Training

**With saved buffers** (`--save-buffers` was used): buffers are loaded automatically
on resume. No freeze/warmup needed — training continues seamlessly.

```bash
python scripts/train.py \
  --resume checkpoints/hu/checkpoint_latest.pt \
  --save-buffers \
  --device cuda --async --train-ahead 4 --sync-every 15 \
  --num-envs 2048 \
  --batch-size 8192 \
  --br-lr 0.0001 --br-train-steps 24 \
  --as-lr 0.0001 --as-train-steps 12 \
  --episodes 300000000 --eval-every 1000000 \
  --eval-hands 5000 \
  --checkpoint-dir checkpoints/hu --checkpoint-every 5000000 \
  --log-dir logs/hu
```

**Without saved buffers:** use `--as-freeze-duration` and `--as-warmup-episodes`.
The AS reservoir buffer fills with only the current BR policy's actions instead of
the historical average. Training AS immediately on this narrow data destroys the
average strategy.

- `--as-freeze-duration N` freezes AS for N episodes while the buffer accumulates
  diverse data from evolving BR policies.
- `--as-warmup-episodes M` ramps the AS learning rate from 1% → 100% over M episodes
  after unfreezing, preventing catastrophic overwriting of the averaged strategy.
- The AS optimizer is automatically reset when the freeze ends (discards stale Adam
  momentum that would cause wild updates on the new buffer data).

A good value is ~5M episodes for both freeze and warmup — the 3M-sample AS buffer
turns over several times in that period, accumulating data from many different BR
policies as BR evolves during training.

```bash
python scripts/train.py \
  --resume checkpoints/hu/checkpoint_latest.pt \
  --as-freeze-duration 5000000 \
  --as-warmup-episodes 5000000 \
  --save-buffers \
  --device cuda --async --train-ahead 4 --sync-every 15 \
  --num-envs 2048 \
  --batch-size 8192 \
  --br-lr 0.0001 --br-train-steps 24 \
  --as-lr 0.0001 --as-train-steps 12 \
  --episodes 300000000 --eval-every 1000000 \
  --eval-hands 5000 \
  --checkpoint-dir checkpoints/hu --checkpoint-every 5000000 \
  --log-dir logs/hu
```

### Warm Restart

Use `--restart-schedules` to reset LR, epsilon, and eta schedules back to their
start values while keeping trained model weights and buffers. This is useful when
a run has converged (LR near minimum, epsilon near zero) but you want to continue
training with renewed exploration and a fresh learning rate cycle.

This implements **cosine annealing with warm restarts** — a well-established technique
for escaping local minima and finding better solutions.

- Schedules restart from step 0: epsilon ramps down again, LR does warmup + cosine
  decay, eta ramps from start to end.
- Model weights are fully preserved — only the schedule counters reset.
- Use `--reset-optimizers` to also clear Adam momentum/variance state for a cleaner
  restart (recommended).
- You can override schedule parameters (e.g. shorter `--epsilon-decay-steps`) for
  the restart cycle.
- The schedule offset is saved in checkpoints, so subsequent resumes continue from
  the correct position.

```bash
python scripts/train.py \
  --resume checkpoints/hu/checkpoint_latest.pt \
  --restart-schedules --reset-optimizers \
  --save-buffers \
  --device cuda --async --train-ahead 4 --sync-every 15 \
  --num-envs 2048 \
  --batch-size 8192 \
  --eta-start 0.01 --eta-end 0.4 --eta-ramp-steps 200000000 \
  --epsilon-start 0.05 --epsilon-end 0.003 --epsilon-decay-steps 600000000 \
  --lr-warmup-steps 4000000 --lr-min-factor 0.01 \
  --br-lr 0.0001 --br-train-steps 24 \
  --as-lr 0.0001 --as-train-steps 12 \
  --episodes 500000000 --eval-every 1000000 \
  --eval-hands 5000 \
  --checkpoint-dir checkpoints/hu_restart \
  --checkpoint-every 5000000 \
  --log-dir logs/hu_restart
```

**Tip:** Use a lower `--epsilon-start` (e.g. 0.05) on restarts since the network
already has reasonable Q-values — it doesn't need as much random exploration as
training from scratch.

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
  --device cuda --async --train-ahead 4 --sync-every 15 \
  --num-envs 512 \
  --batch-size 8192 \
  --eta-start 0.1 \
  --eta-end 0.4 \
  --eta-ramp-steps 100000000 \
  --br-lr 0.0001 --br-train-steps 24 \
  --as-lr 0.0001 --as-train-steps 12 \
  --br-buffer-size 2000000 \
  --as-buffer-size 3000000 \
  --epsilon-start 0.04 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 100000000 \
  --lr-warmup-steps 2000000 \
  --lr-min-factor 0.01 \
  --tau 0.005 \
  --eval-every 2500000 \
  --eval-hands 3000 \
  --checkpoint-every 5000000 \
  --save-buffers \
  --episodes 50000000 \
  --checkpoint-dir checkpoints/6max \
  --log-dir logs/6max \
  --resume checkpoints/hu/checkpoint_best.pt
```

**Key differences from heads-up:**
- `--num-envs 512`: hands have more steps (more players acting), so fewer envs
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
  --device cuda --async --train-ahead 4 --sync-every 15 \
  --num-envs 256 \
  --batch-size 8192 \
  --eta-start 0.1 \
  --eta-end 0.4 \
  --eta-ramp-steps 60000000 \
  --br-lr 0.0001 --br-train-steps 24 \
  --as-lr 0.0001 --as-train-steps 12 \
  --br-buffer-size 2000000 \
  --as-buffer-size 3000000 \
  --epsilon-start 0.03 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 60000000 \
  --lr-warmup-steps 2000000 \
  --lr-min-factor 0.01 \
  --tau 0.005 \
  --eval-every 2500000 \
  --eval-hands 2000 \
  --checkpoint-every 5000000 \
  --save-buffers \
  --episodes 30000000 \
  --checkpoint-dir checkpoints/9ring \
  --log-dir logs/9ring \
  --resume checkpoints/6max/checkpoint_best.pt
```

**Key differences:**
- `--num-envs 256`: 9-player hands are long, fewer envs needed
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
| **Epsilon schedule** | 0.06 → 0.003 over 400M steps | 0.10 → 0.003 over 400M steps | Higher start — Q-values are random early, need more exploration |
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
| **Observation** | 569 floats, 25 game state features | 462 floats, 46 game state features | Pot odds, SPR, street counts, aggressor tracking |
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
| **br_train_steps** | 12 | 24 | More gradient steps to saturate GPU in async mode |
| **as_train_steps** | 6 | 12 | Matches BR steps proportionally |

## ONNX Export

After each stage, export the AS network for use in the backend:

```bash
python scripts/export_onnx.py \
  checkpoints/hu/checkpoint_latest.pt \
  --output ../../backend/models/poker_as_hu.onnx \
  --verify

python scripts/export_onnx.py \
  checkpoints/6max/checkpoint_latest.pt \
  --output ../../backend/models/poker_as_6max.onnx \
  --verify

python scripts/export_onnx.py \
  checkpoints/9ring/checkpoint_latest.pt \
  --output ../../backend/models/poker_as_9ring.onnx \
  --verify
```

The `--verify` flag checks that the ONNX model output matches the PyTorch model
within tolerance (1e-5). The export wrapper pre-computes position indices as
constant tensors so that the Rust backend (tract-onnx) can optimize the
transformer's multi-head attention without symbolic dimension issues.

**Note:** The sequence length axis is fixed at export time (default: 30). The
batch dimension remains dynamic but the backend always uses batch=1.

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
- **Eval runs on CPU** in a background thread, so GPU training continues at full
  speed during evaluation
- **Resuming interrupted training:** use `--resume checkpoints/hu/checkpoint_latest.pt`
  with the same arguments to continue where you left off. With `--save-buffers`,
  buffers load automatically and training resumes seamlessly.
- **Without saved buffers on resume:** use `--as-freeze-duration` and
  `--as-warmup-episodes` to protect AS from collapsing on narrow post-resume data.
- **Warm restart** (`--restart-schedules`): resets LR/epsilon/eta schedules to start
  values while keeping model weights. Use `--reset-optimizers` to also clear Adam
  state. Great for continued training after a run has converged.
- The reward signal is normalized to big blinds, so models transfer across blind levels
- **RAM usage:** the 3M AS reservoir uses ~7GB, 2M BR circular uses ~12GB (~19GB total).
- **Buffer persistence** (`--save-buffers`): saves BR and AS buffers as compressed
  `.npz` files (~11GB total) alongside the latest checkpoint. On resume, buffers are
  loaded automatically if found in the checkpoint directory.
- **Train steps matter:** BR quality drives AS quality in NFSP. Use at least 8/4
  (BR/AS) train steps per round. With 24/12, training is heavier per round but
  produces stronger counter-strategies and better convergence.