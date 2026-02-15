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

## Algorithm Deep Dive

This section explains how the training algorithm works end-to-end, from self-play
to convergence. No math background required — just poker intuition.

### The Core Idea: NFSP (Neural Fictitious Self-Play)

The goal is to find a **Nash equilibrium** — a strategy that can't be exploited.
Against a Nash equilibrium strategy, no opponent can do better than break even in
the long run. This is the theoretical foundation of "GTO" (game-theory optimal)
poker.

NFSP achieves this by training two neural networks that play different roles:

1. **Best Response (BR) network** — the "exploiter." Given the opponent's current
   strategy, BR tries to find the single best counter-strategy. It asks: "if I
   know how my opponent plays, what's the most profitable way to play against them?"

2. **Average Strategy (AS) network** — the "averager." AS learns to mimic the
   average of all BR strategies seen throughout training. Over time, this average
   converges toward a Nash equilibrium.

The key insight: if you repeatedly find the best counter-strategy and average all
those counter-strategies together, the average converges to an unexploitable
strategy. This is the **fictitious play** algorithm, extended to work with neural
networks ("neural" fictitious self-play).

### Self-Play: Generating Training Data

Training data comes from the agent playing against itself. In each hand (episode):

1. Each player independently chooses a role for this hand:
   - With probability **eta**, use the AS network (average strategy)
   - With probability **(1 - eta)**, use the BR network (best response)

2. The hand plays out normally — dealing cards, betting rounds, showdown.

3. Actions are recorded into replay buffers for later training:
   - **All** actions go into the **AS buffer** (for supervised learning)
   - Actions from BR-policy players go into the **BR buffer** (for reinforcement learning)

**Eta scheduling:** Early in training, eta is low (0.01–0.1), so most play uses BR.
This gives BR maximum opportunity to explore and find strong counter-strategies.
As training progresses, eta ramps up to 0.4, mixing in more AS play. This is
important because AS is the actual output policy — it needs to generate training
data that reflects realistic play patterns.

**Epsilon exploration:** When a player uses BR, they take a random action with
probability epsilon (instead of their best action). This ensures BR explores
unusual situations and doesn't get stuck in local optima. Epsilon starts at 0.1
and decays to 0.003 over training.

**Batched environments:** For efficiency, 2048 hands are played simultaneously in
parallel. Each hand is independent — different cards, different positions, different
BR/AS choices. This fills the replay buffers quickly and keeps the GPU busy with
large inference batches.

### The Two Training Loops

After each batch of self-play, two separate training procedures run:

#### BR Training: Deep Q-Learning (DQN)

The BR network learns a **value function**: for each game state and action, it
estimates the expected profit in big blinds. This is reinforcement learning — the
network learns from rewards (winning or losing chips).

The training uses several standard DQN improvements:

- **Double DQN:** Two copies of the BR network exist — the "online" network and the
  "target" network. When computing the learning target, the online network picks
  the best action, but the target network evaluates how good that action is. This
  prevents a common problem where the network overestimates action values (it's
  grading its own homework — Double DQN fixes this by using a slightly older version
  of itself as the grader).

- **Dueling architecture:** The Q-value is split into two parts:
  - **V(s)** — how good is this game state overall? (value stream)
  - **A(s,a)** — how much better/worse is this specific action compared to average?
    (advantage stream)
  - Q(s,a) = V(s) + A(s,a) - mean(A)
  - This helps because in many poker states, the overall situation matters more than
    the specific action (e.g. with a royal flush, all bet sizes win — the value
    stream captures this without needing to learn it separately for each action).

- **Polyak target updates:** The target network is updated slowly — each training
  step blends 0.5% of the online network's weights into the target. This provides
  a stable learning signal (vs hard-copying every N steps, which causes sudden jumps).

- **Huber loss:** Instead of squared error (which can explode on large poker pots),
  the loss transitions from squared to linear for errors above 10 big blinds. This
  makes training robust to the occasional huge pot.

- **Card augmentation:** Each training batch randomly permutes the card suits and
  swaps hole card order (288 possible augmentations). This teaches the network that
  A♠K♥ is equivalent to A♦K♣ — dramatically reducing the effective state space.
  Crucially, the same permutation is applied to both the current and next state in
  each transition, preserving temporal consistency.

Each training round runs **br_train_steps** (e.g. 12) gradient updates, each on a
random batch of 8192 transitions sampled from the BR buffer.

#### AS Training: Supervised Learning

The AS network learns by **imitation** — it tries to predict what action was taken
in each game state. This is simple supervised learning with cross-entropy loss
(the same loss used in language models).

- Input: a game state (cards, betting history, stack sizes)
- Target: the action that was actually played (by BR or AS)
- Loss: cross-entropy between the network's predicted action probabilities and the
  actual action

The AS buffer uses **reservoir sampling** — a technique that maintains a uniform
random sample across the entire training history. Unlike the BR buffer (which only
keeps the most recent 2M transitions), the AS buffer keeps a representative sample
from all of training. This is essential: the average strategy must reflect the
entire history of play, not just recent play.

Card augmentation is also applied to AS training batches.

Each training round runs **as_train_steps** (e.g. 6) gradient updates on batches
from the AS buffer.

### Why BR Quality Drives Everything

This is the most important thing to understand about NFSP: **the AS network can
only be as good as the BR network forces it to be.**

The feedback loop works like this:

1. BR finds a weakness in the current AS strategy (e.g. "AS folds too much to
   river bets")
2. BR exploits this weakness, generating training data that shows profitable
   aggression on the river
3. AS learns from this data and adjusts — it starts calling river bets more often
4. BR now needs to find a new weakness, so it adapts (e.g. "AS now calls too much
   on the river — I should bluff less and value-bet more")
5. The cycle repeats, with AS gradually becoming harder to exploit

If BR is too weak (too few gradient updates), it can't find subtle exploits, and
AS never learns to defend against them. This is why `br_train_steps` matters so
much — more BR updates mean sharper exploits, which push AS toward a stronger
equilibrium.

### The Network Architecture

Both BR and AS share the same architecture — a shared trunk with separate heads:

```
Input: 462 static features + 256 history encoding = 718 dimensions

Static features (462):
  ├── Hole cards: 2 × 52 one-hot encodings (104)
  ├── Community cards: 5 × 52 one-hot, zero-padded pre-flop (260)
  ├── Game state: 46 floats (pot odds, stack-to-pot ratio, position,
  │   street counts, aggressor tracking, bet sizes, etc.)
  └── Hand strength: 52 floats (Monte Carlo win probability estimates)

History encoding (256): from transformer attention (see below)

Shared trunk:
  Linear(718 → 1024) → LayerNorm → ReLU → ResidualBlock(1024)
  Linear(1024 → 512) → LayerNorm → ReLU → ResidualBlock(512)

BR head (Dueling DQN):
  Value stream:     Linear(512) → ReLU → Linear(1)      → V(s)
  Advantage stream: Linear(512) → ReLU → Linear(9)      → A(s,a)
  Q(s,a) = V(s) + A(s,a) - mean(A)

AS head (Policy):
  Linear(512) → ReLU → Linear(9) → action logits
  Softmax over legal actions → action probabilities
```

Illegal actions are masked by adding -10,000 to their logits before softmax,
effectively zeroing their probability. The masking uses -10,000 instead of
negative infinity to avoid numerical overflow in float16 (mixed precision training).

### Transformer Attention Over Action History

A key component is the **action history transformer**, which reads the sequence of
actions taken so far in the hand and produces a 256-dimensional summary. This lets
the network recognize betting patterns — "opponent raised pre-flop, then checked
the flop" suggests a different range than "opponent limped pre-flop, then bet the
flop."

Each action in the history is encoded as an 11-dimensional record:

```
[seat_normalized, action_0, action_1, ..., action_8, bet_size_ratio]
  └── who acted     └── 9-dim one-hot of action type      └── bet/pot
```

The transformer processes up to 30 action records (enough for even complex
multi-street hands):

```
Action records (batch, 30, 11)
  → Linear projection (11 → 64)
  → Add learned positional embeddings
  → 2-layer Transformer Encoder (4 attention heads, 128-dim feedforward)
  → Take mean of unmasked positions (average pooling)
  → Linear(64 → 256) → LayerNorm → ReLU
  → 256-dim history encoding
```

**Why attention works well here:** Multi-head attention lets the network weigh
different actions in the history differently based on context. For example, when
facing a river bet, the pre-flop raise is more informative than the flop check —
attention learns which past actions matter most for the current decision.

**Handling empty history:** At the very start of a hand (first action), there's no
history. The transformer handles this by clamping sequence lengths to at least 1,
then zeroing out the output for genuinely empty histories. This avoids NaN from
softmax over all-masked positions.

**Output scaling:** The transformer's output projection is initialized with very
small weights (0.01×). This means when fine-tuning from a checkpoint, the history
encoding starts near zero and gradually grows in influence, avoiding sudden
disruption to the learned trunk.

### The Nine Actions

The action space uses pot-relative bet sizes:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | Fold | Give up the hand |
| 1 | Check/Call | Match the current bet (or check if no bet) |
| 2 | Raise 0.25× Pot | Small probe bet / min-raise territory |
| 3 | Raise 0.4× Pot | Standard small bet |
| 4 | Raise 0.6× Pot | Medium bet |
| 5 | Raise 0.8× Pot | Larger bet |
| 6 | Raise 1× Pot | Pot-sized bet |
| 7 | Raise 1.5× Pot | Overbet |
| 8 | All-In | Push all remaining chips |

Not all actions are legal in every situation. The environment provides a boolean
mask of legal actions, and the network only considers legal ones. For example, you
can't raise more than your stack, and you can't check when facing a bet.

### Replay Buffers

The two networks use fundamentally different buffer strategies:

**BR Buffer (Circular, 2M entries):**
Stores complete RL transitions: `(state, action, reward, next_state, done)`.
Works like a conveyor belt — new transitions push out old ones. This keeps BR
focused on exploiting the *current* AS policy. Old transitions from when AS played
differently would teach BR outdated counter-strategies.

**AS Buffer (Reservoir, 3M entries):**
Stores supervised examples: `(state, action)`. Uses reservoir sampling to maintain
a uniform sample across the *entire* training history. As the buffer fills, each
new sample has a decreasing probability of replacing an existing one. This
preserves the long-run average — if the buffer only kept recent data, AS would
chase the latest BR policy instead of averaging all past policies (which is required
for Nash equilibrium convergence).

### Learning Rate Schedule

Both networks use cosine decay with linear warmup:

```
Phase 1 — Warmup (0 to 8M env steps):
  LR ramps linearly from 1% to 100% of base rate (1e-4)
  Prevents large, destructive updates before the network has seen enough data

Phase 2 — Cosine decay (8M steps to end):
  LR follows a cosine curve from 100% down to 1% (1e-6)
  Gradually reduces step size as the network converges, preventing oscillation
```

The warm restart feature (`--restart-schedules`) resets this schedule to the
beginning while keeping model weights. This is equivalent to "cosine annealing
with warm restarts" — the LR jumps back up, giving the network a new opportunity
to escape local minima and find better solutions.

### Async Training Architecture

For GPU efficiency, self-play and training run on separate threads:

```
┌─────────────────────┐     ┌──────────────────────┐
│   Self-Play Thread  │     │   Training Thread     │
│                     │     │                       │
│  BR/AS inference    │     │  BR gradient updates  │
│  (GPU, batched)     │     │  AS gradient updates  │
│  2048 envs parallel │     │  (GPU, batched)       │
│         │           │     │         │             │
│         ▼           │     │         │             │
│  Fill BR + AS       │     │  Sample from buffers  │
│  replay buffers     │     │         │             │
│         │           │     │         ▼             │
│         │           │◄────│  Stage new weights    │
│  Apply staged       │     │  (CPU buffers)        │
│  weights between    │     │                       │
│  episodes           │     │                       │
└─────────────────────┘     └──────────────────────┘
                                      │
                               ┌──────▼──────┐
                               │  Eval Thread │
                               │  (CPU only)  │
                               │  No GPU      │
                               │  contention  │
                               └─────────────┘
```

**Weight synchronization** uses staged CPU buffers. The training thread copies
updated weights to CPU buffers and sets a flag. The self-play thread checks this
flag between episodes and copies the staged weights to its inference networks.
This avoids pausing either thread and prevents GPU deadlocks (a real problem on
AMD ROCm where two threads calling GPU sync operations can permanently block each
other).

**Evaluation** runs on CPU in a background thread using a deep copy of the AS
network. This means GPU training continues at full speed during eval.

**Pacing:** Training is allowed to run at most `train_ahead` rounds ahead of
self-play. This prevents the network from training too many times on the same
buffer data (overfitting to stale experience). If training gets too far ahead, it
sleeps briefly until self-play catches up.

### Convergence and Evaluation

The model is evaluated against three scripted baselines:

- **Random:** plays random legal actions — easy to beat, tests basic competence
- **Caller:** always calls — tests whether the agent bets for value and folds weak hands
- **TAG (Tight-Aggressive):** folds weak hands, calls medium hands, raises strong
  hands based on hand strength — the toughest baseline

Results are reported in **bb/100** (big blinds won per 100 hands). Positive means
winning, negative means losing. Confidence intervals use the 95% level.

A well-trained model should:
- Beat Random by 1500+ bb/100
- Beat Caller by 500+ bb/100
- Approach break-even (0 bb/100) against TAG

Getting close to break-even vs TAG indicates the model has learned a reasonable
approximation of GTO play, since TAG plays a simplified but solid strategy.