# Poker AI - Neural Fictitious Self-Play (NFSP)

Self-play neural network training system for No-Limit Texas Hold'em (2-9 players). Uses a fast Rust game engine with Python bindings, training via NFSP which combines DQN (best response) with supervised learning (average strategy) to converge toward Nash equilibrium.

## Project Structure

```
poker_ai/
  engine/           # Rust game engine with PyO3 bindings
  python/
    poker_ai/
      env/           # Gym-like environment wrappers
      model/         # Neural network architectures
      training/      # NFSP training loop, replay buffers, self-play
      export/        # ONNX export
      config/        # Hyperparameters
    scripts/         # Entry points (train, evaluate, benchmark, export)
    tests/           # Unit tests
```

## Setup

### Prerequisites

- Rust toolchain (rustup)
- Python 3.10-3.12 (3.12 recommended; 3.13+ lacks ROCm/CUDA wheel support)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip + venv

### 1. Create virtual environment

```bash
cd poker_ai/python

# With uv (recommended)
uv venv .venv --python 3.12
```

Or with standard venv if you already have Python 3.12:
```bash
python3.12 -m venv .venv
```

### 2. Install PyTorch (GPU-specific)

Activate the venv first:
```bash
source .venv/bin/activate
```

**AMD GPU (ROCm 6.2) — e.g. RX 7900 XTX, RX 7900 XT, RX 7800 XT:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

**NVIDIA GPU (CUDA 12.x) — e.g. RTX 4090, RTX 3090, A100:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> With `uv`, replace `pip install` with `uv pip install`.

### 3. Install remaining dependencies

```bash
pip install numpy pytest tqdm tensorboard onnx onnxruntime maturin
```

### 4. Build the Rust engine

```bash
maturin develop --release
```

### 5. Verify

```bash
python -m pytest tests/ -v
```

## Training

### Basic (heads-up, GPU)

```bash
python scripts/train.py --num-players 2 --device cuda
```

### CPU training

```bash
python scripts/train.py --num-players 2 --device cpu
```

### Recommended Long Run (Heads-Up, 24GB GPU)

```bash
python scripts/train.py \
  --num-players 2 \
  --num-envs 512 \
  --episodes 100000000 \
  --device cuda \
  --batch-size 4096 \
  --br-train-steps 12 \
  --as-train-steps 6 \
  --eta 0.1 \
  --br-lr 0.0001 \
  --as-lr 0.0005 \
  --epsilon-start 0.06 \
  --epsilon-end 0.003 \
  --epsilon-decay-steps 300000000 \
  --eval-every 200000 \
  --eval-hands 5000 \
  --checkpoint-every 200000 \
  --checkpoint-dir checkpoints \
  --log-dir logs
```

Disable mixed precision:
```bash
python scripts/train.py --num-players 2 --device cuda --no-amp
```

### Resume from checkpoint

```bash
python scripts/train.py --resume checkpoints/checkpoint_latest.pt --device cuda
```

### Milestone Evaluation (High Confidence)

Use lightweight in-training eval for trend tracking, then run high-sample
milestone eval on 1M-episode checkpoints:

```bash
python scripts/milestone_eval.py \
  --checkpoint-dir checkpoints \
  --device cuda \
  --num-hands 100000 \
  --exploitability-hands 25000 \
  --milestone-every 1000000 \
  --min-episode 1000000 \
  --csv logs/milestones.csv \
  --best-path checkpoints/checkpoint_best.pt \
  --min-tag-lb95-for-promotion 0.0
```

You can re-run this command while training; it skips milestones already written
to the CSV and only evaluates new checkpoints.
The CSV now includes exploitability proxy fields (BR vs AS); lower is better.

If you are using `run_eval.sh`, run it in a second terminal while training:
```bash
./run_eval.sh
```

### Recommended training progression

1. Train heads-up (2 players) first — fastest convergence
2. Fine-tune on 6-player tables
3. Then 9-player tables

## Evaluation

```bash
python scripts/evaluate.py checkpoints/checkpoint_latest.pt --num-hands 10000 --device cuda
```

Reports bb/100 win rate against random and calling-station baselines.
Evaluation output also includes a tight-aggressive (TAG) scripted baseline and 95% confidence intervals.

### Exploitability Proxy (Heads-Up)

Evaluate approximate exploitability using greedy BR vs AS policy:

```bash
python scripts/eval_exploitability.py checkpoints/checkpoint_latest.pt --num-hands 10000 --device cuda
```

Lower is better; values near 0 suggest lower exploitability in this abstraction.

## ONNX Export

Export the trained average strategy network for use in the Rust backend:

```bash
python scripts/export_onnx.py checkpoints/checkpoint_latest.pt -o poker_as_net.onnx --verify
```

## Benchmarking

Measure engine speed (hands/sec):

```bash
python scripts/benchmark_env.py --hands 100000
```

## Monitoring

Training logs to TensorBoard:

```bash
tensorboard --logdir logs/
```

Tracks: BR/AS loss curves, epsilon decay, buffer sizes, eval win rates.

## GPU Notes

| GPU | Setup | `--device` |
|---|---|---|
| AMD (RDNA3, RDNA2) | ROCm 6.2 PyTorch wheel | `cuda` |
| NVIDIA (Ampere+) | CUDA 12.x PyTorch wheel | `cuda` |
| Apple Silicon | Default PyTorch (pip install torch) | `mps` |
| CPU | Any PyTorch | `cpu` |

- ROCm exposes AMD GPUs as CUDA devices via HIP — use `--device cuda`, not `rocm`
- The `amdgpu.ids: No such file or directory` warning on AMD is harmless
- Apple MPS support is untested but should work with the standard PyTorch install
