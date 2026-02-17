#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PYTHON_DIR"

source .venv/bin/activate

echo "=== Building Rust engine ==="
PYO3_PYTHON="$(which python)" cargo build --release --manifest-path ../engine/Cargo.toml
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
cp ../engine/target/release/libpoker_ai_engine.so \
   "poker_ai/engine.cpython-${PYVER}-x86_64-linux-gnu.so"
python -c "from poker_ai.engine import BatchPokerEnv; print('Engine OK')"
echo "=== Engine built ==="

echo "=== Starting training ==="
python scripts/train.py "$@"
