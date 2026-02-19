#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PYTHON_DIR"

source .venv/bin/activate

echo "=== Building Rust engine ==="
ENGINE_DIR="../engine"
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
SO_NAME="poker_ai/engine.cpython-${PYVER}-x86_64-linux-gnu.so"
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Remove ALL stale engine .so files (local + site-packages)
rm -f "$SO_NAME"
rm -f "$SITE_PACKAGES"/engine.cpython-*.so 2>/dev/null || true
# Clear Python import caches
find poker_ai -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Force cargo to recompile the bindings
touch "$ENGINE_DIR/src/python_bindings.rs"
PYO3_PYTHON="$(which python)" cargo build --release --manifest-path "$ENGINE_DIR/Cargo.toml"

# Copy fresh .so
cp "$ENGINE_DIR/target/release/libpoker_ai_engine.so" "$SO_NAME"

# Verify the new .so has expected methods
python -c "
from poker_ai.env.poker_env import BatchPokerEnv
assert hasattr(BatchPokerEnv, 'reset_player_stats'), \
    'Engine .so is stale — reset_player_stats missing'
print('Engine OK')
"
echo "=== Engine built ==="

echo "=== Starting training ==="
python scripts/train.py "$@"
