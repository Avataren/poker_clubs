#!/usr/bin/env bash
set -euo pipefail

# Always run relative to this script's directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

while true; do
  "$PYTHON_BIN" scripts/milestone_eval.py \
    --checkpoint-dir checkpoints \
    --device cuda \
    --num-hands 100000 \
    --exploitability-hands 25000 \
    --milestone-every 1000000 \
    --min-episode 1000000 \
    --csv logs/milestones.csv \
    --best-path checkpoints/checkpoint_best.pt \
    --min-tag-lb95-for-promotion 0.0
  sleep 1800
done
