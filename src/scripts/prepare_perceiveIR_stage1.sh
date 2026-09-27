#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/perceiveIR_stage1.yaml}"
FOLD0="${FOLD0:-weight/stage1/holdout_0/medium_020000.pth}"
FOLD1="${FOLD1:-weight/stage1/holdout_1/medium_020000.pth}"
"$PYTHON" -m perceiveIR.render_stage1_medium --config "$CONFIG" \
  --heldout-fold 0 --checkpoint "$FOLD0" "$@"
"$PYTHON" -m perceiveIR.render_stage1_medium --config "$CONFIG" \
  --heldout-fold 1 --checkpoint "$FOLD1" "$@"
"$PYTHON" -m perceiveIR.validate_stage1_triplets --config "$CONFIG" --verify-images
