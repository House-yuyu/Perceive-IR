#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/perceiveIR_stage1.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
  medium) MODULE=perceiveIR.train_stage1_medium; EXTRA=() ;;
  prompts) MODULE=perceiveIR.train_stage1_prompts; EXTRA=(--task-balanced) ;;
  *) echo 'Usage: bash scripts/train_perceiveIR_stage1.sh {medium --heldout-fold 0|medium --heldout-fold 1|prompts} [options]' >&2; exit 2 ;;
esac
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m "$MODULE" --config "$CONFIG" "${EXTRA[@]}" "$@"
