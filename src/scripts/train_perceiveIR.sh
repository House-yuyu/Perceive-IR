#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
CONFIG="${1:-configs/perceiveIR_3task.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); print(len(c["data"].get("batch_size_by_rank", [1])))' "$CONFIG")}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m perceiveIR.train --config "$CONFIG" "$@"
