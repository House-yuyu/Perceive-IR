#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
TASKS="${1:-3}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TASKS" in
  3) PROTOCOL=adair3 ;;
  5) PROTOCOL=adair5 ;;
  *) echo 'Usage: bash scripts/test_perceiveIR.sh {3|5} [evaluation options]' >&2; exit 2 ;;
esac
exec "$PYTHON" -m perceiveIR.evaluate_stage2_adair \
  --config "configs/perceiveIR_${TASKS}task.yaml" \
  --checkpoint "weight/perceiveIR_${TASKS}task.pth" \
  --output "results/perceiveIR_${TASKS}task" \
  --protocol "$PROTOCOL" "$@"
