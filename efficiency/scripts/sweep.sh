#!/usr/bin/env bash
#
# Run the full R-KV throughput sweep: single-GPU and
# 8-GPU data-parallel, then summarize into efficiency/benchmark/RESULTS.md.
#
# Usage:
#   bash efficiency/scripts/sweep.sh
#   bash efficiency/scripts/sweep.sh --single-only
#   bash efficiency/scripts/sweep.sh --dp-only
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RKV_VENV="/home/ma-user/.venvs/rkv-fast/bin/python"
export RKV_VENV

SINGLE=1
DP=1
for arg in "$@"; do
  case "$arg" in
    --single-only) DP=0 ;;
    --dp-only) SINGLE=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$SINGLE" == "1" ]]; then
  echo "===== single-GPU sweep ====="
  bash "$ROOT_DIR/efficiency/scripts/bench_single.sh"
fi
if [[ "$DP" == "1" ]]; then
  echo "===== 8-GPU data-parallel sweep ====="
  bash "$ROOT_DIR/efficiency/scripts/bench_dp.sh"
fi

echo "===== summarizing ====="
# Use the persistent patched-vLLM environment configured above.
PY="$RKV_VENV"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: patched-vLLM interpreter not found: $PY" >&2
  exit 1
fi
"$PY" "$ROOT_DIR/efficiency/scripts/summarize.py" \
  "$ROOT_DIR/runs/rkvrepro/fast" > "$ROOT_DIR/efficiency/benchmark/RESULTS.md"
echo "-> efficiency/benchmark/RESULTS.md"
