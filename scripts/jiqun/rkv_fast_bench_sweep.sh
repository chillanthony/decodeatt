#!/usr/bin/env bash
#
# rkv_fast_bench_sweep.sh — run the full R-KV throughput sweep: single-GPU and
# 8-GPU data-parallel, then summarize into efficiency/benchmark/RESULTS.md.
#
# Usage:
#   bash scripts/jiqun/rkv_fast_bench_sweep.sh
#   bash scripts/jiqun/rkv_fast_bench_sweep.sh --single-only
#   bash scripts/jiqun/rkv_fast_bench_sweep.sh --dp-only
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RKV_VENV="/home/ma-user/.venv/rkv-fast/bin/python"
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
  bash "$ROOT_DIR/scripts/jiqun/rkv_fast_bench_single.sh"
fi
if [[ "$DP" == "1" ]]; then
  echo "===== 8-GPU data-parallel sweep ====="
  bash "$ROOT_DIR/scripts/jiqun/rkv_fast_bench_dp8.sh"
fi

echo "===== summarizing ====="
# Use the persistent patched-vLLM environment configured above.
PY="$RKV_VENV"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: patched-vLLM interpreter not found: $PY" >&2
  exit 1
fi
"$PY" "$ROOT_DIR/scripts/jiqun/rkv_fast_summarize.py" \
  "$ROOT_DIR/runs/rkvrepro/fast" > "$ROOT_DIR/efficiency/benchmark/RESULTS.md"
echo "-> efficiency/benchmark/RESULTS.md"
