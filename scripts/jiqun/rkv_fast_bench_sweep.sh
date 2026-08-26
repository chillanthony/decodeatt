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
# Prefer the build venv's interpreter (avoids relying on `python` being on PATH).
PY=""
for cand in "${RKV_VENV:-}" "$ROOT_DIR/efficiency/.venv-rkv/bin/python" python3 python; do
  [[ -n "$cand" ]] || continue
  if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
    PY="$cand"
    break
  fi
done
if [[ -z "$PY" ]]; then
  echo "ERROR: no python interpreter found; activate the build venv first." >&2
  exit 1
fi
"$PY" "$ROOT_DIR/scripts/jiqun/rkv_fast_summarize.py" \
  "$ROOT_DIR/runs/rkvrepro/fast" > "$ROOT_DIR/efficiency/benchmark/RESULTS.md"
echo "-> efficiency/benchmark/RESULTS.md"
