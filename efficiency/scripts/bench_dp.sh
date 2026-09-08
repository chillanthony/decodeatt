#!/usr/bin/env bash
#
# Offline batched benchmark across all 8 A100/A800,
# vLLM data-parallel (--data-parallel-size 8, tp=1). Each replica runs its own
# independent R-KV, so no cross-rank communication; total throughput scales
# ~linearly (reference RESULTS_dp.md reports 7.6x @ DP=8).
#
# Reports TOTAL and PER-GPU decode tok/s (the per-GPU figure is directly
# comparable to the reference paper's single-GPU number).
#
# Usage:
#   bash efficiency/scripts/bench_dp.sh
#   DP=4 bash efficiency/scripts/bench_dp.sh   # fewer replicas
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EFF="$ROOT_DIR/efficiency"
EVAL="$EFF/benchmark/eval_offline.py"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
DATASET="${DATASET:-aime}"
N="${N:-30}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
BUFFER="${BUFFER:-128}"
DP="${DP:-8}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT_DIR/runs/rkvrepro}"
OUT_DIR="$RUNS_ROOT/fast"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for label in fullkv_production fullkv_constrained rkv_b2048_buf${BUFFER}; do
    echo "label=$label dp=${DP} out=$OUT_DIR/${label}_dp${DP}.json"
  done
  exit 0
fi

mkdir -p "$OUT_DIR"
# Prefer the build venv's interpreter; fall back to whatever `python`/`python3`
# the user has on PATH (works if they `source efficiency/.venv-rkv/bin/activate`).
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in "${RKV_VENV:-}" "/home/ma-user/.venvs/rkv-fast/bin/python" "$ROOT_DIR/efficiency/.venv-rkv/bin/python" python python3; do
    [[ -n "$cand" ]] || continue
    if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
      PY="$cand"
      break
    fi
  done
fi
[[ -n "$PY" ]] || { echo "ERROR: no python interpreter found" >&2; exit 1; }
BASE=("$PY" "$EVAL" --model "$MODEL" --data "$DATASET" --n "$N" --max-tokens "$MAX_TOKENS" --dp "$DP")

echo ">> Full-KV production (prefix caching on), DP=$DP"
"${BASE[@]}" --label fullkv_production --out "$OUT_DIR/fullkv_production_dp${DP}.json"

echo ">> Full-KV constrained (prefix caching off), DP=$DP"
"${BASE[@]}" --no-prefix --label fullkv_constrained --out "$OUT_DIR/fullkv_constrained_dp${DP}.json"

for B in 2048 512 256; do
  echo ">> R-KV budget=$B buffer=$BUFFER, DP=$DP"
  VLLM_V1_R_KV_BUDGET="$B" VLLM_V1_R_KV_BUFFER="$BUFFER" VLLM_V1_R_KV_ASYNC=1 \
    "${BASE[@]}" --label "rkv_b${B}_buf${BUFFER}" --dp "$DP" \
    --out "$OUT_DIR/rkv_b${B}_buf${BUFFER}_dp${DP}.json"
done

echo ">> done -> $OUT_DIR"
