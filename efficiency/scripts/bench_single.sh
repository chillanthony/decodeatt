#!/usr/bin/env bash
#
# Offline batched benchmark on one A100/A800.
#
# Reports per-GPU decode throughput for the paper-comparable comparator:
#   * Full-KV production      (prefix caching on)
#   * Full-KV constrained     (prefix caching off -> the fair R-KV A/B)
#   * R-KV @ budget 2048/512/256 (buffer 128)
#
# Model + dataset default to your research workload (R1-Distill-Llama-8B, AIME).
# Override: MODEL, DATASET, RUNS_ROOT.
#
# Usage:
#   bash efficiency/scripts/bench_single.sh
#   MODEL=Qwen/Qwen2.5-Math-7B-Instruct DATASET=gsm8k \
#     bash efficiency/scripts/bench_single.sh   # reproduce the paper number
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
RUNS_ROOT="${RUNS_ROOT:-$ROOT_DIR/runs/rkvrepro}"
OUT_DIR="$RUNS_ROOT/fast"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for label in fullkv_production fullkv_constrained rkv_b2048_buf${BUFFER} rkv_b512_buf${BUFFER} rkv_b256_buf${BUFFER}; do
    echo "label=$label out=$OUT_DIR/${label}_s1gpu.json"
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
BASE=("$PY" "$EVAL" --model "$MODEL" --data "$DATASET" --n "$N" --max-tokens "$MAX_TOKENS")

echo ">> Full-KV production (prefix caching on)"
"${BASE[@]}" --label fullkv_production --out "$OUT_DIR/fullkv_production_s1gpu.json"

echo ">> Full-KV constrained (prefix caching off)"
"${BASE[@]}" --no-prefix --label fullkv_constrained --out "$OUT_DIR/fullkv_constrained_s1gpu.json"

for B in 2048 512 256; do
  echo ">> R-KV budget=$B buffer=$BUFFER"
  VLLM_V1_R_KV_BUDGET="$B" VLLM_V1_R_KV_BUFFER="$BUFFER" VLLM_V1_R_KV_ASYNC=1 \
    "${BASE[@]}" --label "rkv_b${B}_buf${BUFFER}" \
    --out "$OUT_DIR/rkv_b${B}_buf${BUFFER}_s1gpu.json"
done

echo ">> done -> $OUT_DIR"
