#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"
require_python

# Usage:
#   MODEL=/local/DeepSeek-R1-Distill-Llama-8B \
#     scripts/experiments/throughput_smoke.sh
# Optional overrides: SMOKE_BATCHES, SMOKE_ARMS, SMOKE_N, SMOKE_MAX_NEW,
# SMOKE_OUT, SMOKE_DTYPE, SMOKE_DEVICE_MAP.
MODEL_PATH="${MODEL:?set MODEL to a local model directory or HF id}"
CONFIG="$ROOT_DIR/configs/experiments/throughput_smoke.yaml"
OUT_ROOT="${SMOKE_OUT:-$RUNS_ROOT/throughput_smoke}"
DATASET="${SMOKE_DATASET:-sample}"
N="${SMOKE_N:-8}"
MAX_NEW="${SMOKE_MAX_NEW:-512}"
ARMS="${SMOKE_ARMS:-fullkv,rkv@256}"
BATCHES="${SMOKE_BATCHES:-1,2,4}"
BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
DTYPE="${SMOKE_DTYPE:-bfloat16}"
DEVICE_MAP="${SMOKE_DEVICE_MAP:-cuda}"

mkdir -p "$OUT_ROOT"
echo "model=$MODEL_PATH"
echo "dataset=$DATASET n=$N max_new=$MAX_NEW arms=$ARMS"
echo "problem_batch_sizes=$BATCHES candidate_batch_size=$BATCH_SIZE"

IFS=',' read -r -a batch_sizes <<< "$BATCHES"
for problem_batch_size in "${batch_sizes[@]}"; do
  [[ "$problem_batch_size" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid batch size: $problem_batch_size" >&2
    exit 2
  }
  out="$OUT_ROOT/batch${problem_batch_size}.json"
  echo
  echo "=== problem_batch_size=$problem_batch_size ==="
  "$PYTHON_BIN" "$ROOT_DIR/scripts/eval.py" \
    --config "$CONFIG" \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --n "$N" \
    --max-new "$MAX_NEW" \
    --arms "$ARMS" \
    --batch-size "$BATCH_SIZE" \
    --problem-batch-size "$problem_batch_size" \
    --prompt-bucket-size "$((problem_batch_size * 4))" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --greedy \
    --log-mode brief \
    --out "$out"
done

echo
echo "Smoke results written to $OUT_ROOT"
echo "Read summary.mean_batch_tokens_per_sec for aggregate throughput and"
echo "summary.mean_tokens_per_sec for per-request amortized throughput."
