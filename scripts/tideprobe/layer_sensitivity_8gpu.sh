#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$ROOT_DIR"

require_python

export PYTHONPATH HF_HOME HF_ENDPOINT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-$HF_HOME/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/experiments/tideprobe_step1.yaml}"
EVENTS="${EVENTS:-$TIDEPROBE_ROOT/transition_events.jsonl}"
OUT_DIR="${OUT_DIR:-$TIDEPROBE_ROOT/layer_sensitivity}"
SCAN_MODE="${SCAN_MODE:-group}"
BUDGETS="${BUDGETS:-256,512}"
MASK_POLICIES="${MASK_POLICIES:-random,window}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0,1,2}"
PROBE_TYPES="${PROBE_TYPES:-uniform,transition,ordinary}"
LAYERS="${LAYERS:-}"
RUN_TAG="${RUN_TAG:-${MASK_POLICIES//,/_}}"
MASTER_PORT="${MASTER_PORT:-29503}"

CUDA_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if (( CUDA_COUNT < 8 )); then
  echo "TideProbe layer probing requires 8 visible GPUs; found $CUDA_COUNT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
EXTRA_ARGS=()
if [[ -n "$LAYERS" ]]; then
  EXTRA_ARGS+=(--layers "$LAYERS")
fi

echo "[tideprobe] layer scan=$SCAN_MODE budgets=$BUDGETS policies=$MASK_POLICIES"
echo "[tideprobe] traces=$TRACE_DIR events=$EVENTS output=$OUT_DIR"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  --master-port="$MASTER_PORT" \
  --module \
  kvbench.diagnostics.layer_sensitivity \
  --config "$CONFIG" \
  --model "$MODEL" \
  --trace-dir "$TRACE_DIR" \
  --events "$EVENTS" \
  --out-dir "$OUT_DIR" \
  --scan-mode "$SCAN_MODE" \
  --budgets "$BUDGETS" \
  --mask-policies "$MASK_POLICIES" \
  --random-seeds "$RANDOM_SEEDS" \
  --probe-types "$PROBE_TYPES" \
  --run-tag "$RUN_TAG" \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" -m kvbench.diagnostics.layer_analysis \
  --config "$CONFIG" \
  --shard-dir "$OUT_DIR"

echo "[tideprobe] layer sensitivity complete"
