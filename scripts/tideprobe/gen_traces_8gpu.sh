#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$ROOT_DIR"

require_python

export PYTHONPATH HF_HOME HF_ENDPOINT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-$HF_HOME/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/experiments/tideprobe_step1.yaml}"
N="${N:-10}"
MAX_NEW="${MAX_NEW:-32768}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

CUDA_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if (( CUDA_COUNT < 8 )); then
  echo "TideProbe trace generation requires 8 visible GPUs; found $CUDA_COUNT" >&2
  exit 1
fi
if (( NPROC_PER_NODE != 8 )); then
  echo "NPROC_PER_NODE must be 8 for this launcher; got $NPROC_PER_NODE" >&2
  exit 1
fi

mkdir -p "$TRACE_DIR"
echo "[tideprobe] model=$MODEL dataset=aime n=$N max_new=$MAX_NEW seed=$SEED"
echo "[tideprobe] gpus=$NPROC_PER_NODE trace_dir=$TRACE_DIR"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  --master-port="$MASTER_PORT" \
  --module \
  kvbench.diagnostics.gen_traces \
  --config "$CONFIG" \
  --model "$MODEL" \
  --dataset aime \
  --n "$N" \
  --max-new "$MAX_NEW" \
  --seed "$SEED" \
  --trace-dir "$TRACE_DIR"

echo "[tideprobe] complete: $TRACE_DIR"
