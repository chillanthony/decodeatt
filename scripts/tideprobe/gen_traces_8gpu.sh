#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-$HF_HOME/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-configs/experiments/tideprobe_step1.yaml}"
TRACE_DIR="${TRACE_DIR:-/home/ma-user/work/bucket-wulan-green/chenyanbo/trace}"
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
