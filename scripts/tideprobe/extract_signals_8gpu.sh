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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-$HF_HOME/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-configs/experiments/tideprobe_step1.yaml}"
TRACE_DIR="${TRACE_DIR:-/home/ma-user/work/bucket-wulan-green/chenyanbo/trace}"
SIGNALS_DIR="${SIGNALS_DIR:-runs/tideprobe_step1/signals}"
MASTER_PORT="${MASTER_PORT:-29502}"

CUDA_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if (( CUDA_COUNT < 8 )); then
  echo "TideProbe signal extraction requires 8 visible GPUs; found $CUDA_COUNT" >&2
  exit 1
fi

mkdir -p "$SIGNALS_DIR"
echo "[tideprobe] extracting transition signals with 8 GPUs"
echo "[tideprobe] traces=$TRACE_DIR signals=$SIGNALS_DIR"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  --master-port="$MASTER_PORT" \
  --module \
  kvbench.diagnostics.transition_signals \
  --config "$CONFIG" \
  --model "$MODEL" \
  --trace-dir "$TRACE_DIR" \
  --signals-dir "$SIGNALS_DIR"

echo "[tideprobe] signal extraction complete: $SIGNALS_DIR"
