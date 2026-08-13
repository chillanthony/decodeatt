#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/realtest2025072014}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/output}"
RESULT_DIR="${RESULT_DIR:-$RUN_DIR/result}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/math500_official_b1024_n20.log}"
OUT="${OUT:-$RESULT_DIR/math500_official_b1024_n20.json}"

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR" "$(dirname "$LOG_FILE")" "$(dirname "$OUT")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

VENV_DIR="${VENV_DIR:-$HOME/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="${MODEL:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-configs/experiments/math500_official_b1024.yaml}"
DATASET="${DATASET:-math500}"
N="${N:-20}"
MAX_NEW="${MAX_NEW:-4096}"

CUDA_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
NPROC_PER_NODE="${NPROC_PER_NODE:-$CUDA_COUNT}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

if (( NPROC_PER_NODE < 1 )); then
  echo "NPROC_PER_NODE must be at least 1, got $NPROC_PER_NODE" >&2
  exit 1
fi
if (( CUDA_COUNT < NPROC_PER_NODE )); then
  echo "Need at least $NPROC_PER_NODE visible CUDA devices, found $CUDA_COUNT" >&2
  exit 1
fi
if (( NNODES < 1 )); then
  echo "NNODES must be at least 1, got $NNODES" >&2
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, $NNODES), got $NODE_RANK" >&2
  exit 1
fi

if (( NNODES == 1 )); then
  LAUNCH_ARGS=(--standalone)
else
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR is required when NNODES > 1" >&2
    exit 1
  fi
  LAUNCH_ARGS=(
    --nnodes="$NNODES"
    --node-rank="$NODE_RANK"
    --master-addr="$MASTER_ADDR"
    --master-port="$MASTER_PORT"
  )
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[math500-b1024-n20] root=$ROOT_DIR"
echo "[math500-b1024-n20] model=$MODEL"
echo "[math500-b1024-n20] hf_home=$HF_HOME"
echo "[math500-b1024-n20] hf_hub_offline=$HF_HUB_OFFLINE"
echo "[math500-b1024-n20] hf_datasets_offline=$HF_DATASETS_OFFLINE"
echo "[math500-b1024-n20] config=$CONFIG"
echo "[math500-b1024-n20] dataset=$DATASET"
echo "[math500-b1024-n20] n=$N"
echo "[math500-b1024-n20] max_new=$MAX_NEW"
echo "[math500-b1024-n20] nnodes=$NNODES"
echo "[math500-b1024-n20] node_rank=$NODE_RANK"
echo "[math500-b1024-n20] nproc_per_node=$NPROC_PER_NODE"
echo "[math500-b1024-n20] cuda_count=$CUDA_COUNT"
echo "[math500-b1024-n20] output_dir=$OUTPUT_DIR"
echo "[math500-b1024-n20] result_dir=$RESULT_DIR"
echo "[math500-b1024-n20] out=$OUT"
echo "[math500-b1024-n20] log_file=$LOG_FILE"
echo "[math500-b1024-n20] python_bin=$PYTHON_BIN"

"$PYTHON_BIN" -m torch.distributed.run \
  "${LAUNCH_ARGS[@]}" \
  --nproc-per-node="$NPROC_PER_NODE" \
  -- \
  scripts/eval.py \
  --distributed \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --n "$N" \
  --max-new "$MAX_NEW" \
  --model "$MODEL" \
  --out "$OUT"
