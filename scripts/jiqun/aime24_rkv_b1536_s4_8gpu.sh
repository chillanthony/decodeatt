#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-aime24_rkv_b1536_s4_8gpu}"
RUN_DIR="${RUN_DIR:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/output}"
RESULT_DIR="${RESULT_DIR:-$RUN_DIR/result}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/$RUN_NAME.log}"
OUT="${OUT:-$RESULT_DIR/$RUN_NAME.json}"

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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-configs/onestrategy/rkv.yaml}"
DATASET="${DATASET:-aime}"
N="${N:-30}"
ARMS="${ARMS:-rkv@1536}"
MAX_NEW="${MAX_NEW:-32768}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"
PROBLEM_BATCH_SIZE="${PROBLEM_BATCH_SIZE:-1}"
LOG_MODE="${LOG_MODE:-brief}"

CUDA_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-720}"

if (( NPROC_PER_NODE != 8 )); then
  echo "This reproduction is configured for exactly 8 processes/GPUs; got NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi
if (( CUDA_COUNT < NPROC_PER_NODE )); then
  echo "Need at least $NPROC_PER_NODE visible CUDA devices, found $CUDA_COUNT" >&2
  exit 1
fi

echo "[$RUN_NAME] root=$ROOT_DIR"
echo "[$RUN_NAME] model=$MODEL"
echo "[$RUN_NAME] dataset=$DATASET n=$N"
echo "[$RUN_NAME] arms=$ARMS max_new=$MAX_NEW"
echo "[$RUN_NAME] batch_size=$BATCH_SIZE num_return_sequences=$NUM_RETURN_SEQUENCES"
echo "[$RUN_NAME] problem_batch_size=$PROBLEM_BATCH_SIZE"
echo "[$RUN_NAME] log_mode=$LOG_MODE"
echo "[$RUN_NAME] nproc_per_node=$NPROC_PER_NODE cuda_count=$CUDA_COUNT"
echo "[$RUN_NAME] hf_home=$HF_HOME hf_endpoint=$HF_ENDPOINT"
echo "[$RUN_NAME] hf_hub_offline=$HF_HUB_OFFLINE hf_datasets_offline=$HF_DATASETS_OFFLINE"
echo "[$RUN_NAME] hf_datasets_cache=$HF_DATASETS_CACHE"
echo "[$RUN_NAME] out=$OUT"
echo "[$RUN_NAME] log_file=$LOG_FILE"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-port="$MASTER_PORT" \
  -- \
  scripts/eval.py \
  --distributed \
  --distributed-timeout-minutes "$TIMEOUT_MINUTES" \
  --config "$CONFIG" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --n "$N" \
  --arms "$ARMS" \
  --max-new "$MAX_NEW" \
  --batch-size "$BATCH_SIZE" \
  --num-return-sequences "$NUM_RETURN_SEQUENCES" \
  --problem-batch-size "$PROBLEM_BATCH_SIZE" \
  --log-mode "$LOG_MODE" \
  --out "$OUT"

echo "[$RUN_NAME] done"
echo "[$RUN_NAME] result=$OUT"
