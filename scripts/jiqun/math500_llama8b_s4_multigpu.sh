#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-math500_llama8b_rkv_b1024_s4_multigpu}"
RUNS_ROOT="${RUNS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs}"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
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
DATASET="math500"
N="${N:-500}"
ARMS="${ARMS:-rkv@1024}"
MAX_NEW="${MAX_NEW:-16384}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"
PROBLEM_BATCH_SIZE="${PROBLEM_BATCH_SIZE:-1}"
LOG_MODE="${LOG_MODE:-brief}"

echo "[$RUN_NAME] checking cached HuggingFaceH4/MATH-500 test split"
if ! "$PYTHON_BIN" - <<'PY'
from kvbench.datasets import load_problems

problems = load_problems("math500", 500)
if len(problems) != 500:
    raise SystemExit(f"expected 500 cached MATH-500 problems, found {len(problems)}")
print(f"[dataset-check] dataset=math500 cached_problems={len(problems)}")
PY
then
  echo "MATH-500 cache check failed in offline mode." >&2
  echo "Populate it first with: bash scripts/jiqun/download_math500.sh" >&2
  exit 1
fi

CUDA_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
NPROC_PER_NODE="${NPROC_PER_NODE:-$CUDA_COUNT}"
MASTER_PORT="${MASTER_PORT:-29500}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-720}"

if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > CUDA_COUNT )); then
  echo "Invalid GPU count: requested=$NPROC_PER_NODE visible=$CUDA_COUNT" >&2
  exit 1
fi

echo "[$RUN_NAME] root=$ROOT_DIR"
echo "[$RUN_NAME] model=$MODEL"
echo "[$RUN_NAME] dataset=$DATASET n=$N source=offline-cache-only"
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
