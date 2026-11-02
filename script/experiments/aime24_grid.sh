#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/4090.env}"
source "$ENV_FILE"
cd "$ROOT_DIR"

SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
GRID_NAME="aime24_llama8b_main_grid_s${SAMPLES}"
SUMMARY_DIR="${SUMMARY_DIR:-$RUNS_ROOT/$GRID_NAME}"
MODEL_CONFIG="config/models/deepseek_r1_distill_llama8b.yaml"
DATA_CONFIG="config/datasets/aime24.yaml"

STRATEGIES=(
  fullkv
  snapkv snapkv snapkv snapkv
  rkv rkv rkv rkv
)

BUDGETS=(
  ""
  1024 1536 2048 2560
  1024 1536 2048 2560
)

RUN_NAMES=(
  "aime24_llama8b_fullkv_s${SAMPLES}_multigpu"
  "aime24_llama8b_snapkv_b1024_s${SAMPLES}_multigpu"
  "aime24_llama8b_snapkv_b1536_s${SAMPLES}_multigpu"
  "aime24_llama8b_snapkv_b2048_s${SAMPLES}_multigpu"
  "aime24_llama8b_snapkv_b2560_s${SAMPLES}_multigpu"
  "aime24_llama8b_rkv_b1024_s${SAMPLES}_multigpu"
  "aime24_llama8b_rkv_b1536_s${SAMPLES}_multigpu"
  "aime24_llama8b_rkv_b2048_s${SAMPLES}_multigpu"
  "aime24_llama8b_rkv_b2560_s${SAMPLES}_multigpu"
)

RESULT_PATHS=()
require_python

export PYTHONPATH HF_HOME HF_ENDPOINT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for index in "${!STRATEGIES[@]}"; do
  strategy="${STRATEGIES[$index]}"
  budget="${BUDGETS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  echo "[$GRID_NAME] starting $run_name via $strategy${budget:+@$budget}"
  (
    unset RUN_DIR OUT
    export RUN_NAME="$run_name"
    export DATA_CONFIG MODEL_CONFIG
    export STRATEGY_CONFIG="config/strategies/${strategy}.yaml"
    export ARMS="${strategy}${budget:+@$budget}"
    export NUM_RETURN_SEQUENCES="$SAMPLES" RUNS_ROOT
    # fullkv has an unbounded cache; keep its micro-batch small to stay within
    # 80GB on the A100. Sparse-cache arms inherit the outer BATCH_SIZE.
    if [[ "$strategy" == "fullkv" ]]; then
      export BATCH_SIZE="${FULLKV_BATCH_SIZE:-8}"
      unset PROBLEM_BATCH_SIZE
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "run_name=$RUN_NAME dataset=aime strategy=$ARMS model_config=$MODEL_CONFIG data_config=$DATA_CONFIG strategy_config=$STRATEGY_CONFIG"
      echo "result=$RUNS_ROOT/$RUN_NAME/$RUN_NAME.json"
      exit 0
    fi
    RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
    OUT="${OUT:-$RUN_DIR/$RUN_NAME.json}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    PROBLEM_BATCH_SIZE="${PROBLEM_BATCH_SIZE:-1}"
    LOG_MODE="${LOG_MODE:-brief}"
    mkdir -p "$RUN_DIR" "$(dirname "$OUT")"
    CUDA_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
    NPROC_PER_NODE="${NPROC_PER_NODE:-$CUDA_COUNT}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-720}"
    if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > CUDA_COUNT )); then
      echo "Invalid GPU count: requested=$NPROC_PER_NODE visible=$CUDA_COUNT" >&2
      exit 1
    fi
    echo "[$RUN_NAME] GPUs=$NPROC_PER_NODE/$CUDA_COUNT arms=$ARMS batch_size=$BATCH_SIZE output=$OUT"
    "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      --nproc-per-node="$NPROC_PER_NODE" \
      --master-port="$MASTER_PORT" \
      -- \
      script/eval.py \
      --distributed \
      --distributed-timeout-minutes "$TIMEOUT_MINUTES" \
      --model-config "$MODEL_CONFIG" \
      --data-config "$DATA_CONFIG" \
      --strategy-config "$STRATEGY_CONFIG" \
      --arms "$ARMS" \
      --batch-size "$BATCH_SIZE" \
      --num-return-sequences "$NUM_RETURN_SEQUENCES" \
      --problem-batch-size "$PROBLEM_BATCH_SIZE" \
      --log-mode "$LOG_MODE" \
      --out "$OUT"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi

  result_path="$RUNS_ROOT/$run_name/$run_name.json"
  if [[ ! -s "$result_path" ]]; then
    echo "Missing result after successful run: $result_path" >&2
    exit 1
  fi
  RESULT_PATHS+=("$result_path")
  echo "[$GRID_NAME] completed $run_name result=$result_path"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[$GRID_NAME] dry run complete"
  exit 0
fi

mkdir -p "$SUMMARY_DIR"
require_python

"$PYTHON_BIN" script/summarize_results.py \
  "${RESULT_PATHS[@]}" \
  --out "$SUMMARY_DIR/summary.csv" \
  --json-out "$SUMMARY_DIR/summary.json"

echo "[$GRID_NAME] summary_csv=$SUMMARY_DIR/summary.csv"
echo "[$GRID_NAME] summary_json=$SUMMARY_DIR/summary.json"
