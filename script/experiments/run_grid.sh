#!/usr/bin/env bash
# Main budget grid: FullKV plus SnapKV/R-KV at several budgets on one dataset.
#
#   bash script/experiments/run_grid.sh aime24|math500
#
# Every arm runs as its own torchrun job and writes runs/<name>/<name>.json.
# Environment overrides:
#   NUM_RETURN_SEQUENCES  samples per problem (default 4)
#   BUDGETS               space-separated budgets (default per dataset below)
#   GRID_STRATEGIES       space-separated eviction strategies (default "snapkv rkv")
#   BATCH_SIZE / FULLKV_BATCH_SIZE  candidates per problem in one batch (default 4 / per dataset below)
#   ROWS_PER_BATCH        target rows (problems x candidates) per GPU batch for eviction arms
#                         (default 256 * 1024 / max(budget, 1024): 256 rows up to budget 1024, 128 at 2048)
#   FULLKV_ROWS_PER_BATCH same for fullkv, whose cache is not bounded by a budget (default 16 math500, 8 aime24)
#   PROBLEM_BATCH_SIZE / FULLKV_PROBLEM_BATCH_SIZE  set problems per batch directly (override the row targets)
#                         Defaults derive from the row targets, capped at ceil(problems / GPUs) so the
#                         dynamic problem claiming keeps every GPU busy. A batch that runs out of memory is
#                         split in half automatically, so the targets can be generous.
#   LOG_MODE / NPROC_PER_NODE
#   PYTORCH_CUDA_ALLOC_CONF  default expandable_segments:True (a growing cache otherwise fragments memory
#                         and OOMs at batches that fit)
#   OVERWRITE=1           start every arm over instead of resuming its result file
#   DRY_RUN=1             print the planned runs only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/a800.env}"
source "$ENV_FILE"
cd "$ROOT_DIR"

DATASET="${1:?usage: run_grid.sh aime24|math500}"
case "$DATASET" in
  aime24)
    DEFAULT_BUDGETS="1024 1536 2048 2560"
    DEFAULT_FULLKV_BATCH_SIZE=8
    DEFAULT_FULLKV_ROWS=8
    NUM_PROBLEMS=30
    ;;
  math500)
    DEFAULT_BUDGETS="128 256 512 768 1024 1536 2048"
    DEFAULT_FULLKV_BATCH_SIZE=4
    DEFAULT_FULLKV_ROWS=16
    NUM_PROBLEMS=500
    ;;
  *)
    echo "unknown dataset $DATASET; expected aime24 or math500" >&2
    exit 1
    ;;
esac

SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
GRID_NAME="${DATASET}_llama8b_main_grid_s${SAMPLES}"
MODEL_CONFIG="config/models/deepseek_r1_distill_llama8b.yaml"
DATA_CONFIG="config/datasets/${DATASET}.yaml"
read -r -a BUDGET_LIST <<< "${BUDGETS:-$DEFAULT_BUDGETS}"
read -r -a STRATEGY_LIST <<< "${GRID_STRATEGIES:-snapkv rkv}"

STRATEGIES=(fullkv)
ARM_BUDGETS=("")
for strategy in "${STRATEGY_LIST[@]}"; do
  for budget in "${BUDGET_LIST[@]}"; do
    STRATEGIES+=("$strategy")
    ARM_BUDGETS+=("$budget")
  done
done

require_python

export PYTHONPATH HF_HOME HF_ENDPOINT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  # Fail before any GPU job if the offline dataset cache is incomplete.
  "$PYTHON_BIN" - "$DATA_CONFIG" <<'PY'
import sys

import yaml
from kvbench.datasets import load_problems

with open(sys.argv[1]) as stream:
    data = yaml.safe_load(stream)["data"]
problems = load_problems(data["dataset"], int(data["n"]))
if len(problems) != int(data["n"]):
    raise SystemExit(f"expected {data['n']} cached {data['dataset']} problems, found {len(problems)}")
print(f"[dataset-check] dataset={data['dataset']} cached_problems={len(problems)}")
PY
fi

for index in "${!STRATEGIES[@]}"; do
  strategy="${STRATEGIES[$index]}"
  budget="${ARM_BUDGETS[$index]}"
  if [[ "$strategy" == "fullkv" ]]; then
    run_name="${DATASET}_llama8b_fullkv_s${SAMPLES}_multigpu"
  else
    run_name="${DATASET}_llama8b_${strategy}_b${budget}_s${SAMPLES}_multigpu"
  fi

  echo "[$GRID_NAME] starting $run_name"
  (
    unset RUN_DIR OUT
    export RUN_NAME="$run_name"
    export STRATEGY_CONFIG="config/strategies/${strategy}.yaml"
    export ARMS="${strategy}${budget:+@$budget}"
    if [[ "$strategy" == "fullkv" ]]; then
      export BATCH_SIZE="${FULLKV_BATCH_SIZE:-$DEFAULT_FULLKV_BATCH_SIZE}"
      ARM_ROWS="${FULLKV_ROWS_PER_BATCH:-$DEFAULT_FULLKV_ROWS}"
      ARM_PROBLEM_BATCH_SIZE="${FULLKV_PROBLEM_BATCH_SIZE:-}"
    else
      export BATCH_SIZE="${BATCH_SIZE:-4}"
      ARM_ROWS="${ROWS_PER_BATCH:-$(( 256 * 1024 / (budget > 1024 ? budget : 1024) ))}"
      ARM_PROBLEM_BATCH_SIZE="${PROBLEM_BATCH_SIZE:-}"
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "run_name=$RUN_NAME dataset=$DATASET strategy=$ARMS model_config=$MODEL_CONFIG data_config=$DATA_CONFIG strategy_config=$STRATEGY_CONFIG"
      echo "result=$RUNS_ROOT/$RUN_NAME/$RUN_NAME.json"
      exit 0
    fi

    RUN_DIR="$RUNS_ROOT/$RUN_NAME"
    OUT="$RUN_DIR/$RUN_NAME.json"
    LOG_MODE="${LOG_MODE:-brief}"
    mkdir -p "$RUN_DIR"
    CUDA_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
    NPROC_PER_NODE="${NPROC_PER_NODE:-$CUDA_COUNT}"
    if [[ -n "$ARM_PROBLEM_BATCH_SIZE" ]]; then
      PROBLEM_BATCH_SIZE="$ARM_PROBLEM_BATCH_SIZE"
    else
      CANDIDATES=$(( BATCH_SIZE < SAMPLES ? BATCH_SIZE : SAMPLES ))
      PER_GPU=$(( (NUM_PROBLEMS + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
      PROBLEM_BATCH_SIZE=$(( ARM_ROWS / CANDIDATES ))
      PROBLEM_BATCH_SIZE=$(( PROBLEM_BATCH_SIZE < PER_GPU ? PROBLEM_BATCH_SIZE : PER_GPU ))
      PROBLEM_BATCH_SIZE=$(( PROBLEM_BATCH_SIZE > 1 ? PROBLEM_BATCH_SIZE : 1 ))
    fi
    MASTER_PORT="${MASTER_PORT:-29500}"
    TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-720}"
    if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > CUDA_COUNT )); then
      echo "Invalid GPU count: requested=$NPROC_PER_NODE visible=$CUDA_COUNT" >&2
      exit 1
    fi
    EXTRA_ARGS=()
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
      EXTRA_ARGS+=(--overwrite)
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
      --num-return-sequences "$SAMPLES" \
      --problem-batch-size "$PROBLEM_BATCH_SIZE" \
      --log-mode "$LOG_MODE" \
      "${EXTRA_ARGS[@]}" \
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
  echo "[$GRID_NAME] completed $run_name result=$result_path"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[$GRID_NAME] dry run complete"
else
  echo "[$GRID_NAME] all runs complete under $RUNS_ROOT"
fi
