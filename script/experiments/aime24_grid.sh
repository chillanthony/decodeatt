#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$ROOT_DIR"

SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
GRID_NAME="aime24_llama8b_main_grid_s${SAMPLES}"
SUMMARY_DIR="${SUMMARY_DIR:-$RUNS_ROOT/$GRID_NAME}"

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
for index in "${!STRATEGIES[@]}"; do
  strategy="${STRATEGIES[$index]}"
  budget="${BUDGETS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  echo "[$GRID_NAME] starting $run_name via $strategy${budget:+@$budget}"
  (
    unset RUN_NAME RUN_DIR OUTPUT_DIR RESULT_DIR LOG_FILE OUT CONFIG ARMS
    export NUM_RETURN_SEQUENCES="$SAMPLES" RUNS_ROOT
    # fullkv has an unbounded cache; keep its micro-batch small to stay within
    # 80GB on the A100. Sparse-cache arms inherit the outer BATCH_SIZE.
    if [[ "$strategy" == "fullkv" ]]; then
      export BATCH_SIZE="${FULLKV_BATCH_SIZE:-8}"
      unset PROBLEM_BATCH_SIZE
    fi
    if [[ -n "$budget" ]]; then
      bash "$ROOT_DIR/scripts/cluster/run_arm.sh" aime24 "$strategy" "$budget"
    else
      bash "$ROOT_DIR/scripts/cluster/run_arm.sh" aime24 "$strategy"
    fi
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi

  result_path="$RUNS_ROOT/$run_name/result/$run_name.json"
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

"$PYTHON_BIN" scripts/summarize_results.py \
  "${RESULT_PATHS[@]}" \
  --out "$SUMMARY_DIR/summary.csv" \
  --json-out "$SUMMARY_DIR/summary.json"

echo "[$GRID_NAME] summary_csv=$SUMMARY_DIR/summary.csv"
echo "[$GRID_NAME] summary_json=$SUMMARY_DIR/summary.json"
