#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# The runner uses only the cached HuggingFaceH4/MATH-500 test split. Populate
# HF_DATASETS_CACHE first with scripts/bootstrap/download_dataset.sh if necessary.
SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
RUNS_ROOT="${RUNS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs}"
GRID_NAME="math500_llama8b_main_grid_s${SAMPLES}"
SUMMARY_DIR="${SUMMARY_DIR:-$RUNS_ROOT/$GRID_NAME}"
BUDGETS=(128 256 512 768 1024 1536 2048)

STRATEGIES=(fullkv)
ARM_BUDGETS=("")
for strategy in snapkv rkv; do
  for budget in "${BUDGETS[@]}"; do
    STRATEGIES+=("$strategy")
    ARM_BUDGETS+=("$budget")
  done
done

RESULT_PATHS=()
for index in "${!STRATEGIES[@]}"; do
  strategy="${STRATEGIES[$index]}"
  budget="${ARM_BUDGETS[$index]}"
  if [[ "$strategy" == "fullkv" ]]; then
    run_name="math500_llama8b_fullkv_s${SAMPLES}_multigpu"
  else
    run_name="math500_llama8b_${strategy}_b${budget}_s${SAMPLES}_multigpu"
  fi

  echo "[$GRID_NAME] starting $run_name"
  (
    unset RUN_NAME RUN_DIR OUTPUT_DIR RESULT_DIR LOG_FILE OUT CONFIG ARMS
    export NUM_RETURN_SEQUENCES="$SAMPLES" RUNS_ROOT
    if [[ "$strategy" == "fullkv" ]]; then
      export BATCH_SIZE="${FULLKV_BATCH_SIZE:-4}"
      bash "$ROOT_DIR/scripts/cluster/run_arm.sh" math500 "$strategy"
    else
      bash "$ROOT_DIR/scripts/cluster/run_arm.sh" math500 "$strategy" "$budget"
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
VENV_DIR="${VENV_DIR:-$HOME/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_results.py \
  "${RESULT_PATHS[@]}" \
  --out "$SUMMARY_DIR/summary.csv" \
  --json-out "$SUMMARY_DIR/summary.json"

echo "[$GRID_NAME] summary_csv=$SUMMARY_DIR/summary.csv"
echo "[$GRID_NAME] summary_json=$SUMMARY_DIR/summary.json"
