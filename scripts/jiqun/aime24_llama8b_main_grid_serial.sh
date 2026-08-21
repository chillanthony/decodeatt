#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
RUNS_ROOT="${RUNS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs}"
GRID_NAME="aime24_llama8b_main_grid_s${SAMPLES}"
SUMMARY_DIR="${SUMMARY_DIR:-$RUNS_ROOT/$GRID_NAME}"

SCRIPTS=(
  scripts/jiqun/aime24_llama8b_fullkv.sh
  scripts/jiqun/aime24_llama8b_snapkv_b1024.sh
  scripts/jiqun/aime24_llama8b_snapkv_b1536.sh
  scripts/jiqun/aime24_llama8b_snapkv_b2048.sh
  scripts/jiqun/aime24_llama8b_snapkv_b2560.sh
  scripts/jiqun/aime24_llama8b_rkv_b1024.sh
  scripts/jiqun/aime24_llama8b_rkv_b1536.sh
  scripts/jiqun/aime24_llama8b_rkv_b2048.sh
  scripts/jiqun/aime24_llama8b_rkv_b2560.sh
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
for index in "${!SCRIPTS[@]}"; do
  script="${SCRIPTS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  echo "[$GRID_NAME] starting $run_name via $script"
  (
    unset RUN_NAME RUN_DIR OUTPUT_DIR RESULT_DIR LOG_FILE OUT CONFIG ARMS
    export NUM_RETURN_SEQUENCES="$SAMPLES" RUNS_ROOT
    # fullkv has an unbounded cache; keep its micro-batch small to stay within
    # 80GB on the A100. Sparse-cache arms inherit the outer BATCH_SIZE.
    if [[ "$script" == *fullkv* ]]; then
      export BATCH_SIZE="${FULLKV_BATCH_SIZE:-8}"
      unset PROBLEM_BATCH_SIZE
    fi
    bash "$ROOT_DIR/$script"
  )
  result_path="$RUNS_ROOT/$run_name/result/$run_name.json"
  if [[ ! -s "$result_path" ]]; then
    echo "Missing result after successful run: $result_path" >&2
    exit 1
  fi
  RESULT_PATHS+=("$result_path")
  echo "[$GRID_NAME] completed $run_name result=$result_path"
done

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
