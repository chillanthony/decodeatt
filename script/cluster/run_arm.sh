#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 {aime24|math500} {fullkv|snapkv|rkv} [budget]" >&2
  exit 2
fi

DATASET_KEY="$1"
STRATEGY="$2"
BUDGET="${3:-}"
SAMPLES="${NUM_RETURN_SEQUENCES:-4}"

case "$DATASET_KEY" in
  aime24|math500) ;;
  *)
    echo "Unsupported dataset: $DATASET_KEY" >&2
    exit 2
    ;;
esac

case "$STRATEGY" in
  fullkv)
    if [[ -n "$BUDGET" ]]; then
      echo "FullKV does not accept a fixed KV budget" >&2
      exit 2
    fi
    STRATEGY_CONFIG="config/strategies/fullkv.yaml"
    ARMS="fullkv"
    DEFAULT_RUN_NAME="${DATASET_KEY}_llama8b_fullkv_s${SAMPLES}_multigpu"
    ;;
  snapkv|rkv)
    if [[ ! "$BUDGET" =~ ^[0-9]+$ ]] || (( BUDGET <= 8 )); then
      echo "$STRATEGY requires an integer budget greater than 8" >&2
      exit 2
    fi
    STRATEGY_CONFIG="config/strategies/${STRATEGY}.yaml"
    ARMS="${STRATEGY}@${BUDGET}"
    DEFAULT_RUN_NAME="${DATASET_KEY}_llama8b_${STRATEGY}_b${BUDGET}_s${SAMPLES}_multigpu"
    ;;
  *)
    echo "Unsupported strategy: $STRATEGY" >&2
    exit 2
    ;;
esac

DATA_CONFIG="config/datasets/${DATASET_KEY}.yaml"
MODEL_CONFIG="config/models/deepseek_r1_distill_llama8b.yaml"
export MODEL_CONFIG DATA_CONFIG STRATEGY_CONFIG ARMS
export NUM_RETURN_SEQUENCES="$SAMPLES"
export RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
export RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "run_name=$RUN_NAME"
  echo "data_config=$DATA_CONFIG"
  echo "model_config=$MODEL_CONFIG"
  echo "strategy_config=$STRATEGY_CONFIG"
  echo "dataset=$DATASET_KEY"
  if [[ "$DATASET_KEY" == "math500" ]]; then
    echo "dataset_source=offline-cache-only"
  fi
  echo "arms=$ARMS"
  echo "num_return_sequences=$NUM_RETURN_SEQUENCES"
  echo "result=$RUN_DIR/result/$RUN_NAME.json"
  exit 0
fi

exec bash "$ROOT_DIR/script/cluster/run_eval_multigpu.sh" "$DATASET_KEY"
