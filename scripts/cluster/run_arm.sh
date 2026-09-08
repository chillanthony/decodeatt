#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 {aime24|math500} {fullkv|snapkv|rkv} [budget]" >&2
  exit 2
fi

DATASET_KEY="$1"
STRATEGY="$2"
BUDGET="${3:-}"
SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
RUNS_ROOT="${RUNS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs}"

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
    CONFIG="configs/onestrategy/fullkv.yaml"
    ARMS="fullkv"
    DEFAULT_RUN_NAME="${DATASET_KEY}_llama8b_fullkv_s${SAMPLES}_multigpu"
    ;;
  snapkv|rkv)
    if [[ ! "$BUDGET" =~ ^[0-9]+$ ]] || (( BUDGET <= 8 )); then
      echo "$STRATEGY requires an integer budget greater than 8" >&2
      exit 2
    fi
    CONFIG="configs/onestrategy/${STRATEGY}.yaml"
    ARMS="${STRATEGY}@${BUDGET}"
    DEFAULT_RUN_NAME="${DATASET_KEY}_llama8b_${STRATEGY}_b${BUDGET}_s${SAMPLES}_multigpu"
    ;;
  *)
    echo "Unsupported strategy: $STRATEGY" >&2
    exit 2
    ;;
esac

export CONFIG ARMS
export NUM_RETURN_SEQUENCES="$SAMPLES"
export RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
export RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "run_name=$RUN_NAME"
  echo "config=$CONFIG"
  echo "dataset=$DATASET_KEY"
  if [[ "$DATASET_KEY" == "math500" ]]; then
    echo "dataset_source=offline-cache-only"
  fi
  echo "arms=$ARMS"
  echo "num_return_sequences=$NUM_RETURN_SEQUENCES"
  echo "result=$RUN_DIR/result/$RUN_NAME.json"
  exit 0
fi

exec bash "$ROOT_DIR/scripts/cluster/run_eval_multigpu.sh" "$DATASET_KEY"
