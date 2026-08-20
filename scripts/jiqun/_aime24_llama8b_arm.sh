#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 {fullkv|snapkv|rkv} [budget]" >&2
  exit 2
fi

STRATEGY="$1"
BUDGET="${2:-}"
SAMPLES="${NUM_RETURN_SEQUENCES:-4}"
RUNS_ROOT="${RUNS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs}"

case "$STRATEGY" in
  fullkv)
    if [[ -n "$BUDGET" ]]; then
      echo "FullKV does not accept a fixed KV budget" >&2
      exit 2
    fi
    CONFIG="configs/onestrategy/fullkv.yaml"
    ARMS="fullkv"
    DEFAULT_RUN_NAME="aime24_llama8b_fullkv_s${SAMPLES}_multigpu"
    ;;
  snapkv|rkv)
    if [[ ! "$BUDGET" =~ ^[0-9]+$ ]] || (( BUDGET <= 8 )); then
      echo "$STRATEGY requires an integer budget greater than 8" >&2
      exit 2
    fi
    CONFIG="configs/onestrategy/${STRATEGY}.yaml"
    ARMS="${STRATEGY}@${BUDGET}"
    DEFAULT_RUN_NAME="aime24_llama8b_${STRATEGY}_b${BUDGET}_s${SAMPLES}_multigpu"
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
  echo "arms=$ARMS"
  echo "num_return_sequences=$NUM_RETURN_SEQUENCES"
  echo "result=$RUN_DIR/result/$RUN_NAME.json"
  exit 0
fi

exec bash "$ROOT_DIR/scripts/jiqun/aime24_rkv_b1536_s4_multigpu.sh"
