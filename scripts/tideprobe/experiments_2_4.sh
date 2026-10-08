#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/experiments/tideprobe_step1.yaml}"
EVENTS="${EVENTS:-$TIDEPROBE_ROOT/transition_events.jsonl}"

LAYER_OUT_DIR="${LAYER_OUT_DIR:-$TIDEPROBE_ROOT/layer_sensitivity}"
LAYER_BUDGETS="${LAYER_BUDGETS:-256,512}"
LAYER_MASK_POLICIES="${LAYER_MASK_POLICIES:-random,window}"
LAYER_RANDOM_SEEDS="${LAYER_RANDOM_SEEDS:-0,1,2}"
LAYER_PROBE_TYPES="${LAYER_PROBE_TYPES:-uniform,transition,ordinary}"
LAYER_RUN_TAG="${LAYER_RUN_TAG:-pilot}"

EVICTION_RAW_DIR="${EVICTION_RAW_DIR:-$TIDEPROBE_ROOT/eviction_alignment_raw}"
EVICTION_STRATEGIES="${EVICTION_STRATEGIES:-rkv,snapkv,window,random}"
EVICTION_BUDGETS="${EVICTION_BUDGETS:-512,1024,1536}"

if ! compgen -G "$TRACE_DIR/*.pt" >/dev/null; then
  echo "No trace files found in $TRACE_DIR" >&2
  exit 1
fi
if [[ ! -s "$EVENTS" ]]; then
  echo "Missing experiment 1 events: $EVENTS" >&2
  exit 1
fi

echo "[tideprobe serial] experiment 2/4: layer sensitivity pilot"
env \
  CONFIG="$CONFIG" \
  TRACE_DIR="$TRACE_DIR" \
  EVENTS="$EVENTS" \
  OUT_DIR="$LAYER_OUT_DIR" \
  SCAN_MODE=group \
  BUDGETS="$LAYER_BUDGETS" \
  MASK_POLICIES="$LAYER_MASK_POLICIES" \
  RANDOM_SEEDS="$LAYER_RANDOM_SEEDS" \
  PROBE_TYPES="$LAYER_PROBE_TYPES" \
  RUN_TAG="$LAYER_RUN_TAG" \
  bash "$ROOT_DIR/scripts/tideprobe/layer_sensitivity_8gpu.sh"

echo "[tideprobe serial] experiment 3/4: eviction alignment"
env \
  CONFIG="$CONFIG" \
  TRACE_DIR="$TRACE_DIR" \
  EVENTS="$EVENTS" \
  RAW_DIR="$EVICTION_RAW_DIR" \
  STRATEGIES="$EVICTION_STRATEGIES" \
  BUDGETS="$EVICTION_BUDGETS" \
  bash "$ROOT_DIR/scripts/tideprobe/eviction_alignment_8gpu.sh"

echo "[tideprobe serial] experiment 4/4: sparse cliff"
env \
  CONFIG="$CONFIG" \
  TRACE_DIR="$TRACE_DIR" \
  EVENTS="$EVENTS" \
  RAW_DIR="$EVICTION_RAW_DIR" \
  STRATEGIES="$EVICTION_STRATEGIES" \
  BUDGETS="$EVICTION_BUDGETS" \
  bash "$ROOT_DIR/scripts/tideprobe/sparse_cliff.sh"

echo "[tideprobe serial] experiments 2-4 complete"
