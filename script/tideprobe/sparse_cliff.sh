#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$ROOT_DIR"

require_python

export PYTHONPATH

CONFIG="${CONFIG:-$ROOT_DIR/configs/experiments/tideprobe_step1.yaml}"
RAW_DIR="${RAW_DIR:-$TIDEPROBE_ROOT/eviction_alignment_raw}"
EVENTS="${EVENTS:-$TIDEPROBE_ROOT/transition_events.jsonl}"
STRATEGIES="${STRATEGIES:-rkv,snapkv,window,random}"
BUDGETS="${BUDGETS:-512,1024,1536}"

echo "[tideprobe] sparse cliff strategies=$STRATEGIES budgets=$BUDGETS"
"$PYTHON_BIN" -m kvbench.diagnostics.sparse_cliff \
  --config "$CONFIG" \
  --raw-dir "$RAW_DIR" \
  --trace-dir "$TRACE_DIR" \
  --events "$EVENTS" \
  --strategies "$STRATEGIES" \
  --budgets "$BUDGETS"

echo "[tideprobe] sparse cliff complete"
