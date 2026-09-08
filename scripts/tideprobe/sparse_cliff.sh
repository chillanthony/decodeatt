#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/experiments/tideprobe_step1.yaml}"
RAW_DIR="${RAW_DIR:-runs/tideprobe_step1/eviction_alignment_raw}"
TRACE_DIR="${TRACE_DIR:-/home/ma-user/work/bucket-wulan-green/chenyanbo/trace}"
EVENTS="${EVENTS:-runs/tideprobe_step1/transition_events.jsonl}"
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
