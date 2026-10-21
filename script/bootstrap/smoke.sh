#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_OVERRIDE="${MODEL:-}"
source "$ROOT_DIR/script/env.sh"
require_python

MODEL_PATH="$MODEL_OVERRIDE"
if [[ -z "$MODEL_PATH" ]]; then
  echo "Set MODEL to the local directory of a downloaded Hugging Face model." >&2
  echo "Example: MODEL=/path/to/DeepSeek-R1-Distill-Llama-8B bash script/bootstrap/smoke.sh" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "MODEL must point to a local model directory containing config.json: $MODEL_PATH" >&2
  exit 2
fi

N="${SMOKE_N:-1}"
MAX_NEW="${SMOKE_MAX_NEW:-32}"
BUDGET="${SMOKE_BUDGET:-64}"
EVICT_EVERY="${SMOKE_EVICT_EVERY:-8}"
OUT="${SMOKE_OUT:-$ROOT_DIR/runs/smoke/eval.json}"

if [[ ! "$N" =~ ^[1-9][0-9]*$ || ! "$MAX_NEW" =~ ^[1-9][0-9]*$ \
   || ! "$BUDGET" =~ ^[1-9][0-9]*$ || ! "$EVICT_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_N, SMOKE_MAX_NEW, SMOKE_BUDGET and SMOKE_EVICT_EVERY must be positive integers." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
echo "[smoke] python=$PYTHON_BIN"
echo "[smoke] model=$MODEL_PATH"
echo "[smoke] dataset=sample n=$N max_new=$MAX_NEW arms=fullkv,rkv@$BUDGET"
echo "[smoke] output=$OUT"

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; this smoke script expects the target GPU environment.")
print(f"[smoke] torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

"$PYTHON_BIN" -m pytest -q test

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  "$PYTHON_BIN" "$ROOT_DIR/script/eval.py" \
  --config "$ROOT_DIR/config/experiments/throughput_smoke.yaml" \
  --model "$MODEL_PATH" \
  --dataset sample \
  --n "$N" \
  --max-new "$MAX_NEW" \
  --arms "fullkv,rkv@$BUDGET" \
  --recent 8 \
  --evict-every "$EVICT_EVERY" \
  --batch-size 1 \
  --problem-batch-size 1 \
  --dtype "${SMOKE_DTYPE:-bfloat16}" \
  --device-map cuda \
  --attn auto \
  --greedy \
  --log-mode brief \
  --out "$OUT"

"$PYTHON_BIN" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Smoke test did not produce result file: {path}")
payload = json.loads(path.read_text())
summary = payload.get("summary", {})
missing = {"fullkv", next((key for key in summary if key.startswith("rkv@")), "rkv")} - set(summary)
if missing:
    raise SystemExit(f"Smoke result is missing expected arms {sorted(missing)}: {path}")
if any(int(arm.get("total", 0)) <= 0 for arm in summary.values()):
    raise SystemExit(f"Smoke result contains no evaluated samples: {path}")
print(f"[smoke] PASS: eval.py ran and wrote valid results to {path}")
PY
