#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/4090.env}"
source "$ENV_FILE"
require_python

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
echo "[smoke] model_config=$ROOT_DIR/config/models/deepseek_r1_distill_llama8b.yaml"
echo "[smoke] dataset=sample n=$N max_new=$MAX_NEW arm=rkv@$BUDGET"
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
  --model-config "$ROOT_DIR/config/models/deepseek_r1_distill_llama8b.yaml" \
  --data-config "$ROOT_DIR/config/datasets/sample.yaml" \
  --strategy-config "$ROOT_DIR/config/strategies/rkv.yaml" \
  --dataset sample \
  --n "$N" \
  --max-new "$MAX_NEW" \
  --arms "rkv@$BUDGET" \
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
missing = {next((key for key in summary if key.startswith("rkv@")), "rkv")} - set(summary)
if missing:
    raise SystemExit(f"Smoke result is missing expected arms {sorted(missing)}: {path}")
if any(int(arm.get("total", 0)) <= 0 for arm in summary.values()):
    raise SystemExit(f"Smoke result contains no evaluated samples: {path}")
print(f"[smoke] PASS: eval.py ran and wrote valid results to {path}")
PY
