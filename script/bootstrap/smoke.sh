#!/usr/bin/env bash
# Environment smoke test: CUDA visible, unit tests pass, one short eval writes a result.
#   SMOKE_MODEL_CONFIG (default: R1-Distill-Llama-8B), SMOKE_MAX_NEW, SMOKE_BUDGET, SMOKE_OUT
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ENV_FILE:-$ROOT_DIR/config/runtime/4090.env}"
require_python
cd "$ROOT_DIR"

MODEL_CONFIG="${SMOKE_MODEL_CONFIG:-config/models/deepseek_r1_distill_llama8b.yaml}"
OUT="${SMOKE_OUT:-$ROOT_DIR/runs/smoke/eval.json}"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"[smoke] torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'
"$PYTHON_BIN" -m pytest -q test

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" script/eval.py \
  --model-config "$MODEL_CONFIG" \
  --data-config config/datasets/sample.yaml \
  --strategy-config config/strategies/rkv.yaml \
  --n 1 --max-new "${SMOKE_MAX_NEW:-32}" --arms "rkv@${SMOKE_BUDGET:-64}" \
  --recent 8 --evict-every 8 --greedy --log-mode brief --overwrite --out "$OUT"

"$PYTHON_BIN" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]; assert s and all(a["total"] > 0 for a in s.values()), s; print("[smoke] PASS", sys.argv[1])' "$OUT"
