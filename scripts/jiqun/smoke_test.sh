#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_FILE="${LOG_FILE:-$ROOT_DIR/scripts/jiqun/smoke_test.log}"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

VENV_DIR="${VENV_DIR:-/home/ma-user/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL="${MODEL:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache/models/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-configs/experiments/math500_official_b1024.yaml}"
OUT="${OUT:-results/smoke.json}"
ARMS="${ARMS:-fullkv,rkv@128}"
MAX_NEW="${MAX_NEW:-128}"
N="${N:-1}"

mkdir -p "$(dirname "$OUT")"

echo "[smoke] root=$ROOT_DIR"
echo "[smoke] model=$MODEL"
echo "[smoke] hf_home=$HF_HOME"
echo "[smoke] out=$OUT"
echo "[smoke] log_file=$LOG_FILE"
echo "[smoke] python_bin=$PYTHON_BIN"

"$PYTHON_BIN" scripts/eval.py \
  --config "$CONFIG" \
  --dataset sample \
  --n "$N" \
  --max-new "$MAX_NEW" \
  --arms "$ARMS" \
  --model "$MODEL" \
  --out "$OUT"
