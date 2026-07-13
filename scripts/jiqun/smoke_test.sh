#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run 'uv sync' in $ROOT_DIR first." >&2
  exit 1
fi

source .venv/bin/activate

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
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

python scripts/eval.py \
  --config "$CONFIG" \
  --dataset sample \
  --n "$N" \
  --max-new "$MAX_NEW" \
  --arms "$ARMS" \
  --model "$MODEL" \
  --out "$OUT"
