#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HFD_SCRIPT="$ROOT_DIR/scripts/jiqun/hfd.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
REVISION="${REVISION:-main}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
THREADS="${THREADS:-4}"
CONCURRENT="${CONCURRENT:-2}"
SFS_ROOT="${SFS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
MODEL_DIR="${MODEL_DIR:-$SFS_ROOT/models/DeepSeek-R1-Distill-Llama-8B}"
LOG_FILE="${LOG_FILE:-$MODEL_DIR/hfd.log}"

if [[ ! -f "$HFD_SCRIPT" ]]; then
  echo "Missing downloader: $HFD_SCRIPT" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"

echo "[run-hfd] model=$MODEL"
echo "[run-hfd] revision=$REVISION"
echo "[run-hfd] endpoint=$HF_ENDPOINT"
echo "[run-hfd] model_dir=$MODEL_DIR"
echo "[run-hfd] tool=aria2c"
echo "[run-hfd] threads=$THREADS"
echo "[run-hfd] concurrent=$CONCURRENT"
echo "[run-hfd] log_file=$LOG_FILE"
echo "[run-hfd] proxy=inherited"
echo "[run-hfd] warning=SSL certificate verification is disabled" >&2

env HF_ENDPOINT="$HF_ENDPOINT" \
  bash "$HFD_SCRIPT" "$MODEL" \
  --tool aria2c \
  -x "$THREADS" \
  -j "$CONCURRENT" \
  --local-dir "$MODEL_DIR" \
  --revision "$REVISION" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "[run-hfd] done"
echo "Run the smoke test with: MODEL=$MODEL_DIR scripts/jiqun/smoke_test.sh"
