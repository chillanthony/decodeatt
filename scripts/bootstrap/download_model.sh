#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HFD_SCRIPT="$ROOT_DIR/scripts/bootstrap/hfd.sh"

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

echo "[download-model] model=$MODEL"
echo "[download-model] revision=$REVISION"
echo "[download-model] endpoint=$HF_ENDPOINT"
echo "[download-model] model_dir=$MODEL_DIR"
echo "[download-model] tool=aria2c"
echo "[download-model] threads=$THREADS"
echo "[download-model] concurrent=$CONCURRENT"
echo "[download-model] log_file=$LOG_FILE"
echo "[download-model] proxy=inherited"
echo "[download-model] warning=SSL certificate verification is disabled" >&2

env HF_ENDPOINT="$HF_ENDPOINT" \
  bash "$HFD_SCRIPT" "$MODEL" \
  --tool aria2c \
  -x "$THREADS" \
  -j "$CONCURRENT" \
  --local-dir "$MODEL_DIR" \
  --revision "$REVISION" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "[download-model] done"
echo "Model directory: $MODEL_DIR"
