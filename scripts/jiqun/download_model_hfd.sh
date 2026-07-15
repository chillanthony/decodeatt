#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HFD_SCRIPT="$ROOT_DIR/scripts/jiqun/hfd.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
MODEL_ROOT="${MODEL_ROOT:-/home/ma-user/work/sfsturbo/models}"
LOCAL_DIR="${LOCAL_DIR:-$MODEL_ROOT/${MODEL##*/}}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
TOOL="${TOOL:-aria2c}"
THREADS="${THREADS:-4}"
JOBS="${JOBS:-2}"
REVISION="${REVISION:-main}"

if [[ ! -f "$HFD_SCRIPT" ]]; then
    echo "Missing downloader: $HFD_SCRIPT" >&2
    exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "[download-model-hfd] model=$MODEL"
echo "[download-model-hfd] local_dir=$LOCAL_DIR"
echo "[download-model-hfd] endpoint=$HF_ENDPOINT"
echo "[download-model-hfd] tool=$TOOL threads=$THREADS jobs=$JOBS"
echo "[download-model-hfd] revision=$REVISION"

exec env HF_ENDPOINT="$HF_ENDPOINT" \
    bash "$HFD_SCRIPT" "$MODEL" \
    --tool "$TOOL" \
    -x "$THREADS" \
    -j "$JOBS" \
    --local-dir "$LOCAL_DIR" \
    --revision "$REVISION" \
    "$@"
