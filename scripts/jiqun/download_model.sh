#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run 'uv sync' in $ROOT_DIR first." >&2
  exit 1
fi

source .venv/bin/activate

export HF_HOME="${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"

mkdir -p "$HF_HOME"

echo "[download-model] root=$ROOT_DIR"
echo "[download-model] model=$MODEL"
echo "[download-model] hf_home=$HF_HOME"
echo "[download-model] hf_endpoint=$HF_ENDPOINT"

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL"
else
  python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download

model = sys.argv[1]
path = snapshot_download(repo_id=model)
print(f"[download-model] downloaded_to={path}")
PY
fi
