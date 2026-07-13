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
export MODEL

mkdir -p "$HF_HOME"

echo "[download-model] root=$ROOT_DIR"
echo "[download-model] model=$MODEL"
echo "[download-model] hf_home=$HF_HOME"
echo "[download-model] hf_endpoint=$HF_ENDPOINT"

python - <<'PY'
import os
import warnings

import requests
from huggingface_hub import snapshot_download
from huggingface_hub import configure_http_backend
from urllib3.exceptions import InsecureRequestWarning


def backend_factory():
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

model = os.environ["MODEL"]
path = snapshot_download(repo_id=model)
print(f"[download-model] downloaded_to={path}")
PY
