#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run 'uv sync' in $ROOT_DIR first." >&2
  exit 1
fi

source .venv/bin/activate

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
REVISION="${REVISION:-main}"
MAX_WORKERS="${MAX_WORKERS:-1}"
SFS_HF_HOME="${SFS_HF_HOME:-${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}}"

export HF_HOME="$SFS_HF_HOME"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export MODEL REVISION MAX_WORKERS

mkdir -p "$HF_HUB_CACHE"
if [[ ! -w "$HF_HUB_CACHE" ]]; then
  echo "HF cache is not writable: $HF_HUB_CACHE" >&2
  exit 1
fi

echo "[download-model-sfs] model=$MODEL"
echo "[download-model-sfs] revision=$REVISION"
echo "[download-model-sfs] hf_home=$HF_HOME"
echo "[download-model-sfs] hf_hub_cache=$HF_HUB_CACHE"
echo "[download-model-sfs] endpoint=$HF_ENDPOINT"
echo "[download-model-sfs] disable_xet=$HF_HUB_DISABLE_XET"
echo "[download-model-sfs] max_workers=$MAX_WORKERS"
echo "[download-model-sfs] download_timeout=$HF_HUB_DOWNLOAD_TIMEOUT"
echo "[download-model-sfs] warning=SSL certificate verification is disabled" >&2

python - <<'PY'
import os
import warnings

import requests
from huggingface_hub import configure_http_backend, snapshot_download
from urllib3.exceptions import InsecureRequestWarning


def backend_factory():
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

path = snapshot_download(
    repo_id=os.environ["MODEL"],
    revision=os.environ["REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
    max_workers=int(os.environ["MAX_WORKERS"]),
)
print(f"[download-model-sfs] downloaded_to={path}")
PY

echo "[download-model-sfs] done"
echo "Run the smoke test with: HF_HOME=$HF_HOME scripts/jiqun/smoke_test.sh"
