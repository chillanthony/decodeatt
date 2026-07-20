#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python: $PYTHON_BIN" >&2
  exit 1
fi

DATASET_REPO="${DATASET_REPO:-HuggingFaceH4/MATH-500}"
SPLIT="${SPLIT:-test}"
REVISION="${REVISION:-main}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SFS_ROOT="${SFS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
HF_HOME="${HF_HOME:-$SFS_ROOT}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
LOG_FILE="${LOG_FILE:-$HF_HOME/logs/math500-download.log}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[download-math500] root=$ROOT_DIR"
echo "[download-math500] dataset_repo=$DATASET_REPO"
echo "[download-math500] split=$SPLIT"
echo "[download-math500] revision=$REVISION"
echo "[download-math500] endpoint=$HF_ENDPOINT"
echo "[download-math500] hf_home=$HF_HOME"
echo "[download-math500] datasets_cache=$HF_DATASETS_CACHE"
echo "[download-math500] log_file=$LOG_FILE"
echo "[download-math500] python_bin=$PYTHON_BIN"
echo "[download-math500] proxy=inherited"
echo "[download-math500] ssl_verify=false"

env \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HUB_OFFLINE=0 \
  HF_DATASETS_OFFLINE=0 \
  "$PYTHON_BIN" - "$DATASET_REPO" "$SPLIT" "$REVISION" <<'PY'
from __future__ import annotations

import sys

import requests
from datasets import load_dataset
from huggingface_hub import configure_http_backend
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


repo, split, revision = sys.argv[1:]
disable_warnings(InsecureRequestWarning)
configure_http_backend(backend_factory=backend_factory)
dataset = load_dataset(repo, split=split, revision=revision)
print(f"[download-math500] downloaded_rows={len(dataset)}")
print(f"[download-math500] columns={dataset.column_names}")
print(f"[download-math500] cache_files={dataset.cache_files}")
PY

echo "[download-math500] done"
echo "Run offline with: scripts/jiqun/math500_official_b1024_n20.sh"
