#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/a800.env}"
source "$ENV_FILE"

if (( $# != 1 )); then
  echo "Usage: $0 {aime24|math500}" >&2
  exit 2
fi

DATASET_KEY="$1"
case "$DATASET_KEY" in
  aime24)
    DEFAULT_DATASET_REPO="Maxwell-Jia/AIME_2024"
    DEFAULT_SPLIT="train"
    DEFAULT_BUDGET=1536
    ;;
  math500)
    DEFAULT_DATASET_REPO="HuggingFaceH4/MATH-500"
    DEFAULT_SPLIT="test"
    DEFAULT_BUDGET=1024
    ;;
  *)
    echo "Unsupported dataset: $DATASET_KEY" >&2
    exit 2
    ;;
esac

require_python

DATASET_REPO="${DATASET_REPO:-$DEFAULT_DATASET_REPO}"
SPLIT="${SPLIT:-$DEFAULT_SPLIT}"
REVISION="${REVISION:-main}"
LOG_FILE="${LOG_FILE:-$HF_HOME/logs/${DATASET_KEY}-download.log}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[download-dataset] root=$ROOT_DIR"
echo "[download-dataset] dataset=$DATASET_KEY"
echo "[download-dataset] dataset_repo=$DATASET_REPO"
echo "[download-dataset] split=$SPLIT"
echo "[download-dataset] revision=$REVISION"
echo "[download-dataset] endpoint=$HF_ENDPOINT"
echo "[download-dataset] hf_home=$HF_HOME"
echo "[download-dataset] datasets_cache=$HF_DATASETS_CACHE"
echo "[download-dataset] log_file=$LOG_FILE"
echo "[download-dataset] python_bin=$PYTHON_BIN"
echo "[download-dataset] proxy=inherited"
echo "[download-dataset] ssl_verify=false"

env \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HUB_OFFLINE=0 \
  HF_DATASETS_OFFLINE=0 \
  "$PYTHON_BIN" - "$DATASET_REPO" "$SPLIT" "$REVISION" <<'PY'
from __future__ import annotations

import sys

import httpx
from datasets import load_dataset
from huggingface_hub import set_client_factory

# huggingface_hub >= 1.0 uses httpx (configure_http_backend/requests were removed).
# Keep the default client's settings and only disable SSL verification.
try:
    from huggingface_hub.utils._http import hf_request_event_hook
    event_hooks = {"request": [hf_request_event_hook]}
except ImportError:  # private helper; fall back to a plain client if it moves
    event_hooks = {}

set_client_factory(lambda: httpx.Client(
    verify=False, event_hooks=event_hooks, follow_redirects=True, timeout=None,
))

repo, split, revision = sys.argv[1:]
dataset = load_dataset(repo, split=split, revision=revision)
print(f"[download-dataset] downloaded_rows={len(dataset)}")
print(f"[download-dataset] columns={dataset.column_names}")
print(f"[download-dataset] cache_files={dataset.cache_files}")
PY

echo "[download-dataset] done"
echo "Dataset cache is ready; run the corresponding grid under script/experiments/."
