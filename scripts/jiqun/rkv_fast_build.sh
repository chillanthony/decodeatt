#!/usr/bin/env bash
#
# rkv_fast_build.sh — build the patched-vLLM R-KV tree for the "fast" branch.
#
# Pinned upstream vLLM v0.25.1 (commit 752a3a50...), exactly as zefancai/R-KV
# vendors it. Copies efficiency/src/rkv/* into the tree, applies efficiency/patch/
# rkv-vllm-0.25.1.patch, and pip-installs the tree editable in an isolated venv
# (so the build never touches the decodeatt research env).
#
# Usage:
#   bash scripts/jiqun/rkv_fast_build.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EFF="$ROOT_DIR/efficiency"

# --- Pinned reference (do not change casually; the patch is generated against it). ---
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_TAG="${VLLM_TAG:-v0.25.1}"
VLLM_COMMIT="${VLLM_COMMIT:-752a3a504485790a2e8491cacbb35c137339ad34}"

# Keep the patched source tree and its environment on persistent storage.
VLLM_BUILD_SRC="/home/ma-user/work/decodeatt-vllm-src"
RKV_VENV="/home/ma-user/.venvs/rkv-fast/bin/python"
PYTHON_BIN="/home/ma-user/.venv/decodeatt/bin/python"
PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
VLLM_SRC="$VLLM_BUILD_SRC"
PATCH="$EFF/patch/rkv-vllm-0.25.1.patch"
RKV_SRC="$EFF/src/rkv"
VENV_BIN="$RKV_VENV"
VENV="$(dirname "$(dirname "$VENV_BIN")")"   # strip /bin/python -> venv root

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "vllm_src=$VLLM_SRC"
  echo "patch=$PATCH"
  echo "rkv_src=$RKV_SRC"
  echo "venv_bin=$VENV_BIN"
  echo "venv_root=$VENV"
  exit 0
fi

# --- Preflight: a source build needs a GPU + CUDA toolchain. ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This is a CUDA source build; run on a GPU node." >&2
  if [[ "${VLLM_BUILD_ALLOW_HOST:-0}" != "1" ]]; then
    exit 1
  fi
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "WARNING: CUDA_HOME not set; vLLM's build may not find cuDNN/its toolchain." >&2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -e "$VLLM_SRC" ]]; then
  echo "ERROR: $VLLM_SRC already exists. Remove it before rebuilding." >&2
  exit 1
fi

echo ">> [1/4] Cloning vLLM @ $VLLM_TAG ($VLLM_COMMIT)"
git clone --depth 1 --branch "$VLLM_TAG" "$VLLM_REPO" "$VLLM_SRC"
GOT="$(git -C "$VLLM_SRC" rev-parse HEAD)"
if [[ "$GOT" != "$VLLM_COMMIT" ]]; then
  echo "ERROR: tag $VLLM_TAG resolved to $GOT, expected $VLLM_COMMIT." >&2
  exit 1
fi
echo "         checked out $(git -C "$VLLM_SRC" rev-parse --short HEAD)"

echo ">> [2/4] Installing R-KV package into the vLLM tree"
DEST="$VLLM_SRC/vllm/rkv"
mkdir -p "$DEST"
cp "$RKV_SRC"/*.py "$DEST"/
echo "         copied $(ls "$RKV_SRC"/*.py | wc -l | tr -d ' ') files -> $DEST"

echo ">> [3/4] Applying R-KV wiring patch (13 vLLM files)"
git -C "$VLLM_SRC" apply --check "$PATCH"
git -C "$VLLM_SRC" apply --whitespace=nowarn "$PATCH"
echo "         patch applied cleanly"

echo ">> [4/4] Creating venv + installing patched vLLM (this is the long step)"
"$PYTHON_BIN" -m venv "$VENV"
export PIP_INDEX_URL
unset PIP_NO_INDEX
"$VENV/bin/pip" install --upgrade pip
# vLLM pins its torch/CUDA; install the tree's own requirements first so the
# editable install resolves against the intended versions, then the tree.
"$VENV/bin/pip" install -e "$VLLM_SRC"
echo "         installed"

cat <<EOF

Done. Patched vLLM tree at:
  $VLLM_SRC
venv (activate before benchmarking):
  source $VENV/bin/activate

Next: run the benchmark.
  bash scripts/jiqun/rkv_fast_bench_single.sh
  bash scripts/jiqun/rkv_fast_bench_dp8.sh
EOF
