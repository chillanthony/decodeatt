#!/usr/bin/env bash
# Shared repository and runtime defaults for all shell launchers.

if [[ -z "${ROOT_DIR:-}" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export ROOT_DIR

VENV_DIR="${VENV_DIR:-$HOME/.venvs/decodeatt}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
SFS_ROOT="${SFS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
HF_HOME="${HF_HOME:-$SFS_ROOT}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs}"
MODEL="${MODEL:-$HF_HOME/models/DeepSeek-R1-Distill-Llama-8B}"
TIDEPROBE_ROOT="${TIDEPROBE_ROOT:-$RUNS_ROOT/tideprobe_step1}"
TRACE_DIR="${TRACE_DIR:-$(dirname "$RUNS_ROOT")/trace}"

export VENV_DIR PYTHON_BIN SFS_ROOT HF_HOME HF_ENDPOINT HF_DATASETS_CACHE RUNS_ROOT MODEL TIDEPROBE_ROOT TRACE_DIR

require_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing executable python: $PYTHON_BIN" >&2
    exit 1
  fi
}

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
