#!/usr/bin/env bash
# Single-GPU maximum throughput of FullKV vs R-KV (R-KV paper Table 1 protocol); see script/speedtest.py.
#
#   bash script/experiments/speedtest.sh
#   GEN_LEN=4096 bash script/experiments/speedtest.sh          # shorter run (about half the time)
#   GPUS="2 3" bash script/experiments/speedtest.sh            # pick the GPUs
#
# With at least as many GPUs as strategies (GPUS, else CUDA_VISIBLE_DEVICES, else all visible), every strategy
# runs as its own process on its own GPU at the same time; otherwise they run one after another on one GPU.
# Each strategy: probe a small batch for memory per row, estimate the largest batch that fits, measure there.
# Results: $OUT (all strategies) plus one <OUT stem>.<strategy>.json per strategy in parallel mode.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/a800.env}"
source "$ENV_FILE"
require_python
cd "$ROOT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

OUT="${OUT:-$RUNS_ROOT/speedtest/speedtest_$(date +%Y%m%d_%H%M%S).json}"
read -r -a STRATEGY_LIST <<< "${STRATEGIES:-fullkv rkv}"
COMMON=(--budget "${BUDGET:-1024}" --gen-len "${GEN_LEN:-8192}")

if [[ -n "${GPUS:-}" ]]; then
  read -r -a GPU_LIST <<< "$GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
  read -r -a GPU_LIST <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
fi

if (( ${#GPU_LIST[@]} < ${#STRATEGY_LIST[@]} )); then
  echo "[speedtest] ${#GPU_LIST[@]} GPU(s) for ${#STRATEGY_LIST[@]} strategies: running them in sequence"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]:-0}" "$PYTHON_BIN" script/speedtest.py \
    --strategies "${STRATEGY_LIST[*]}" "${COMMON[@]}" --out "$OUT" "$@"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
PIDS=()
PARTS=()
for i in "${!STRATEGY_LIST[@]}"; do
  strategy="${STRATEGY_LIST[$i]}"
  part="${OUT%.json}.${strategy}.json"
  PARTS+=("$part")
  echo "[speedtest] $strategy on GPU ${GPU_LIST[$i]} -> $part"
  (
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" "$PYTHON_BIN" script/speedtest.py \
      --strategies "$strategy" "${COMMON[@]}" --out "$part" "$@" 2>&1 | sed -u "s/^/[$strategy] /"
  ) &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done
if (( FAILED )); then
  echo "[speedtest] a strategy failed; see its log lines above" >&2
  exit 1
fi

# Merge the per-strategy files and compare every eviction strategy with FullKV.
"$PYTHON_BIN" - "$OUT" "${PARTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out, parts = Path(sys.argv[1]), [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
rows = [row for part in parts for row in part["rows"]]
out.write_text(json.dumps({"gpu": parts[0]["gpu"], "args": [p["args"] for p in parts], "rows": rows}, indent=1))
best = {row["arm"]: row for row in rows if row["run"] in ("max", "fixed")}
full = best.get("fullkv")
for name, row in best.items():
    print(f"[speedtest] {name:12s} batch {row['batch']:5d}  {row['tokens_per_sec']:8.0f} tok/s  "
          f"{row['ms_per_step']:7.1f} ms/step  peak {row['peak_memory_bytes'] / 2 ** 30:5.1f} GiB")
    if full and name != "fullkv":
        print(f"[speedtest] {name} vs fullkv: batch {row['batch'] / full['batch']:.1f}x, "
              f"throughput {row['tokens_per_sec'] / full['tokens_per_sec']:.1f}x")
print(f"[speedtest] wrote {out}")
PY
