#!/usr/bin/env bash
# 2026-11-11 quick baseline grid on MATH-500: FullKV + SnapKV / R-KV / H2O / StreamingLLM at budget 512,
# one sample per problem, then a one-line-per-arm table of the results.
#
#   bash script/experiments/gridtest-1111.sh          # run the grid, then print the table
#   TABLE_ONLY=1 bash script/experiments/gridtest-1111.sh   # only print the table
#
# Concurrency: PROBLEM_BATCH_SIZE problems per GPU at once (default 32). A batch that hits CUDA OOM is
# split in half and retried, and the size that fit is kept for later batches, so a large value is safe.
# Other run_grid.sh overrides (NPROC_PER_NODE, OVERWRITE, DRY_RUN, ...) pass through.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/a800.env}"
source "$ENV_FILE"
export ENV_FILE
cd "$ROOT_DIR"

export GRID_STRATEGIES="snapkv rkv h2o streamingllm"
export BUDGETS="512"
export NUM_RETURN_SEQUENCES=1
export PROBLEM_BATCH_SIZE="${PROBLEM_BATCH_SIZE:-32}"

if [[ "${TABLE_ONLY:-0}" != "1" ]]; then
  bash script/experiments/run_grid.sh math500
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# One row per run: run_grid.sh writes $RUNS_ROOT/<run_name>/<run_name>.json (rank shards sit next to it).
"$PYTHON_BIN" - "$RUNS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = ["fullkv"] + [f"{s}_b512" for s in ("snapkv", "rkv", "h2o", "streamingllm")]
print(f"{'arm':20s} {'pass@1':>7s} {'correct':>9s} {'d_full':>7s} {'trunc':>6s} {'no_ans':>6s} {'gen_len':>8s} {'cache':>6s}")
full = None
for name in names:
    run = f"math500_llama8b_{name}_s1_multigpu"
    path = root / run / f"{run}.json"
    if not path.exists():
        print(f"{name:20s} missing: {path}")
        continue
    payload = json.loads(path.read_text())
    for arm, s in payload["summary"].items():
        if arm == "fullkv":
            full = s["pass_at_1"]
        delta = "" if full is None or arm == "fullkv" else f"{100 * (s['pass_at_1'] - full):+.1f}"
        complete = "" if payload.get("complete", True) else "  (incomplete)"
        cache = s.get("mean_effective_cache_len")
        cache = "-" if cache is None else f"{cache:.0f}"
        print(
            f"{arm:20s} {s['pass_at_1']:7.4f} {s['correct']:>4d}/{s['total']:<4d} {delta:>7s} "
            f"{s['truncation_rate']:6.3f} {s['no_answer_rate']:6.3f} {s['mean_gen_len']:8.0f} "
            f"{cache:>6s}{complete}"
        )
print("d_full: pass@1 minus FullKV, in percentage points")
PY
