#!/usr/bin/env bash
# 2026-11-13 full budget grid on MATH-500: FullKV + R-KV / SnapKV / H2O / StreamingLLM / Random at budgets
# 128 256 512 768 1024 1536 2048, 4 samples per problem, then pass@1 / truncation tables (method x budget).
#
#   bash script/experiments/gridtest-1113.sh                # run the grid, then print the tables
#   TABLE_ONLY=1 bash script/experiments/gridtest-1113.sh   # only print the tables
#
# Multi-node: start the same command on every node; no node rank is needed. Each node walks the job
# list (FullKV first, then budgets small to large) and claims the next unfinished job by creating
# $RUNS_ROOT/.claims/<run> (mkdir is the lock), so nodes share the work dynamically and none idles
# before the last job. Each job runs as single-node torchrun on that node's GPUs. RUNS_ROOT must be
# shared by all nodes. A running job touches a heartbeat file every minute; a claim whose heartbeat is
# older than CLAIM_STALE_MIN (default 15) minutes belongs to a dead node and is taken over.
# Every arm resumes its own result file, so rerunning the same command after a crash continues.
# Other run_grid.sh overrides (BUDGETS, NUM_RETURN_SEQUENCES, ROWS_PER_BATCH, OVERWRITE, DRY_RUN, ...) pass through.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/config/runtime/a800.env}"
source "$ENV_FILE"
export ENV_FILE
cd "$ROOT_DIR"

export BUDGETS="${BUDGETS:-128 256 512 768 1024 1536 2048}"
export NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"
export GRID_STRATEGIES="${GRID_STRATEGIES:-rkv snapkv h2o streamingllm random}"

CLAIM_DIR="$RUNS_ROOT/.claims"
CLAIM_STALE_MIN="${CLAIM_STALE_MIN:-15}"

run_name() {
  if [[ "$1" == "fullkv" ]]; then
    echo "math500_llama8b_fullkv_s${NUM_RETURN_SEQUENCES}_multigpu"
  else
    echo "math500_llama8b_${1%@*}_b${1#*@}_s${NUM_RETURN_SEQUENCES}_multigpu"
  fi
}

is_complete() {
  "$PYTHON_BIN" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("complete") else 1)' \
    "$RUNS_ROOT/$1/$1.json" 2>/dev/null
}

claim() {  # succeed if this node now owns the job
  local dir="$CLAIM_DIR/$1"
  if mkdir "$dir" 2>/dev/null; then
    hostname > "$dir/owner"
    touch "$dir/heartbeat"
    return 0
  fi
  # Existing claim: take it over if it is this host's own (a rerun after a crash) or its heartbeat
  # went stale (the owner died).
  if [[ "$(cat "$dir/owner" 2>/dev/null)" == "$(hostname)" ]] \
    || [[ -n "$(find "$dir/heartbeat" -mmin +"$CLAIM_STALE_MIN" 2>/dev/null)" ]]; then
    echo "[gridtest-1113] taking over stale claim $1 (owner $(cat "$dir/owner" 2>/dev/null))"
    hostname > "$dir/owner"
    touch "$dir/heartbeat"
    return 0
  fi
  return 1
}

run_job() {
  local job="$1"
  if [[ "$job" == "fullkv" ]]; then
    RUN_FULLKV=1 GRID_STRATEGIES="" bash script/experiments/run_grid.sh math500
  else
    RUN_FULLKV=0 GRID_STRATEGIES="${job%@*}" BUDGETS="${job#*@}" bash script/experiments/run_grid.sh math500
  fi
}

if [[ "${TABLE_ONLY:-0}" != "1" ]]; then
  JOBS=(fullkv)
  for budget in $BUDGETS; do
    for strategy in $GRID_STRATEGIES; do
      JOBS+=("$strategy@$budget")
    done
  done
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[gridtest-1113] ${#JOBS[@]} jobs: ${JOBS[*]}"
    for job in "${JOBS[@]}"; do run_job "$job"; done
    exit 0
  fi
  mkdir -p "$CLAIM_DIR"
  sleep $(( RANDOM % 30 ))  # nodes start together; spread the first claims apart
  done_here=()
  for job in "${JOBS[@]}"; do
    name="$(run_name "$job")"
    if [[ "${OVERWRITE:-0}" != "1" ]] && is_complete "$name"; then
      continue
    fi
    claim "$name" || continue
    echo "[gridtest-1113] $(hostname) claimed $job"
    ( while sleep 60; do touch "$CLAIM_DIR/$name/heartbeat"; done ) &
    heartbeat_pid=$!
    status=0
    run_job "$job" || status=$?
    kill "$heartbeat_pid" 2>/dev/null || true
    if (( status != 0 )); then
      rm -rf "$CLAIM_DIR/$name"  # release so a rerun (or the other node) can retry it
      echo "[gridtest-1113] $job failed with status $status" >&2
      exit "$status"
    fi
    done_here+=("$job")
  done
  echo "[gridtest-1113] $(hostname) finished ${#done_here[@]} jobs: ${done_here[*]}"
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# Tables (the last node to finish prints the complete one): one row per method, one column per budget. run_grid.sh writes $RUNS_ROOT/<run>/<run>.json.
"$PYTHON_BIN" - "$RUNS_ROOT" "$NUM_RETURN_SEQUENCES" "$BUDGETS" $GRID_STRATEGIES <<'PY'
import json
import sys
from pathlib import Path

root, samples, budgets, strategies = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(), sys.argv[4:]


def load(name):
    run = f"math500_llama8b_{name}_s{samples}_multigpu"
    path = root / run / f"{run}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    summary = next(iter(payload["summary"].values()))
    summary["incomplete"] = not payload.get("complete", True)
    return summary


full = load("fullkv")
cells = {(s, b): load(f"{s}_b{b}") for s in strategies for b in budgets}
if full:
    print(f"FullKV: pass@1 {full['pass_at_1']:.4f} ({full['correct']}/{full['total']}), "
          f"trunc {full['truncation_rate']:.3f}, gen_len {full['mean_gen_len']:.0f}"
          f"{'  (incomplete)' if full['incomplete'] else ''}")
else:
    print("FullKV: missing")

for title, key, fmt in [
    ("pass@1 (%)", "pass_at_1", lambda v: f"{100 * v:.1f}"),
    ("pass@1 - FullKV (pp)", "pass_at_1", lambda v: f"{100 * (v - full['pass_at_1']):+.1f}" if full else "-"),
    ("truncation rate (%)", "truncation_rate", lambda v: f"{100 * v:.1f}"),
    ("mean gen len", "mean_gen_len", lambda v: f"{v:.0f}"),
]:
    print(f"\n{title}")
    print(f"{'method':14s}" + "".join(f"{b:>8s}" for b in budgets))
    for s in strategies:
        row = []
        for b in budgets:
            c = cells[(s, b)]
            row.append("-" if c is None else fmt(c[key]) + ("*" if c["incomplete"] else ""))
        print(f"{s:14s}" + "".join(f"{v:>8s}" for v in row))
print("\n-: missing run, *: incomplete run")
PY
