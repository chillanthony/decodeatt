"""Teacher-forced eviction scoring for TideProbe transition alignment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from kv_eviction.runner_token import score_trace_token
from kvbench.diagnostics.gen_traces import _atomic_torch_save, _load_config, _pick
from kvbench.diagnostics.layer_sensitivity import parse_int_list
from kvbench.models import load_causal_lm


def _distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid distributed context rank={rank}, world_size={world_size}"
        )
    return rank, world_size, local_rank


def load_event_positions(path: str | Path) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            positions.setdefault(str(row["id"]), []).append(int(row["token_index"]))
    return {trace_id: sorted(set(rows)) for trace_id, rows in positions.items()}


def assigned_strategy(
    strategies: list[str], rank: int, world_size: int
) -> tuple[str, int, int]:
    if world_size < len(strategies):
        raise ValueError(
            f"need at least {len(strategies)} ranks for strategies {strategies}, got {world_size}"
        )
    strategy_index = rank % len(strategies)
    ranks = list(range(strategy_index, world_size, len(strategies)))
    return strategies[strategy_index], ranks.index(rank), len(ranks)


def attention_backend(strategy: str, requested: str, fast_backend: str) -> str:
    if requested != "auto":
        return requested
    return "eager" if strategy == "window" else fast_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--dtype", default=None, choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--attn", default="auto")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--only-ids", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    diagnostic_config = config.get("diagnostics", {})
    alignment_config = diagnostic_config.get("eviction_alignment", {})
    rank, world_size, local_rank = _distributed_context()

    model_name = _pick(args.model, config, ("model", "name"))
    dtype = _pick(args.dtype, config, ("model", "dtype"), "bfloat16")
    fast_backend = _pick(None, config, ("model", "fast_attn"), "sdpa")
    trace_dir = Path(
        _pick(
            args.trace_dir,
            config,
            ("diagnostics", "trace_dir"),
            "/home/ma-user/work/bucket-wulan-green/chenyanbo/trace",
        )
    )
    events_path = Path(
        _pick(
            args.events,
            config,
            ("diagnostics", "events_path"),
            "runs/tideprobe_step1/transition_events.jsonl",
        )
    )
    raw_dir = Path(
        _pick(
            args.raw_dir,
            config,
            ("diagnostics", "eviction_alignment_dir"),
            "runs/tideprobe_step1/eviction_alignment_raw",
        )
    )
    strategies = [
        item.strip()
        for item in (
            args.strategies
            or alignment_config.get("strategies", "rkv,snapkv,window,random")
        ).split(",")
        if item.strip()
    ]
    budgets = parse_int_list(
        args.budgets or alignment_config.get("budgets", "512,1024,1536")
    )
    only_ids = set(args.only_ids.split(",")) if args.only_ids else None
    if not events_path.exists():
        raise SystemExit(
            f"missing transition events: {events_path}; run Experiment 1 first"
        )
    event_positions = load_event_positions(events_path)
    if not event_positions:
        raise SystemExit(f"transition event file is empty: {events_path}")

    strategy, strategy_rank, strategy_world_size = assigned_strategy(
        strategies, rank, world_size
    )
    trace_paths = sorted(trace_dir.glob("*.pt"))
    if only_ids:
        trace_paths = [path for path in trace_paths if path.stem in only_ids]
    if not trace_paths:
        raise SystemExit(f"no trace files found in {trace_dir}")
    tasks = [(trace_path, budget) for trace_path in trace_paths for budget in budgets]
    assigned_tasks = tasks[strategy_rank::strategy_world_size]
    raw_dir.mkdir(parents=True, exist_ok=True)
    backend = attention_backend(strategy, args.attn, fast_backend)
    print(
        f"[rank {rank}/{world_size}] strategy={strategy} strategy_shard="
        f"{strategy_rank}/{strategy_world_size} tasks={len(assigned_tasks)}/{len(tasks)} "
        f"attn={backend} output={raw_dir}",
        flush=True,
    )
    if not assigned_tasks:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: str | dict = {"": f"cuda:{local_rank}"}
    else:
        device_map = "cpu"
    model, tokenizer = load_causal_lm(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=backend,
    )
    policy_params = dict(alignment_config.get("policy_params", {}).get(strategy, {}))
    for task_index, (trace_path, budget) in enumerate(assigned_tasks, start=1):
        output_path = raw_dir / f"{trace_path.stem}.{strategy}.b{budget}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"[rank {rank}] skip existing {output_path}", flush=True)
            continue
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        if (
            trace.get("model") != model_name
            and Path(str(trace.get("model"))).name != Path(model_name).name
        ):
            raise ValueError(
                f"trace/model mismatch for {trace_path}: {trace.get('model')} != {model_name}"
            )
        print(
            f"[rank {rank} {task_index}/{len(assigned_tasks)}] {trace['id']} "
            f"{strategy}@{budget} tokens={trace['gen_ids'].numel()}",
            flush=True,
        )
        full_ids = torch.cat([trace["prompt_ids"], trace["gen_ids"]])
        result = score_trace_token(
            model,
            tokenizer,
            full_ids,
            prompt_len=int(trace["prompt_ids"].numel()),
            refl_steps=event_positions.get(str(trace["id"]), []),
            anchor_mode="none",
            budget=budget,
            recent=int(alignment_config.get("recent", 8)),
            sink=int(alignment_config.get("sink", 0)),
            backend=strategy,
            evict_every=int(alignment_config.get("evict_every", 128)),
            obs_window=int(alignment_config.get("obs_window", 8)),
            obs_decay=float(alignment_config.get("obs_decay", 0.9)),
            span_len=int(alignment_config.get("downstream_horizon", 32)),
            policy_params=policy_params,
            debug=True,
            seed=int(alignment_config.get("seed", 0)),
        )
        first_eviction_step = (
            int(result["eviction_events"][0]["eviction_step"])
            if result["eviction_events"]
            else int(result["per_token_nll"].numel())
        )
        reference_nll = trace["fullkv_nll"].float()
        pre_eviction_nll_max_abs_error = (
            float(
                (
                    result["per_token_nll"][:first_eviction_step]
                    - reference_nll[:first_eviction_step]
                )
                .abs()
                .max()
            )
            if first_eviction_step > 0
            else 0.0
        )
        payload = {
            "schema_version": 1,
            "trace_id": trace["id"],
            "source_trace": trace_path.name,
            "model": trace["model"],
            "strategy": strategy,
            "budget": budget,
            "prompt_length": int(trace["prompt_ids"].numel()),
            "num_tokens": int(trace["gen_ids"].numel()),
            "pre_eviction_nll_max_abs_error": pre_eviction_nll_max_abs_error,
            **result,
        }
        _atomic_torch_save(payload, output_path)
        print(
            f"[rank {rank}] wrote {len(result['eviction_events'])} evictions -> {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
