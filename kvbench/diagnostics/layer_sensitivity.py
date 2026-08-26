"""Layer-wise counterfactual sparse-attention probes on fixed FullKV traces."""

from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from kv_eviction.runner_token import _attention_logit_rows
from kv_eviction.strategies.rkv import select_rkv_global
from kvbench.diagnostics.gen_traces import _load_config, _pick
from kvbench.models import load_causal_lm

CSV_FIELDS = [
    "trace_id",
    "probe_type",
    "probe_index",
    "transition_score",
    "history_length",
    "horizon",
    "scan_mode",
    "layer_start",
    "layer_end",
    "layers",
    "budget",
    "mask_policy",
    "mask_seed",
    "nll_fullkv",
    "nll_sparse",
    "nll_delta",
    "baseline_max_abs_error",
]


def _distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid distributed context rank={rank}, world_size={world_size}"
        )
    return rank, world_size, local_rank


def model_layers(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None and hasattr(base, "model"):
        layers = getattr(base.model, "layers", None)
    if layers is None:
        raise TypeError(f"cannot locate decoder layers on {type(model).__name__}")
    return layers


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError(f"invalid integer range {part!r}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def layer_sets(
    num_layers: int,
    *,
    scan_mode: str,
    group_size: int,
    selected_layers: list[int] | None = None,
) -> list[tuple[int, ...]]:
    if scan_mode == "group":
        return [
            tuple(range(start, min(start + group_size, num_layers)))
            for start in range(0, num_layers, group_size)
        ]
    if scan_mode != "layer":
        raise ValueError("scan_mode must be group or layer")
    indices = selected_layers if selected_layers else list(range(num_layers))
    if any(index < 0 or index >= num_layers for index in indices):
        raise ValueError(f"layer indices must be in [0, {num_layers})")
    return [(index,) for index in indices]


def load_transition_events(path: str | Path | None) -> dict[str, list[dict]]:
    if not path or not Path(path).exists():
        return {}
    events: dict[str, list[dict]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            events.setdefault(str(row["id"]), []).append(row)
    for rows in events.values():
        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return events


def select_probe_points(
    trace_length: int,
    transition_events: list[dict],
    *,
    horizon: int,
    min_probe_index: int,
    uniform_count: int,
    max_transitions: int,
    ordinary_offset: int,
    transition_exclusion: int,
    probe_types: set[str],
) -> list[dict]:
    """Choose uniform, transition, and nearby matched-normal probe positions."""
    lower = max(1, min_probe_index)
    upper = trace_length - horizon
    if upper < lower:
        return []
    points = []
    if "uniform" in probe_types and uniform_count > 0:
        fractions = np.linspace(lower, upper, uniform_count + 2)[1:-1]
        for position in sorted({round(float(value)) for value in fractions}):
            points.append(
                {"probe_type": "uniform", "probe_index": position, "score": None}
            )

    valid_transitions = [
        event
        for event in transition_events
        if lower <= int(event["token_index"]) <= upper
    ][:max_transitions]
    if "transition" in probe_types:
        for event in valid_transitions:
            points.append(
                {
                    "probe_type": "transition",
                    "probe_index": int(event["token_index"]),
                    "score": float(event.get("score", 0.0)),
                }
            )

    if "ordinary" in probe_types:
        transition_positions = [
            int(event["token_index"]) for event in valid_transitions
        ]
        for transition_position in transition_positions:
            chosen = None
            for multiplier in range(1, 5):
                for direction in (1, -1):
                    candidate = (
                        transition_position + direction * multiplier * ordinary_offset
                    )
                    if candidate < lower or candidate > upper:
                        continue
                    if all(
                        abs(candidate - event_position) >= transition_exclusion
                        for event_position in transition_positions
                    ):
                        chosen = candidate
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                points.append(
                    {
                        "probe_type": "ordinary",
                        "probe_index": chosen,
                        "score": None,
                        "matched_transition_index": transition_position,
                    }
                )
    return points


def random_keep_indices(length: int, budget: int, seed: int, device) -> torch.Tensor:
    if length <= budget:
        return torch.arange(length, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = torch.randperm(length, generator=generator)[:budget].sort().values
    return selected.to(device)


def window_keep_indices(length: int, budget: int, device) -> torch.Tensor:
    return torch.arange(max(0, length - budget), length, device=device)


def evicted_indices(length: int, kept: torch.Tensor, device) -> torch.Tensor:
    keep_mask = torch.zeros(length, dtype=torch.bool, device=device)
    keep_mask[kept.to(device=device, dtype=torch.long)] = True
    return (~keep_mask).nonzero(as_tuple=True)[0]


def _counterfactual_attention_hook(evicted: torch.Tensor):
    def hook(_module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise RuntimeError("attention hook could not locate hidden_states")
        query_length = int(hidden_states.shape[1])
        cache_position = kwargs.get("cache_position")
        if cache_position is None:
            raise RuntimeError("attention hook requires cache_position")
        key_length = int(cache_position[-1]) + 1
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            minimum = torch.finfo(hidden_states.dtype).min
            key_positions = torch.arange(key_length, device=hidden_states.device)
            allowed = key_positions.unsqueeze(0) <= cache_position.reshape(-1, 1)
            attention_mask = torch.full(
                (1, 1, query_length, key_length),
                minimum,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_mask[0, 0].masked_fill_(allowed, 0.0)
        else:
            attention_mask = attention_mask.clone()
        valid_evicted = evicted[evicted < attention_mask.shape[-1]]
        if attention_mask.dtype == torch.bool:
            attention_mask[..., valid_evicted] = False
        else:
            attention_mask[..., valid_evicted] = torch.finfo(attention_mask.dtype).min
        kwargs["attention_mask"] = attention_mask
        return args, kwargs

    return hook


@contextmanager
def masked_attention_layers(
    model, layer_indices: tuple[int, ...], evicted: torch.Tensor
):
    handles = []
    layers = model_layers(model)
    try:
        for layer_index in layer_indices:
            handles.append(
                layers[layer_index].self_attn.register_forward_pre_hook(
                    _counterfactual_attention_hook(evicted), with_kwargs=True
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def _crop_cache(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]
        return
    raise TypeError(f"cannot crop cache type {type(cache).__name__}")


@torch.no_grad()
def score_query_block(
    model,
    cache,
    query_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    history_length: int,
    layer_indices: tuple[int, ...] | None = None,
    evicted: torch.Tensor | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    query_ids = query_ids.reshape(1, -1).to(device)
    target_ids = target_ids.reshape(-1).to(device)
    cache_positions = torch.arange(
        history_length, history_length + query_ids.shape[1], device=device
    )
    context = (
        masked_attention_layers(model, layer_indices, evicted)
        if layer_indices is not None and evicted is not None and evicted.numel()
        else nullcontext()
    )
    try:
        with context:
            output = model(
                input_ids=query_ids,
                past_key_values=cache,
                use_cache=True,
                attention_mask=torch.ones(
                    1,
                    history_length + query_ids.shape[1],
                    dtype=torch.long,
                    device=device,
                ),
                position_ids=cache_positions.unsqueeze(0),
                cache_position=cache_positions,
            )
        log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
        return -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).detach().cpu()
    finally:
        _crop_cache(cache, history_length)


@torch.no_grad()
def build_prefix_cache(
    model, prefix_ids: torch.Tensor, *, collect_rkv: bool, alpha: int
):
    """Build FullKV prefix cache and optionally retain R-KV observation rows."""
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.reshape(-1).to(device)
    length = int(prefix_ids.numel())
    if length < 1:
        raise ValueError("probe prefix cannot be empty")
    observation_count = min(alpha, max(0, length - 1)) if collect_rkv else 0
    prefill_length = length - observation_count
    prefill_ids = prefix_ids[:prefill_length].reshape(1, -1)
    output = model(
        input_ids=prefill_ids,
        attention_mask=torch.ones_like(prefill_ids),
        use_cache=True,
        output_hidden_states=False,
    )
    cache = output.past_key_values
    attention_history = []
    for position in range(prefill_length, length):
        position_ids = torch.tensor([[position]], device=device)
        output = model(
            input_ids=prefix_ids[position].reshape(1, 1),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=torch.ones(1, position + 1, dtype=torch.long, device=device),
            position_ids=position_ids,
            cache_position=torch.tensor([position], device=device),
        )
        cache = output.past_key_values
        rows = _attention_logit_rows(model, output, cache, position_ids, position + 1)
        if rows is None:
            raise RuntimeError(
                "R-KV attention-logit extraction is unsupported for this model"
            )
        attention_history.append(rows)
    return cache, attention_history


def _rkv_keep_by_budget(
    cache, attention_history, length: int, budgets: list[int], params: dict
):
    compressed_budgets = [budget for budget in budgets if budget < length]
    if not compressed_budgets:
        return {
            budget: torch.arange(length, device=cache.layers[0].keys.device)
            for budget in budgets
        }
    alpha = min(int(params.get("alpha", 8)), length)
    if any(budget <= alpha for budget in compressed_budgets):
        raise ValueError(f"R-KV budgets must be greater than alpha={alpha}")
    maximum_budget = max(compressed_budgets)
    _, debug = select_rkv_global(
        cache,
        attention_history,
        length,
        maximum_budget,
        params,
        return_debug=True,
    )
    score = debug["policy_score"]
    candidate_length = length - alpha
    result = {}
    for budget in budgets:
        if length <= budget:
            result[budget] = torch.arange(length, device=score.device)
            continue
        selected = torch.argsort(score[:candidate_length], descending=True)[
            : budget - alpha
        ]
        recent = torch.arange(candidate_length, length, device=score.device)
        result[budget] = torch.cat([selected, recent]).sort().values
    return result


def _mask_specs(
    *,
    length: int,
    budgets: list[int],
    policies: list[str],
    random_seeds: list[int],
    device,
    cache,
    attention_history,
    rkv_params: dict,
):
    rkv_keeps = (
        _rkv_keep_by_budget(cache, attention_history, length, budgets, rkv_params)
        if "rkv" in policies
        else {}
    )
    for budget in budgets:
        if budget < 1:
            raise ValueError("budgets must be positive")
        for policy in policies:
            if policy == "random":
                for seed in random_seeds:
                    yield (
                        policy,
                        seed,
                        budget,
                        random_keep_indices(length, budget, seed, device),
                    )
            elif policy == "window":
                yield policy, None, budget, window_keep_indices(length, budget, device)
            elif policy == "rkv":
                yield policy, None, budget, rkv_keeps[budget]
            else:
                raise ValueError(f"unsupported mask policy {policy!r}")


@torch.no_grad()
def probe_trace_point(
    model,
    trace: dict,
    point: dict,
    *,
    layer_groups: list[tuple[int, ...]],
    scan_mode: str,
    budgets: list[int],
    mask_policies: list[str],
    random_seeds: list[int],
    horizon: int,
    rkv_params: dict,
) -> list[dict]:
    device = next(model.parameters()).device
    prompt_ids = trace["prompt_ids"].reshape(-1)
    generated_ids = trace["gen_ids"].reshape(-1)
    probe_index = int(point["probe_index"])
    actual_horizon = min(horizon, int(generated_ids.numel()) - probe_index)
    if probe_index < 1 or actual_horizon < 1:
        return []
    prefix_ids = torch.cat([prompt_ids, generated_ids[: probe_index - 1]])
    query_ids = generated_ids[probe_index - 1 : probe_index + actual_horizon - 1]
    target_ids = generated_ids[probe_index : probe_index + actual_horizon]
    history_length = int(prefix_ids.numel())
    cache, attention_history = build_prefix_cache(
        model,
        prefix_ids,
        collect_rkv="rkv" in mask_policies,
        alpha=int(rkv_params.get("alpha", 8)),
    )
    baseline = score_query_block(
        model,
        cache,
        query_ids,
        target_ids,
        history_length=history_length,
    )
    reference = trace["fullkv_nll"][probe_index : probe_index + actual_horizon].float()
    baseline_error = float((baseline - reference).abs().max().item())
    baseline_mean = float(reference.mean().item())

    rows = []
    for policy, seed, requested_budget, kept in _mask_specs(
        length=history_length,
        budgets=budgets,
        policies=mask_policies,
        random_seeds=random_seeds,
        device=device,
        cache=cache,
        attention_history=attention_history,
        rkv_params=rkv_params,
    ):
        evicted = evicted_indices(history_length, kept, device)
        for group in layer_groups:
            sparse = score_query_block(
                model,
                cache,
                query_ids,
                target_ids,
                history_length=history_length,
                layer_indices=group,
                evicted=evicted,
            )
            sparse_mean = float(sparse.mean().item())
            rows.append(
                {
                    "trace_id": trace["id"],
                    "probe_type": point["probe_type"],
                    "probe_index": probe_index,
                    "transition_score": point.get("score"),
                    "history_length": history_length,
                    "horizon": actual_horizon,
                    "scan_mode": scan_mode,
                    "layer_start": group[0],
                    "layer_end": group[-1],
                    "layers": f"{group[0]}-{group[-1]}"
                    if len(group) > 1
                    else str(group[0]),
                    "budget": requested_budget,
                    "mask_policy": policy,
                    "mask_seed": seed,
                    "nll_fullkv": baseline_mean,
                    "nll_sparse": sparse_mean,
                    "nll_delta": sparse_mean - baseline_mean,
                    "baseline_max_abs_error": baseline_error,
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--dtype", default=None, choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--attn", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--scan-mode", choices=["group", "layer"], default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--layers", default=None, help="Fine scan layers, e.g. 8-11,20")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--mask-policies", default=None)
    parser.add_argument("--random-seeds", default=None)
    parser.add_argument("--probe-types", default=None)
    parser.add_argument(
        "--run-tag", default=None, help="Shard filename tag for separate runs"
    )
    parser.add_argument("--only-ids", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    layer_config = config.get("diagnostics", {}).get("layer_sensitivity", {})
    rank, world_size, local_rank = _distributed_context()
    model_name = _pick(args.model, config, ("model", "name"))
    dtype = _pick(args.dtype, config, ("model", "dtype"), "bfloat16")
    attention_backend = _pick(args.attn, config, ("model", "fast_attn"), "sdpa")
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
    output_dir = Path(
        _pick(
            args.out_dir,
            config,
            ("diagnostics", "layer_sensitivity_dir"),
            "runs/tideprobe_step1/layer_sensitivity",
        )
    )
    scan_mode = args.scan_mode or str(layer_config.get("scan_mode", "group"))
    group_size = int(args.group_size or layer_config.get("group_size", 4))
    budgets = parse_int_list(args.budgets or layer_config.get("budgets", "256,512"))
    mask_policies = [
        item.strip()
        for item in (
            args.mask_policies or layer_config.get("mask_policies", "random,window")
        ).split(",")
        if item.strip()
    ]
    random_seeds = parse_int_list(
        args.random_seeds or layer_config.get("random_seeds", "0,1,2")
    )
    probe_types = {
        item.strip()
        for item in (
            args.probe_types
            or layer_config.get("probe_types", "uniform,transition,ordinary")
        ).split(",")
        if item.strip()
    }
    selected_layers = parse_int_list(args.layers) if args.layers else None
    run_tag = args.run_tag or "_".join(mask_policies)
    if not run_tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "run-tag may contain only letters, numbers, underscores, and hyphens"
        )
    only_ids = set(args.only_ids.split(",")) if args.only_ids else None

    trace_paths = sorted(trace_dir.glob("*.pt"))
    if only_ids:
        trace_paths = [path for path in trace_paths if path.stem in only_ids]
    if not trace_paths:
        raise SystemExit(f"no trace files found in {trace_dir}")
    transition_events = load_transition_events(events_path)
    if {"transition", "ordinary"} & probe_types and not transition_events:
        raise SystemExit(
            f"probe types {sorted(probe_types)} require non-empty transition events at "
            f"{events_path}; run Experiment 1 first or use --probe-types uniform"
        )
    jobs = []
    for trace_path in trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        points = select_probe_points(
            int(trace["gen_ids"].numel()),
            transition_events.get(str(trace["id"]), []),
            horizon=int(layer_config.get("horizon", 32)),
            min_probe_index=int(layer_config.get("min_probe_index", 32)),
            uniform_count=int(layer_config.get("uniform_count", 1)),
            max_transitions=int(layer_config.get("max_transitions_per_trace", 1)),
            ordinary_offset=int(layer_config.get("ordinary_offset", 128)),
            transition_exclusion=int(layer_config.get("transition_exclusion", 64)),
            probe_types=probe_types,
        )
        jobs.extend((trace_path, point) for point in points)
    assigned = jobs[rank::world_size]
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"layer_sensitivity.{scan_mode}.{run_tag}.rank{rank}.csv"
    print(
        f"[rank {rank}/{world_size}] jobs={len(assigned)}/{len(jobs)} scan={scan_mode} "
        f"budgets={budgets} policies={mask_policies} output={shard_path}",
        flush=True,
    )
    if not assigned:
        with open(shard_path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: str | dict = {"": f"cuda:{local_rank}"}
    else:
        device_map = "cpu"
    model, _ = load_causal_lm(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attention_backend,
    )
    groups = layer_sets(
        len(model_layers(model)),
        scan_mode=scan_mode,
        group_size=group_size,
        selected_layers=selected_layers,
    )
    rkv_params = dict((config.get("policy_defaults") or {}).get("rkv", {}))
    rkv_params.update(layer_config.get("rkv", {}))
    all_rows = []
    for job_index, (trace_path, point) in enumerate(assigned, start=1):
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        print(
            f"[rank {rank} {job_index}/{len(assigned)}] {trace['id']} "
            f"{point['probe_type']}@{point['probe_index']}",
            flush=True,
        )
        all_rows.extend(
            probe_trace_point(
                model,
                trace,
                point,
                layer_groups=groups,
                scan_mode=scan_mode,
                budgets=budgets,
                mask_policies=mask_policies,
                random_seeds=random_seeds,
                horizon=int(layer_config.get("horizon", 32)),
                rkv_params=rkv_params,
            )
        )
    with open(shard_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[rank {rank}] wrote {len(all_rows)} rows to {shard_path}", flush=True)


if __name__ == "__main__":
    main()
