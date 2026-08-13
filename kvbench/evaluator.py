"""Generation evaluator for KV eviction policies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from kv_eviction.runner_token import generate_token_evict, generate_token_evict_batch

from kvbench.datasets import load_problems
from kvbench.metrics import extract_answer, is_correct, summarize_accuracy
from kvbench.models import build_input_ids, pad_input_ids
from kvbench.policies import EvalArm


def run_generation_eval(
    model,
    tokenizer,
    dataset: str,
    n: int,
    arms: list[EvalArm],
    out_path: str | Path,
    max_new: int = 4096,
    recent: int = 64,
    sink: int = 8,
    evict_every: int = 128,
    obs_window: int = 16,
    obs_decay: float = 0.9,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 0,
    batch_size: int = 1,
    num_return_sequences: int = 1,
    seed_offset: int = 0,
    problem_batch_size: int = 1,
    prompt_bucket_size: int = 0,
    only_ids: set[str] | None = None,
    debug_dir: str | Path | None = None,
    debug_topk: int = 0,
    policy_params: dict | None = None,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    progress_prefix: str = "",
    claim_problem_index: Callable[[], int | None] | None = None,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be at least 1")
    if problem_batch_size < 1:
        raise ValueError("problem_batch_size must be at least 1")
    if prompt_bucket_size < 0:
        raise ValueError("prompt_bucket_size cannot be negative")
    if shard_world_size < 1:
        raise ValueError("shard_world_size must be at least 1")
    if shard_rank < 0 or shard_rank >= shard_world_size:
        raise ValueError(
            f"shard_rank must be in [0, {shard_world_size}), got {shard_rank}"
        )
    if claim_problem_index is not None and (shard_rank != 0 or shard_world_size != 1):
        raise ValueError("dynamic problem claiming cannot be combined with static sharding")

    problems = load_problems(dataset, n)
    if only_ids:
        problems = [problem for problem in problems if problem["id"] in only_ids]
    if claim_problem_index is None:
        problems = problems[shard_rank::shard_world_size]

        def next_problem():
            for local_index, problem in enumerate(problems, start=1):
                yield local_index, len(problems), problem

    else:

        def next_problem():
            while True:
                problem_index = claim_problem_index()
                if problem_index is None:
                    return
                if problem_index < 0:
                    raise IndexError(f"claimed negative problem index {problem_index}")
                if problem_index >= len(problems):
                    return
                yield problem_index + 1, len(problems), problems[problem_index]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir) if debug_dir else None

    if problem_batch_size > 1:
        if debug_dir:
            raise ValueError("debug traces currently require problem_batch_size=1")
        return _run_cross_problem_eval(
            model=model,
            tokenizer=tokenizer,
            problem_iter=next_problem(),
            arms=arms,
            out_path=out_path,
            progress_prefix=progress_prefix,
            problem_batch_size=problem_batch_size,
            prompt_bucket_size=(
                prompt_bucket_size
                or (problem_batch_size if claim_problem_index is not None else problem_batch_size * 4)
            ),
            num_return_sequences=num_return_sequences,
            candidate_batch_size=batch_size,
            seed=seed,
            seed_offset=seed_offset,
            max_new=max_new,
            recent=recent,
            sink=sink,
            evict_every=evict_every,
            obs_window=obs_window,
            obs_decay=obs_decay,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            policy_params=policy_params,
        )

    records = []
    for progress_index, progress_total, problem in next_problem():
        row = {"id": problem["id"], "gold": problem["answer"], "arms": {}}
        input_ids = build_input_ids(tokenizer, problem["question"], next(model.parameters()).device)
        for arm in arms:
            debug_path = None
            if debug_dir:
                debug_path = debug_dir / f"{problem['id']}_{arm.name}.json"
            if debug_path and num_return_sequences > 1:
                raise ValueError("debug traces currently require num_return_sequences=1")
            candidate_results = []
            candidate_start = 0
            while candidate_start < num_return_sequences:
                micro_batch_size = min(batch_size, num_return_sequences - candidate_start)
                candidate_seeds = [
                    seed + seed_offset + candidate_start + local_index
                    for local_index in range(micro_batch_size)
                ]
                if num_return_sequences == 1:
                    candidate_results.append(generate_token_evict(
                        model,
                        tokenizer,
                        input_ids,
                        budget=arm.budget,
                        recent=recent,
                        sink=sink,
                        backend=arm.backend,
                        anchor_mode=arm.anchor_mode,
                        anchor_frac=0.0,
                        evict_every=evict_every,
                        obs_window=obs_window,
                        obs_decay=obs_decay,
                        max_new=max_new,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        seed=candidate_seeds[0],
                        policy_params=arm.params or (policy_params or {}).get(arm.backend, {}),
                        debug_path=str(debug_path) if debug_path else None,
                        debug_topk=debug_topk,
                    ))
                else:
                    batch_result = generate_token_evict_batch(
                        model,
                        tokenizer,
                        input_ids,
                        batch_size=micro_batch_size,
                        budget=arm.budget,
                        recent=recent,
                        sink=sink,
                        backend=arm.backend,
                        anchor_mode=arm.anchor_mode,
                        anchor_frac=0.0,
                        evict_every=evict_every,
                        obs_window=obs_window,
                        obs_decay=obs_decay,
                        max_new=max_new,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        seeds=candidate_seeds,
                        policy_params=arm.params or (policy_params or {}).get(arm.backend, {}),
                    )
                    candidate_results.extend(batch_result["candidates"])
                candidate_start += micro_batch_size

            evaluated_candidates = []
            for candidate_idx, result in enumerate(candidate_results):
                pred = extract_answer(result["text"])
                ok = is_correct(pred, problem["answer"])
                evaluated_candidates.append({
                    "candidate_idx": candidate_idx,
                    "seed": seed + seed_offset + candidate_idx,
                    "ok": ok,
                    "pred": pred,
                    "raw_output": result["text"],
                    "gen_len": len(result["gen_ids"]),
                    **{key: value for key, value in result.items() if key not in {"text", "gen_ids"}},
                })
            result = candidate_results[0]
            pred = evaluated_candidates[0]["pred"]
            ok = evaluated_candidates[0]["ok"]
            extra_metric_keys = [
                "mean_effective_cache_len",
                "max_effective_cache_len",
                "min_effective_cache_len",
                "total_effective_kv_tokens",
                "effective_kv_tokens_per_layer_head",
                "head_budget_mean",
                "head_budget_std",
                "head_budget_min",
                "head_budget_max",
                "head_budget_entropy",
                "num_underfilled_heads",
                "total_evicted_tokens",
                "total_effective_evicted_tokens",
                "mean_evicted_per_event",
                "mean_effective_evicted_per_event",
                "min_compression_ratio",
                "max_compression_ratio",
                "mean_effective_compression_ratio",
                "min_effective_compression_ratio",
                "max_effective_compression_ratio",
                "cache_len_curve",
                "effective_cache_len_curve",
                "prefill_sec",
                "decode_sec",
                "decode_forward_sec",
                "attention_observation_sec",
                "eviction_sec_total",
                "eviction_sec_mean",
                "other_decode_sec",
            ]
            row["arms"][arm.name] = {
                "ok": ok,
                "pred": pred,
                "raw_output": result["text"],
                "gen_len": len(result["gen_ids"]),
                "n_evict": result["n_evict"],
                "final_cache_len": result["final_cache_len"],
                "elapsed_sec": result["elapsed_sec"],
                "tokens_per_sec": result["tokens_per_sec"],
                "peak_memory_bytes": result["peak_memory_bytes"],
                "mean_compression_ratio": result["mean_compression_ratio"],
                "evict_events": result["evict_events"],
                "num_return_sequences": num_return_sequences,
                "pass_at_1": sum(candidate["ok"] for candidate in evaluated_candidates) / len(evaluated_candidates),
                "candidates": evaluated_candidates,
                **{key: result[key] for key in extra_metric_keys if key in result},
            }
            print(
                f"{progress_prefix}[{progress_index}/{progress_total}] "
                f"{problem['id']} {arm.name:12s} "
                f"ok={ok} pass@1={row['arms'][arm.name]['pass_at_1']:.3f} "
                f"candidates={num_return_sequences} len={len(result['gen_ids'])} cache={result['final_cache_len']} "
                f"tok/s={result['tokens_per_sec']:.2f}",
                flush=True,
            )
        records.append(row)
        out_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=1))

    summary = summarize_accuracy(records)
    payload = {"records": records, "summary": summary}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return payload


_EXTRA_METRIC_KEYS = [
    "mean_effective_cache_len",
    "max_effective_cache_len",
    "min_effective_cache_len",
    "total_effective_kv_tokens",
    "effective_kv_tokens_per_layer_head",
    "head_budget_mean",
    "head_budget_std",
    "head_budget_min",
    "head_budget_max",
    "head_budget_entropy",
    "num_underfilled_heads",
    "total_evicted_tokens",
    "total_effective_evicted_tokens",
    "mean_evicted_per_event",
    "mean_effective_evicted_per_event",
    "min_compression_ratio",
    "max_compression_ratio",
    "mean_effective_compression_ratio",
    "min_effective_compression_ratio",
    "max_effective_compression_ratio",
    "cache_len_curve",
    "effective_cache_len_curve",
    "prefill_sec",
    "decode_sec",
    "decode_forward_sec",
    "attention_observation_sec",
    "eviction_sec_total",
    "eviction_sec_mean",
    "other_decode_sec",
]


def _evaluated_candidates(candidate_results, gold: str, seed: int, seed_offset: int):
    evaluated = []
    for candidate_idx, result in enumerate(candidate_results):
        pred = extract_answer(result["text"])
        ok = is_correct(pred, gold)
        evaluated.append({
            "candidate_idx": candidate_idx,
            "seed": seed + seed_offset + candidate_idx,
            "ok": ok,
            "pred": pred,
            "raw_output": result["text"],
            "gen_len": len(result["gen_ids"]),
            **{key: value for key, value in result.items() if key not in {"text", "gen_ids"}},
        })
    return evaluated


def _arm_payload(candidate_results, gold: str, seed: int, seed_offset: int):
    evaluated = _evaluated_candidates(candidate_results, gold, seed, seed_offset)
    first_result = candidate_results[0]
    first = evaluated[0]
    return {
        "ok": first["ok"],
        "pred": first["pred"],
        "raw_output": first_result["text"],
        "gen_len": len(first_result["gen_ids"]),
        "n_evict": first_result["n_evict"],
        "final_cache_len": first_result["final_cache_len"],
        "elapsed_sec": first_result["elapsed_sec"],
        "tokens_per_sec": first_result["tokens_per_sec"],
        "peak_memory_bytes": first_result["peak_memory_bytes"],
        "mean_compression_ratio": first_result["mean_compression_ratio"],
        "evict_events": first_result["evict_events"],
        "num_return_sequences": len(evaluated),
        "pass_at_1": sum(candidate["ok"] for candidate in evaluated) / len(evaluated),
        "candidates": evaluated,
        **{key: first_result[key] for key in _EXTRA_METRIC_KEYS if key in first_result},
    }


def _take_window(iterator, size: int):
    window = []
    for _ in range(size):
        try:
            window.append(next(iterator))
        except StopIteration:
            break
    return window


def _run_cross_problem_eval(
    *,
    model,
    tokenizer,
    problem_iter,
    arms,
    out_path,
    progress_prefix,
    problem_batch_size,
    prompt_bucket_size,
    num_return_sequences,
    candidate_batch_size,
    seed,
    seed_offset,
    max_new,
    recent,
    sink,
    evict_every,
    obs_window,
    obs_decay,
    do_sample,
    temperature,
    top_p,
    policy_params,
):
    """Evaluate length-bucketed heterogeneous prompts in static batches."""
    device = next(model.parameters()).device
    iterator = iter(problem_iter)
    window_size = prompt_bucket_size
    records_with_order = []

    while True:
        raw_window = _take_window(iterator, window_size)
        if not raw_window:
            break
        tokenized = []
        for progress_index, progress_total, problem in raw_window:
            ids = build_input_ids(tokenizer, problem["question"], device)
            tokenized.append((int(ids.numel()), progress_index, progress_total, problem, ids))
        tokenized.sort(key=lambda item: item[0])

        for group_start in range(0, len(tokenized), problem_batch_size):
            group = tokenized[group_start:group_start + problem_batch_size]
            prompt_rows = [item[4] for item in group]
            padded_ids, prompt_mask, _ = pad_input_ids(tokenizer, prompt_rows, device)
            group_records = [
                {
                    "id": item[3]["id"],
                    "gold": item[3]["answer"],
                    "arms": {},
                    "_order": item[1],
                    "_total": item[2],
                }
                for item in group
            ]

            for arm in arms:
                per_problem_results = [[] for _ in group]
                candidate_start = 0
                while candidate_start < num_return_sequences:
                    candidates_this_batch = min(
                        candidate_batch_size, num_return_sequences - candidate_start
                    )
                    expanded_ids = padded_ids.repeat_interleave(candidates_this_batch, dim=0)
                    expanded_mask = prompt_mask.repeat_interleave(candidates_this_batch, dim=0)
                    seeds = [
                        seed + seed_offset + candidate_start + candidate_idx
                        for _ in group
                        for candidate_idx in range(candidates_this_batch)
                    ]
                    result = generate_token_evict_batch(
                        model,
                        tokenizer,
                        expanded_ids,
                        batch_size=len(seeds),
                        input_attention_mask=expanded_mask,
                        budget=arm.budget,
                        recent=recent,
                        sink=sink,
                        backend=arm.backend,
                        anchor_mode=arm.anchor_mode,
                        anchor_frac=0.0,
                        evict_every=evict_every,
                        obs_window=obs_window,
                        obs_decay=obs_decay,
                        max_new=max_new,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        seeds=seeds,
                        policy_params=arm.params or (policy_params or {}).get(arm.backend, {}),
                    )
                    for problem_idx in range(len(group)):
                        begin = problem_idx * candidates_this_batch
                        end = begin + candidates_this_batch
                        per_problem_results[problem_idx].extend(result["candidates"][begin:end])
                    candidate_start += candidates_this_batch

                for problem_idx, record in enumerate(group_records):
                    record["arms"][arm.name] = _arm_payload(
                        per_problem_results[problem_idx], record["gold"], seed, seed_offset
                    )

            for record in group_records:
                representative = next(iter(record["arms"].values()))
                print(
                    f"{progress_prefix}[{record['_order']}/{record['_total']}] "
                    f"{record['id']} cross_batch={len(group)} "
                    f"candidates={num_return_sequences} tok/s={representative['tokens_per_sec']:.2f}",
                    flush=True,
                )
                records_with_order.append(record)

            serializable = []
            for record in sorted(records_with_order, key=lambda item: item["_order"]):
                serializable.append({key: value for key, value in record.items() if not key.startswith("_")})
            out_path.write_text(json.dumps({"records": serializable}, ensure_ascii=False, indent=1))

    records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in sorted(records_with_order, key=lambda item: item["_order"])
    ]
    payload = {"records": records, "summary": summarize_accuracy(records)}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return payload
