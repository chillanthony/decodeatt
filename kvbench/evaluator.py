"""Generation evaluator for KV eviction policies."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from kveviction.runner_token import generate_token_evict, generate_token_evict_batch
from kveviction.backends import ExecutionBackend
from kveviction.cache_manager import CacheManager

from kvbench.datasets import load_problems
from kvbench.metrics import extract_answer, is_correct, summarize_accuracy
from kvbench.models import build_input_ids, pad_input_ids
from kvbench.arms import EvalArm
from kvbench.result_schema import check_candidate
from kvbench.resume import atomic_write_json, load_resumable, run_fingerprint


_LOG_MODES = {"full", "brief"}


def _result_payload(
    records: list[dict], log_mode: str, *, final: bool, run_config: dict | None = None
) -> dict:
    """Build the on-disk payload without changing the in-memory evaluation data.

    Partial brief-mode files keep their (scalar-only) records so an interrupted
    run can resume; the final brief file keeps only the summary.
    """
    if log_mode == "brief" and final:
        payload = {
            "log_mode": "brief",
            "num_problems": len(records),
            "summary": summarize_accuracy(records),
        }
    else:
        payload = {"records": records}
        if log_mode == "brief":
            payload["log_mode"] = "brief"
        if final:
            payload["summary"] = summarize_accuracy(records)
    if run_config is not None:
        payload["run_config"] = run_config
        payload["complete"] = final
    return payload


def _write_result(
    out_path: Path, records: list[dict], log_mode: str, *, final: bool, run_config: dict | None = None
) -> dict:
    payload = _result_payload(records, log_mode, final=final, run_config=run_config)
    atomic_write_json(out_path, payload)
    return payload


def _arm_generate_kwargs(arm: EvalArm, gen: dict) -> dict:
    """Keyword arguments shared by every generation call for one arm."""
    return dict(
        budget=arm.budget,
        backend=arm.backend,
        anchor_mode=arm.anchor_mode,
        anchor_frac=arm.anchor_frac,
        sig=arm.sig,
        policy_params=arm.params or (gen["policy_params"] or {}).get(arm.backend, {}),
        recent=gen["recent"],
        sink=gen["sink"],
        evict_every=gen["evict_every"],
        obs_window=gen["obs_window"],
        obs_decay=gen["obs_decay"],
        max_new=gen["max_new"],
        do_sample=gen["do_sample"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        execution_backend=gen["execution_backend"],
        cache_manager=gen["cache_manager"],
    )


def _generate_arm(
    model,
    tokenizer,
    prompt_ids,
    arm: EvalArm,
    gen: dict,
    *,
    prompt_mask=None,
    debug_path: Path | None = None,
    debug_topk: int = 0,
) -> list[list[dict]]:
    """Generate ``num_return_sequences`` candidates for every prompt row.

    ``prompt_ids`` is ``[num_prompts, P]`` (left padded when ``prompt_mask`` is
    given).  Candidates are generated in micro-batches of ``candidate_batch_size``
    per prompt; candidate ``j`` always uses seed ``seed + seed_offset + j``.
    A single unpadded prompt with one candidate takes the solo path, the only
    one that supports debug dumps.  Returns one candidate list per prompt.
    """
    kwargs = _arm_generate_kwargs(arm, gen)
    num_prompts = int(prompt_ids.shape[0])
    total = gen["num_return_sequences"]
    base_seed = gen["seed"] + gen["seed_offset"]
    if debug_path is not None and (total > 1 or num_prompts > 1 or prompt_mask is not None):
        raise ValueError("debug traces currently require one unpadded prompt and num_return_sequences=1")
    if total == 1 and num_prompts == 1 and prompt_mask is None:
        result = generate_token_evict(
            model, tokenizer, prompt_ids, seed=base_seed,
            debug_path=str(debug_path) if debug_path else None, debug_topk=debug_topk,
            **kwargs,
        )
        return [[result]]

    per_prompt: list[list[dict]] = [[] for _ in range(num_prompts)]
    start = 0
    while start < total:
        count = min(gen["candidate_batch_size"], total - start)
        seeds = [base_seed + start + j for _ in range(num_prompts) for j in range(count)]
        result = generate_token_evict_batch(
            model,
            tokenizer,
            prompt_ids.repeat_interleave(count, dim=0),
            batch_size=len(seeds),
            input_attention_mask=(
                None if prompt_mask is None else prompt_mask.repeat_interleave(count, dim=0)
            ),
            seeds=seeds,
            **kwargs,
        )
        for prompt_idx in range(num_prompts):
            per_prompt[prompt_idx].extend(result["candidates"][prompt_idx * count:(prompt_idx + 1) * count])
        start += count
    return per_prompt


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
    log_mode: str = "full",
    execution_backend: ExecutionBackend | None = None,
    cache_manager: CacheManager | None = None,
    resume: bool = True,
    resume_from: list[str | Path] | None = None,
) -> dict:
    """Run every arm on every problem and write the result file.

    With ``resume`` (the default) an existing ``out_path`` written by the same
    configuration is continued: its problems are kept and skipped.  A file from
    a different configuration raises ``ResumeMismatch``; ``resume=False``
    overwrites it.  ``resume_from`` lists further result files (other ranks'
    shards) whose problems are also skipped but not copied into this file.
    """
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
    if log_mode not in _LOG_MODES:
        raise ValueError(f"log_mode must be one of {sorted(_LOG_MODES)}, got {log_mode!r}")
    if debug_dir and problem_batch_size > 1:
        raise ValueError("debug traces currently require problem_batch_size=1")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir) if debug_dir else None
    run_config = run_fingerprint(
        model, arms=arms, policy_params=policy_params, only_ids=only_ids,
        dataset=dataset, n=n, max_new=max_new, recent=recent, sink=sink,
        evict_every=evict_every, obs_window=obs_window, obs_decay=obs_decay,
        do_sample=do_sample, temperature=temperature, top_p=top_p, seed=seed,
        seed_offset=seed_offset, num_return_sequences=num_return_sequences,
        batch_size=batch_size, problem_batch_size=problem_batch_size,
        prompt_bucket_size=prompt_bucket_size, log_mode=log_mode,
        shard_rank=shard_rank, shard_world_size=shard_world_size,
    )
    records: list[dict] = []
    skip_ids: set[str] = set()
    if resume:
        records, complete = load_resumable(out_path, run_config)
        if complete is not None:
            print(f"{progress_prefix}{out_path} is already complete; nothing to run", flush=True)
            return complete
        for other in resume_from or []:
            other_records, _ = load_resumable(Path(other), run_config)
            skip_ids.update(record["id"] for record in other_records)
        if records:
            print(f"{progress_prefix}resuming {out_path}: {len(records)} problems done", flush=True)
    skip_ids.update(record["id"] for record in records)

    problems = load_problems(dataset, n)
    if only_ids:
        problems = [problem for problem in problems if problem["id"] in only_ids]
    order = {problem["id"]: index for index, problem in enumerate(problems)}
    if claim_problem_index is None:
        problems = problems[shard_rank::shard_world_size]

        def next_problem():
            for local_index, problem in enumerate(problems, start=1):
                if problem["id"] not in skip_ids:
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
                if problems[problem_index]["id"] not in skip_ids:
                    yield problem_index + 1, len(problems), problems[problem_index]

    gen = dict(
        recent=recent, sink=sink, evict_every=evict_every, obs_window=obs_window,
        obs_decay=obs_decay, max_new=max_new, do_sample=do_sample,
        temperature=temperature, top_p=top_p, seed=seed, seed_offset=seed_offset,
        num_return_sequences=num_return_sequences, candidate_batch_size=batch_size,
        policy_params=policy_params, execution_backend=execution_backend,
        cache_manager=cache_manager,
    )

    def write(final: bool) -> dict:
        records.sort(key=lambda record: order.get(record["id"], len(order)))
        return _write_result(out_path, records, log_mode, final=final, run_config=run_config)

    if problem_batch_size > 1:
        _run_cross_problem_eval(
            model=model,
            tokenizer=tokenizer,
            problem_iter=next_problem(),
            arms=arms,
            records=records,
            write=write,
            gen=gen,
            progress_prefix=progress_prefix,
            problem_batch_size=problem_batch_size,
            prompt_bucket_size=(
                prompt_bucket_size
                or (problem_batch_size if claim_problem_index is not None else problem_batch_size * 4)
            ),
            log_mode=log_mode,
        )
        return write(final=True)

    device = next(model.parameters()).device
    for progress_index, progress_total, problem in next_problem():
        row = {"id": problem["id"], "gold": problem["answer"], "arms": {}}
        input_ids = build_input_ids(tokenizer, problem["question"], device)
        for arm in arms:
            debug_path = debug_dir / f"{problem['id']}_{arm.name}.json" if debug_dir else None
            candidate_results = _generate_arm(
                model, tokenizer, input_ids, arm, gen,
                debug_path=debug_path, debug_topk=debug_topk,
            )[0]
            row["arms"][arm.name] = arm_result = _arm_payload(
                candidate_results, problem["answer"], seed, seed_offset, log_mode=log_mode,
            )
            result = candidate_results[0]
            print(
                f"{progress_prefix}[{progress_index}/{progress_total}] "
                f"{problem['id']} {arm.name:12s} "
                f"ok={arm_result['ok']} pass@1={arm_result['pass_at_1']:.3f} "
                f"candidates={num_return_sequences} len={len(result['gen_ids'])} cache={result['final_cache_len']} "
                f"tok/s={result['tokens_per_sec']:.2f}",
                flush=True,
            )
        records.append(row)
        write(final=False)

    return write(final=True)


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


def _evaluated_candidates(
    candidate_results,
    gold: str,
    seed: int,
    seed_offset: int,
    *,
    log_mode: str,
):
    evaluated = []
    for candidate_idx, result in enumerate(candidate_results):
        check_candidate(result)
        pred = extract_answer(result["text"])
        ok = is_correct(pred, gold)
        candidate = {
            "candidate_idx": candidate_idx,
            "seed": seed + seed_offset + candidate_idx,
            "ok": ok,
            "pred": pred,
            "gen_len": len(result["gen_ids"]),
            "batch_elapsed_sec": result["batch_elapsed_sec"],
            "batch_tokens_per_sec": result["batch_tokens_per_sec"],
            "batch_size": result["batch_size"],
        }
        if log_mode == "full":
            candidate["raw_output"] = result["text"]
            candidate.update(
                {key: value for key, value in result.items() if key not in {"text", "gen_ids"}}
            )
        else:
            # Brief mode keeps only scalar fields needed to compute the exact summary.
            candidate.update({
                key: result[key]
                for key in (
                    "n_evict",
                    "final_cache_len",
                    "elapsed_sec",
                    "tokens_per_sec",
                    "peak_memory_bytes",
                    "mean_compression_ratio",
                )
            })
            candidate.update({
                key: value
                for key, value in result.items()
                if key not in {"text", "gen_ids"} and isinstance(value, (int, float, bool))
            })
        evaluated.append(candidate)
    return evaluated


def _arm_payload(
    candidate_results,
    gold: str,
    seed: int,
    seed_offset: int,
    *,
    log_mode: str = "full",
):
    evaluated = _evaluated_candidates(
        candidate_results, gold, seed, seed_offset, log_mode=log_mode
    )
    first_result = candidate_results[0]
    first = evaluated[0]
    payload = {
        "ok": first["ok"],
        "pred": first["pred"],
        "gen_len": len(first_result["gen_ids"]),
        "n_evict": first_result["n_evict"],
        "final_cache_len": first_result["final_cache_len"],
        "elapsed_sec": first_result["elapsed_sec"],
        "tokens_per_sec": first_result["tokens_per_sec"],
        "peak_memory_bytes": first_result["peak_memory_bytes"],
        "mean_compression_ratio": first_result["mean_compression_ratio"],
        "num_return_sequences": len(evaluated),
        "pass_at_1": sum(candidate["ok"] for candidate in evaluated) / len(evaluated),
        "candidates": evaluated,
    }
    # Batched generation reports both per-request and aggregate throughput. Keep
    # the latter at arm level so serving-style smoke tests do not lose it when
    # candidates are serialized in brief mode.
    payload["batch_elapsed_sec"] = first_result["batch_elapsed_sec"]
    payload["batch_tokens_per_sec"] = first_result["batch_tokens_per_sec"]
    payload["batch_size"] = first_result["batch_size"]
    if log_mode == "full":
        payload["raw_output"] = first_result["text"]
        payload["evict_events"] = first_result["evict_events"]
        payload.update({key: first_result[key] for key in _EXTRA_METRIC_KEYS if key in first_result})
    else:
        payload.update({
            key: value
            for key, value in first_result.items()
            if key not in {"text", "gen_ids"} and isinstance(value, (int, float, bool))
        })
    return payload


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
    records,
    write,
    gen,
    progress_prefix,
    problem_batch_size,
    prompt_bucket_size,
    log_mode="full",
):
    """Evaluate length-bucketed heterogeneous prompts in static batches.

    Finished records are appended to ``records`` and flushed with ``write``.
    """
    device = next(model.parameters()).device
    iterator = iter(problem_iter)

    while True:
        raw_window = _take_window(iterator, prompt_bucket_size)
        if not raw_window:
            break
        tokenized = []
        for progress_index, progress_total, problem in raw_window:
            ids = build_input_ids(tokenizer, problem["question"], device)
            tokenized.append((int(ids.numel()), progress_index, progress_total, problem, ids))
        tokenized.sort(key=lambda item: item[0])

        for group_start in range(0, len(tokenized), problem_batch_size):
            group = tokenized[group_start:group_start + problem_batch_size]
            padded_ids, prompt_mask, _ = pad_input_ids(tokenizer, [item[4] for item in group], device)
            group_records = [
                {"id": item[3]["id"], "gold": item[3]["answer"], "arms": {}}
                for item in group
            ]
            for arm in arms:
                per_problem_results = _generate_arm(
                    model, tokenizer, padded_ids, arm, gen, prompt_mask=prompt_mask,
                )
                for problem_idx, record in enumerate(group_records):
                    record["arms"][arm.name] = _arm_payload(
                        per_problem_results[problem_idx],
                        record["gold"],
                        gen["seed"],
                        gen["seed_offset"],
                        log_mode=log_mode,
                    )

            for item, record in zip(group, group_records):
                representative = next(iter(record["arms"].values()))
                print(
                    f"{progress_prefix}[{item[1]}/{item[2]}] "
                    f"{record['id']} cross_batch={len(group)} "
                    f"candidates={gen['num_return_sequences']} tok/s={representative['tokens_per_sec']:.2f}",
                    flush=True,
                )
                records.append(record)
            write(final=False)
