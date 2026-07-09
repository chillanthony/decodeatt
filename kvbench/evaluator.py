"""Generation evaluator for KV eviction policies."""
from __future__ import annotations

import json
from pathlib import Path

from kv_eviction.runner_token import generate_token_evict

from kvbench.datasets import load_problems
from kvbench.metrics import extract_answer, is_correct, summarize_accuracy
from kvbench.models import build_input_ids
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
    only_ids: set[str] | None = None,
    debug_dir: str | Path | None = None,
    debug_topk: int = 64,
    policy_params: dict | None = None,
) -> dict:
    problems = load_problems(dataset, n)
    if only_ids:
        problems = [problem for problem in problems if problem["id"] in only_ids]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir) if debug_dir else None

    records = []
    for index, problem in enumerate(problems):
        row = {"id": problem["id"], "gold": problem["answer"], "arms": {}}
        input_ids = build_input_ids(tokenizer, problem["question"], next(model.parameters()).device)
        for arm in arms:
            debug_path = None
            if debug_dir:
                debug_path = debug_dir / f"{problem['id']}_{arm.name}.json"
            result = generate_token_evict(
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
                seed=seed,
                policy_params=(policy_params or {}).get(arm.backend, {}),
                debug_path=str(debug_path) if debug_path else None,
                debug_topk=debug_topk,
            )
            pred = extract_answer(result["text"])
            ok = is_correct(pred, problem["answer"])
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
                **{key: result[key] for key in extra_metric_keys if key in result},
            }
            print(
                f"[{index + 1}/{len(problems)}] {problem['id']} {arm.name:12s} "
                f"ok={ok} len={len(result['gen_ids'])} cache={result['final_cache_len']} "
                f"tok/s={result['tokens_per_sec']:.2f}",
                flush=True,
            )
        records.append(row)
        out_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=1))

    summary = summarize_accuracy(records)
    payload = {"records": records, "summary": summary}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return payload
