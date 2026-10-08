"""Serving-style evaluator for arms explicitly assigned to the vLLM engine.

This module intentionally supports FullKV only for now.  The reference token
eviction policies require direct KV ownership, which vLLM does not expose
through ``LLM.generate``.  Such arms must stay on the HF engine until a
paged/block CacheManager is implemented.
"""
from __future__ import annotations

import time
from pathlib import Path

from kvbench.datasets import load_problems
from kvbench.evaluator import _arm_payload, _write_result


def _prompt_text(tokenizer, question: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )
    ids = tokenizer(question, add_special_tokens=True).input_ids
    return tokenizer.decode(ids, skip_special_tokens=False)


def _sampling_params(vllm, *, do_sample, temperature, top_p, max_new, n, seed):
    # vLLM uses temperature=0 for greedy decoding; top_p=1 avoids accidental
    # truncation in that mode while preserving the HF sampling contract.
    return vllm.SamplingParams(
        n=n,
        max_tokens=max_new,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        seed=seed,
    )


def run_vllm_generation_eval(
    *,
    model_name: str,
    dataset: str,
    n: int,
    arms: list,
    out_path: str | Path,
    max_new: int = 4096,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 0,
    num_return_sequences: int = 1,
    seed_offset: int = 0,
    log_mode: str = "full",
    debug_dir: str | Path | None = None,
    vllm_config: dict | None = None,
    only_ids: set[str] | None = None,
) -> dict:
    """Run explicitly vLLM-selected arms as one serving-style batch.

    vLLM owns scheduling and KV storage here, so ``problem_batch_size`` is not
    used to split requests.  Passing all prompts together measures aggregate
    engine throughput rather than the reference runner's static micro-batch.
    """
    unsupported = [arm.name for arm in arms if arm.backend != "fullkv"]
    if unsupported:
        raise ValueError(
            "vllm engine currently supports fullkv only; move these arms to "
            f"engine: hf: {', '.join(unsupported)}"
        )
    if debug_dir:
        raise ValueError("debug traces are not supported by the vllm engine")

    try:
        import vllm
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "engine=vllm requires the vllm and transformers packages"
        ) from exc

    options = dict(vllm_config or {})
    options.setdefault("dtype", "auto")
    options.setdefault("trust_remote_code", True)
    options = {key: value for key, value in options.items() if value is not None}
    llm = vllm.LLM(model=model_name, **options)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    problems = load_problems(dataset, n)
    if only_ids:
        problems = [problem for problem in problems if problem["id"] in only_ids]
    prompts = [_prompt_text(tokenizer, problem["question"]) for problem in problems]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": problem["id"], "gold": problem["answer"], "arms": {}}
        for problem in problems
    ]

    for arm in arms:
        sampling = _sampling_params(
            vllm,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new=max_new,
            n=num_return_sequences,
            seed=seed + seed_offset,
        )
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        elapsed = max(time.perf_counter() - start, 1e-9)
        total_tokens = sum(
            len(candidate.token_ids)
            for request in outputs
            for candidate in request.outputs
        )
        batch_tokens_per_sec = total_tokens / elapsed
        response_count = max(sum(len(request.outputs) for request in outputs), 1)
        per_response_elapsed = elapsed / response_count

        for record, request in zip(records, outputs):
            candidates = []
            prompt_len = len(getattr(request, "prompt_token_ids", []) or [])
            for candidate in request.outputs:
                gen_ids = list(candidate.token_ids)
                candidates.append({
                    "text": candidate.text,
                    "gen_ids": gen_ids,
                    "prompt_len": prompt_len,
                    "n_evict": 0,
                    "final_cache_len": prompt_len + len(gen_ids),
                    "elapsed_sec": per_response_elapsed,
                    "tokens_per_sec": len(gen_ids) / per_response_elapsed,
                    "batch_elapsed_sec": elapsed,
                    "batch_tokens_per_sec": batch_tokens_per_sec,
                    "batch_size": response_count,
                    "peak_memory_bytes": None,
                    "mean_compression_ratio": 1.0,
                    "evict_events": [],
                })
            record["arms"][arm.name] = _arm_payload(
                candidates,
                record["gold"],
                seed,
                seed_offset,
                log_mode=log_mode,
            )

    return _write_result(out_path, records, log_mode, final=True)


__all__ = ["run_vllm_generation_eval"]
