"""Serving-style evaluator for arms explicitly assigned to the vLLM engine.

This module intentionally supports FullKV only for now.  The reference token
eviction policies require direct KV ownership, which vLLM does not expose
through ``LLM.generate``.  Such arms must stay on the HF engine until a
paged/block CacheManager is implemented.

Candidates are built with ``kvbench.result_schema`` so vLLM and HF results
share one schema and can be compared or merged directly; once eviction runs
inside vLLM, its candidates only need to fill the eviction/timing fields.
"""
from __future__ import annotations

import time
from pathlib import Path

from kvbench.datasets import load_problems
from kvbench.evaluator import _arm_payload, _write_result
from kvbench.result_schema import no_eviction_candidate


def _heads_per_layer(model_name: str) -> list[int]:
    """KV heads per layer, used for the cache-length summary of a full cache."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = getattr(config, "text_config", None) or config
    kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    return [int(kv_heads)] * int(config.num_hidden_layers)


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
    heads_per_layer = _heads_per_layer(model_name)
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

    # vLLM rejects n > 1 for greedy decoding, so greedy candidates are issued
    # as repeated n=1 requests and regrouped per problem afterwards.
    repeats = 1 if do_sample else num_return_sequences
    request_prompts = [prompt for prompt in prompts for _ in range(repeats)]
    for arm in arms:
        sampling = _sampling_params(
            vllm,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new=max_new,
            n=num_return_sequences // repeats,
            seed=seed + seed_offset,
        )
        start = time.perf_counter()
        outputs = llm.generate(request_prompts, sampling_params=sampling, use_tqdm=False)
        elapsed = max(time.perf_counter() - start, 1e-9)
        total_tokens = sum(
            len(candidate.token_ids)
            for request in outputs
            for candidate in request.outputs
        )
        batch_tokens_per_sec = total_tokens / elapsed
        response_count = max(sum(len(request.outputs) for request in outputs), 1)
        per_response_elapsed = elapsed / response_count

        for index, record in enumerate(records):
            requests = outputs[index * repeats:(index + 1) * repeats]
            prompt_len = len(getattr(requests[0], "prompt_token_ids", []) or [])
            # vLLM seeds the request once for all n samples; candidates are
            # labelled seed + seed_offset + idx like the HF engine.
            candidates = [
                no_eviction_candidate(
                    text=candidate.text,
                    gen_ids=list(candidate.token_ids),
                    seed=seed + seed_offset + candidate_idx,
                    prompt_len=prompt_len,
                    heads_per_layer=heads_per_layer,
                    elapsed_sec=per_response_elapsed,
                    batch_elapsed_sec=elapsed,
                    batch_tokens_per_sec=batch_tokens_per_sec,
                    batch_size=response_count,
                )
                for candidate_idx, candidate in enumerate(
                    candidate for request in requests for candidate in request.outputs
                )
            ]
            record["arms"][arm.name] = _arm_payload(
                candidates,
                record["gold"],
                seed,
                seed_offset,
                log_mode=log_mode,
            )

    return _write_result(out_path, records, log_mode, final=True)


__all__ = ["run_vllm_generation_eval"]
