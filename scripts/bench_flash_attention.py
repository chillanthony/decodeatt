"""Benchmark eager/FlashAttention decode paths on one problem.

This isolates attention implementation overhead from the eviction policy.  The
hybrid mode uses a FlashAttention model for normal decode steps and an eager
model only during the same observation windows used by runner_token.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rkv.runner_evict import _sample
from scripts.gen_traces import load_dataset_problems


def _load_model(model_name: str, attn_impl: str):
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_impl,
    ).eval()
    torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def _sync_free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _want_observation(step: int, evict_every: int, obs_window: int) -> bool:
    to_evict = evict_every - ((step + 1) % evict_every or evict_every)
    return to_evict < obs_window


@torch.no_grad()
def _decode(
    *,
    model,
    tokenizer,
    input_ids,
    max_new: int,
    evict_every: int,
    obs_window: int,
    observe: bool,
    fixed_tokens: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos = tokenizer.eos_token_id
    attention_checksum = torch.zeros((), device=device)

    out = model(input_ids=input_ids, use_cache=True, output_attentions=False)
    cache = out.past_key_values
    logits = out.logits[:, -1]

    gen = []
    eos_seen = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(max_new):
        true_pos = prompt_len + step
        nxt = _sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            eos_seen = True
            if not fixed_tokens:
                break

        cache_len = prompt_len + step
        want_attn = observe and _want_observation(step, evict_every, obs_window)
        out = model(
            input_ids=torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
            attention_mask=torch.ones(1, cache_len + 1, device=device, dtype=torch.long),
            position_ids=torch.tensor([[true_pos]], device=device),
            cache_position=torch.tensor([cache_len], device=device),
        )
        cache = out.past_key_values
        logits = out.logits[:, -1]
        if want_attn:
            for attn in out.attentions:
                attention_checksum = attention_checksum + attn[0, :, -1, :].mean()

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "gen_tokens": len(gen),
        "prompt_tokens": prompt_len,
        "eos_seen": eos_seen,
        "generate_seconds": gen_time,
        "tokens_per_second": len(gen) / gen_time if gen_time else None,
        "peak_allocated_gb": peak_gb,
        "attention_checksum": float(attention_checksum.detach().cpu()),
    }


@torch.no_grad()
def _decode_hybrid(
    *,
    flash_model,
    eager_model,
    tokenizer,
    input_ids,
    max_new: int,
    evict_every: int,
    obs_window: int,
    fixed_tokens: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
):
    torch.manual_seed(seed)
    device = next(flash_model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos = tokenizer.eos_token_id
    attention_checksum = torch.zeros((), device=device)

    out = flash_model(input_ids=input_ids, use_cache=True, output_attentions=False)
    cache = out.past_key_values
    logits = out.logits[:, -1]

    gen = []
    eos_seen = False
    observed_steps = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(max_new):
        true_pos = prompt_len + step
        nxt = _sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            eos_seen = True
            if not fixed_tokens:
                break

        cache_len = prompt_len + step
        want_attn = _want_observation(step, evict_every, obs_window)
        model = eager_model if want_attn else flash_model
        out = model(
            input_ids=torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
            attention_mask=torch.ones(1, cache_len + 1, device=device, dtype=torch.long),
            position_ids=torch.tensor([[true_pos]], device=device),
            cache_position=torch.tensor([cache_len], device=device),
        )
        cache = out.past_key_values
        logits = out.logits[:, -1]
        if want_attn:
            observed_steps += 1
            for attn in out.attentions:
                attention_checksum = attention_checksum + attn[0, :, -1, :].mean()

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "gen_tokens": len(gen),
        "prompt_tokens": prompt_len,
        "eos_seen": eos_seen,
        "observed_steps": observed_steps,
        "generate_seconds": gen_time,
        "tokens_per_second": len(gen) / gen_time if gen_time else None,
        "peak_allocated_gb": peak_gb,
        "attention_checksum": float(attention_checksum.detach().cpu()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--problem-index", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--evict-every", type=int, default=128)
    ap.add_argument("--obs-window", type=int, default=16)
    ap.add_argument("--fixed-tokens", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/flash_attention_bench.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    problems = load_dataset_problems(args.dataset, max(args.n, args.problem_index + 1))
    problem = problems[args.problem_index]
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem["question"]}],
        add_generation_prompt=True,
        return_tensors="pt",
    )

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "problem_id": problem["id"],
        "max_new": args.max_new,
        "fixed_tokens": args.fixed_tokens,
        "evict_every": args.evict_every,
        "obs_window": args.obs_window,
        "modes": {},
    }

    for name, impl, observe in [
        ("eager", "eager", True),
        ("flash", "flash_attention_2", False),
    ]:
        _sync_free()
        model, load_seconds = _load_model(args.model, impl)
        rec = _decode(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new=args.max_new,
            evict_every=args.evict_every,
            obs_window=args.obs_window,
            observe=observe,
            fixed_tokens=args.fixed_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        rec["load_seconds"] = load_seconds
        rec["attn_implementation"] = impl
        rec["observe"] = observe
        results["modes"][name] = rec
        del model

    _sync_free()
    flash_model, flash_load = _load_model(args.model, "flash_attention_2")
    eager_model, eager_load = _load_model(args.model, "eager")
    hybrid = _decode_hybrid(
        flash_model=flash_model,
        eager_model=eager_model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new=args.max_new,
        evict_every=args.evict_every,
        obs_window=args.obs_window,
        fixed_tokens=args.fixed_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    hybrid["load_seconds"] = flash_load + eager_load
    hybrid["attn_implementation"] = "flash_attention_2+eager"
    hybrid["observe"] = True
    results["modes"]["hybrid"] = hybrid
    del flash_model, eager_model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
