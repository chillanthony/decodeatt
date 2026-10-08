"""Lightweight speed benchmark for FullKV vs SnapKV generation."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from kvbench.datasets import load_problems
from kvbench.metrics import extract_answer, is_correct
from kvbench.models import build_input_ids, load_causal_lm
from kvbench.policies import parse_arm
from kv_eviction.runner_token import _sample, generate_token_evict


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _pick(args, cfg: dict, name: str, default=None):
    value = getattr(args, name.replace("-", "_"), None)
    return cfg.get(name, default) if value is None else value


def _mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def _summarize(rows: list[dict]) -> dict:
    gen_tokens = [row["gen_len"] for row in rows]
    elapsed = [row["elapsed_sec"] for row in rows]
    toks = [row["tokens_per_sec"] for row in rows]
    obs = [row["attention_observation_sec"] for row in rows]
    evict = [row["eviction_sec_total"] for row in rows]
    forward = [row["decode_forward_sec"] for row in rows]
    total_gen = sum(gen_tokens)
    total_elapsed = sum(elapsed)
    return {
        "total": len(rows),
        "correct": sum(1 for row in rows if row["ok"]),
        "accuracy": sum(1 for row in rows if row["ok"]) / len(rows) if rows else 0.0,
        "total_gen_tokens": total_gen,
        "total_elapsed_sec": total_elapsed,
        "weighted_tokens_per_sec": total_gen / total_elapsed if total_elapsed > 0 else 0.0,
        "mean_tokens_per_sec": _mean(toks),
        "median_tokens_per_sec": _median(toks),
        "mean_gen_len": _mean(gen_tokens),
        "mean_final_cache_len": _mean([row["final_cache_len"] for row in rows]),
        "mean_elapsed_sec": _mean(elapsed),
        "mean_decode_forward_sec": _mean(forward),
        "mean_attention_observation_sec": _mean(obs),
        "mean_eviction_sec_total": _mean(evict),
        "max_peak_memory_bytes": max((row["peak_memory_bytes"] or 0) for row in rows) if rows else None,
    }


def _legacy_cache_len(cache) -> int:
    if cache is None:
        return 0
    if isinstance(cache, tuple):
        return int(cache[0][0].shape[-2])
    if hasattr(cache, "layers"):
        return int(cache.layers[0].keys.shape[-2])
    if hasattr(cache, "key_cache") and cache.key_cache:
        return int(cache.key_cache[0].shape[-2])
    return 0


@torch.no_grad()
def generate_integrated_qwen_snapkv(
    model,
    tokenizer,
    input_ids,
    *,
    controller,
    max_new: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict:
    device = next(model.parameters()).device
    if do_sample:
        torch.manual_seed(seed)
    input_ids = input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    eos = tokenizer.eos_token_id
    controller.reset()
    controller.enabled = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        start = time.perf_counter()
        prefill_start = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True, output_attentions=False)
        prefill_sec = time.perf_counter() - prefill_start
        cache = out.past_key_values
        cache_len = _legacy_cache_len(cache)
        logits = out.logits[:, -1]
        gen = []
        decode_forward_sec = 0.0

        for step in range(max_new):
            nxt = _sample(logits, do_sample, temperature, top_p)
            gen.append(nxt)
            if nxt == eos:
                break
            position_ids = torch.tensor([[cache_len]], device=device)
            decode_start = time.perf_counter()
            out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                attention_mask=torch.ones(1, cache_len + 1, device=device, dtype=torch.long),
                position_ids=position_ids,
                cache_position=torch.tensor([cache_len], device=device),
            )
            decode_forward_sec += time.perf_counter() - decode_start
            cache = out.past_key_values
            logits = out.logits[:, -1]
            cache_len = controller.stats.cache_len or _legacy_cache_len(cache)
    finally:
        controller.enabled = False

    text = tokenizer.decode(gen, skip_special_tokens=True)
    elapsed = time.perf_counter() - start
    decode_sec = elapsed - prefill_sec
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "text": text,
        "gen_ids": gen,
        "n_evict": controller.stats.evictions,
        "final_cache_len": cache_len,
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(gen) / elapsed if elapsed > 0 else 0.0,
        "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "decode_forward_sec": decode_forward_sec,
        "attention_observation_sec": controller.stats.attention_observation_sec,
        "eviction_sec_total": controller.stats.eviction_sec_total,
        "other_decode_sec": max(
            0.0,
            decode_sec
            - decode_forward_sec
            - controller.stats.attention_observation_sec
            - controller.stats.eviction_sec_total,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rkv.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None, choices=["sample", "aime", "math500", "mix"])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--recent", type=int, default=None)
    parser.add_argument("--sink", type=int, default=None)
    parser.add_argument("--evict-every", type=int, default=None)
    parser.add_argument("--obs-window", type=int, default=None)
    parser.add_argument("--obs-decay", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--integrated-qwen-snapkv",
        action="store_true",
        help="Use the Qwen2 attention-integrated SnapKV prototype for the snapkv arm.",
    )
    parser.add_argument("--out", default="results/speed_fullkv_snapkv.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    model_name = args.model or cfg.get("model")
    if not model_name:
        raise SystemExit("missing --model or model in config")

    model, tokenizer = load_causal_lm(
        model_name,
        dtype=_pick(args, cfg, "dtype", "bfloat16"),
        device_map=_pick(args, cfg, "device_map", "cuda"),
        attn_implementation=_pick(args, cfg, "attn", "eager"),
    )
    arms = [parse_arm("fullkv"), parse_arm(f"snapkv@{args.budget}")]
    policy_params = cfg.get("policy_params") or {}
    snapkv_controller = None
    if args.integrated_qwen_snapkv:
        from kv_eviction.qwen_snapkv_patch import QwenSnapKVController, install_qwen2_snapkv

        snapkv_params = policy_params.get("snapkv", {})
        snapkv_controller = QwenSnapKVController(
            budget=args.budget,
            window_size=int(snapkv_params.get("window_size", _pick(args, cfg, "obs_window", 8))),
            kernel_size=int(snapkv_params.get("kernel_size", snapkv_params.get("pool_kernel", 7))),
            pooling=str(snapkv_params.get("pooling", "avgpool")),
        )
        install_qwen2_snapkv(model, snapkv_controller)
    problems = load_problems(_pick(args, cfg, "dataset", "sample"), _pick(args, cfg, "n", 2))

    records: list[dict] = []
    by_arm: dict[str, list[dict]] = {arm.name: [] for arm in arms}
    for problem_index, problem in enumerate(problems, start=1):
        input_ids = build_input_ids(tokenizer, problem["question"], next(model.parameters()).device)
        row = {"id": problem["id"], "gold": problem["answer"], "arms": {}}
        for arm in arms:
            if arm.backend == "snapkv" and snapkv_controller is not None:
                result = generate_integrated_qwen_snapkv(
                    model,
                    tokenizer,
                    input_ids,
                    controller=snapkv_controller,
                    max_new=_pick(args, cfg, "max_new", 512),
                    do_sample=not args.greedy and bool(cfg.get("do_sample", True)),
                    temperature=_pick(args, cfg, "temperature", 0.6),
                    top_p=_pick(args, cfg, "top_p", 0.95),
                    seed=_pick(args, cfg, "seed", 0),
                )
            else:
                result = generate_token_evict(
                    model,
                    tokenizer,
                    input_ids,
                    budget=arm.budget,
                    recent=_pick(args, cfg, "recent", 8),
                    sink=_pick(args, cfg, "sink", 0),
                    backend=arm.backend,
                    evict_every=_pick(args, cfg, "evict_every", 128),
                    obs_window=_pick(args, cfg, "obs_window", 8),
                    obs_decay=_pick(args, cfg, "obs_decay", 0.9),
                    max_new=_pick(args, cfg, "max_new", 512),
                    do_sample=not args.greedy and bool(cfg.get("do_sample", True)),
                    temperature=_pick(args, cfg, "temperature", 0.6),
                    top_p=_pick(args, cfg, "top_p", 0.95),
                    seed=_pick(args, cfg, "seed", 0),
                    policy_params=policy_params.get(arm.backend, {}),
                    debug_path=None,
                    debug_topk=0,
                )
            pred = extract_answer(result["text"])
            arm_record = {
                "ok": is_correct(pred, problem["answer"]),
                "pred": pred,
                "gen_len": len(result["gen_ids"]),
                "n_evict": result["n_evict"],
                "final_cache_len": result["final_cache_len"],
                "elapsed_sec": result["elapsed_sec"],
                "tokens_per_sec": result["tokens_per_sec"],
                "peak_memory_bytes": result["peak_memory_bytes"],
                "prefill_sec": result.get("prefill_sec", 0.0),
                "decode_sec": result.get("decode_sec", 0.0),
                "decode_forward_sec": result.get("decode_forward_sec", 0.0),
                "attention_observation_sec": result.get("attention_observation_sec", 0.0),
                "eviction_sec_total": result.get("eviction_sec_total", 0.0),
                "other_decode_sec": result.get("other_decode_sec", 0.0),
            }
            row["arms"][arm.name] = arm_record
            by_arm[arm.name].append(arm_record)
            print(
                f"[{problem_index}/{len(problems)}] {problem['id']} {arm.name:10s} "
                f"ok={arm_record['ok']} len={arm_record['gen_len']} "
                f"cache={arm_record['final_cache_len']} tok/s={arm_record['tokens_per_sec']:.2f} "
                f"obs={arm_record['attention_observation_sec']:.2f}s "
                f"evict={arm_record['eviction_sec_total']:.2f}s",
                flush=True,
            )
        records.append(row)

    payload = {
        "config": {
            "model": model_name,
            "dataset": _pick(args, cfg, "dataset", "sample"),
            "n": len(problems),
            "budget": args.budget,
            "max_new": _pick(args, cfg, "max_new", 512),
            "attn": _pick(args, cfg, "attn", "eager"),
            "integrated_qwen_snapkv": args.integrated_qwen_snapkv,
        },
        "summary": {arm: _summarize(rows) for arm, rows in by_arm.items()},
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print("\n=== Speed summary ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=1))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
