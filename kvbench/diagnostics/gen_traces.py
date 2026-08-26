"""Generate reproducible FullKV trajectories for TideProbe diagnostics."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from kvbench.datasets import load_problems
from kvbench.models import build_input_ids, load_causal_lm


def _nested_get(config: dict, path: tuple[str, ...], default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _pick(cli_value, config: dict, path: tuple[str, ...], default=None):
    if cli_value is not None:
        return cli_value
    value = _nested_get(config, path)
    return default if value is None else value


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _distributed_context() -> tuple[int, int, int]:
    """Return rank, world size and local rank without requiring collectives."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid distributed context rank={rank}, world_size={world_size}")
    return rank, world_size, local_rank


def _seed_everything(seed: int) -> None:
    """Reset every relevant RNG before each trajectory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def token_statistics(logits_steps, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-model entropy and NLL in nats for each generated token."""
    if len(logits_steps) != token_ids.numel():
        raise ValueError(
            f"logit/token length mismatch: {len(logits_steps)} != {token_ids.numel()}"
        )
    entropies = []
    nlls = []
    for logits, token_id in zip(logits_steps, token_ids):
        logits = logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropies.append(-(probs * log_probs).nan_to_num().sum())
        nlls.append(-log_probs[int(token_id)])
    if not entropies:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty.clone()
    return torch.stack(entropies).cpu(), torch.stack(nlls).cpu()


@torch.no_grad()
def generate_trace(
    model,
    tokenizer,
    problem: dict,
    *,
    problem_index: int,
    model_name: str,
    dataset: str,
    max_new: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    attn_implementation: str,
) -> dict:
    """Generate one FullKV trace and retain teacher-forcing targets/statistics."""
    _seed_everything(seed)
    device = next(model.parameters()).device
    input_ids = build_input_ids(tokenizer, problem["question"], device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    generation_kwargs = {
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_logits": True,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    output = model.generate(input_ids, **generation_kwargs)
    prompt_length = input_ids.shape[1]
    generated_ids = output.sequences[0, prompt_length:].detach().cpu()
    entropy, fullkv_nll = token_statistics(output.logits, generated_ids)

    return {
        "schema_version": 1,
        "id": problem["id"],
        "problem_index": problem_index,
        "question": problem["question"],
        "answer": problem["answer"],
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "generation": {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "attn_implementation": attn_implementation,
            "cache_policy": "fullkv",
        },
        "prompt_ids": input_ids[0].detach().cpu(),
        "gen_ids": generated_ids,
        "token_text": [tokenizer.decode([int(token_id)]) for token_id in generated_ids],
        "step_entropy": entropy,
        "fullkv_nll": fullkv_nll,
        "gen_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", choices=["sample", "aime", "math500", "mix"], default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate complete trace files that already exist."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    rank, world_size, local_rank = _distributed_context()

    model_name = _pick(args.model, config, ("model", "name"))
    if not model_name:
        raise SystemExit("missing model name")
    dataset = _pick(args.dataset, config, ("data", "dataset"), "aime")
    count = int(_pick(args.n, config, ("data", "n"), 10))
    max_new = int(_pick(args.max_new, config, ("generation", "max_new"), 32768))
    base_seed = int(_pick(args.seed, config, ("generation", "seed"), 0))
    do_sample = not args.greedy and bool(
        _pick(None, config, ("generation", "do_sample"), True)
    )
    temperature = float(_pick(args.temperature, config, ("generation", "temperature"), 0.6))
    top_p = float(_pick(args.top_p, config, ("generation", "top_p"), 0.95))
    dtype = _pick(args.dtype, config, ("model", "dtype"), "bfloat16")
    attn = _pick(args.attn, config, ("model", "fast_attn"), "sdpa")
    trace_dir = Path(
        _pick(args.trace_dir, config, ("diagnostics", "trace_dir"), "/home/ma-user/work/trace")
    )
    trace_dir.mkdir(parents=True, exist_ok=True)

    if count < 1 or max_new < 1:
        raise SystemExit("n and max-new must be positive")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: str | dict = {"": f"cuda:{local_rank}"}
    else:
        device_map = "cpu"

    problems = load_problems(dataset, count)
    assigned = list(enumerate(problems))[rank::world_size]
    print(
        f"[rank {rank}/{world_size}] model={model_name} dataset={dataset} "
        f"assigned={len(assigned)}/{len(problems)} max_new={max_new} output={trace_dir}",
        flush=True,
    )
    if not assigned:
        return

    model, tokenizer = load_causal_lm(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn,
    )
    for local_progress, (problem_index, problem) in enumerate(assigned, start=1):
        output_path = trace_dir / f"{problem['id']}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"[rank {rank}] skip existing {output_path}", flush=True)
            continue
        trace = generate_trace(
            model,
            tokenizer,
            problem,
            problem_index=problem_index,
            model_name=model_name,
            dataset=dataset,
            max_new=max_new,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            seed=base_seed,
            attn_implementation=attn,
        )
        _atomic_torch_save(trace, output_path)
        mean_nll = trace["fullkv_nll"].mean().item() if trace["fullkv_nll"].numel() else float("nan")
        print(
            f"[rank {rank} {local_progress}/{len(assigned)}] {problem['id']}: "
            f"tokens={trace['gen_ids'].numel()} mean_nll={mean_nll:.4f} -> {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
