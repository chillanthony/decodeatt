"""Run a generation-time KV eviction policy benchmark.

Config loading/validation lives in ``kvbench.config``; result merging and
distributed sharding helpers in ``kvbench.sharding``.  This script only parses
the CLI and orchestrates model loading per attention backend.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import timedelta

import torch
import yaml

from kvbench.config import ConfigError, load_run_config, resolve_settings
from kvbench.datasets import load_problems
from kvbench.evaluator import run_generation_eval
from kvbench.models import load_causal_lm
from kvbench.resume import ResumeMismatch
from kvbench.sharding import (
    allocate_group_ranks,
    claim_next_problem,
    distributed_shard_path,
    merge_payloads,
)
from kvbench.vllm_evaluator import run_vllm_generation_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default=None, help="Model-only YAML for one run.")
    parser.add_argument("--data-config", default=None, help="Dataset-only YAML for one run.")
    parser.add_argument("--print-config", action="store_true", help="Print the effective config and exit.")
    parser.add_argument("--strategy-config", default=None, help="Strategy-only YAML for one run.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None, choices=["sample", "aime", "math500", "mix"])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument(
        "--answer-extraction",
        choices=["boxed", "lenient"],
        default=None,
        help="boxed (default): only the last \\boxed{} counts; lenient: legacy fallbacks.",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="Comma-separated arms, e.g. fullkv,snapkv@1024,h2o@1024,streamingllm@1024,rkv@1024",
    )
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--recent", type=int, default=None)
    parser.add_argument("--sink", type=int, default=None)
    parser.add_argument("--evict-every", type=int, default=None)
    parser.add_argument("--obs-window", type=int, default=None)
    parser.add_argument("--obs-decay", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn", default=None, help="Attention backend, or auto for per-strategy selection")
    parser.add_argument("--fast-attn", default=None, help="Fast backend used by --attn auto, default sdpa")
    parser.add_argument(
        "--fast-observe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With --attn auto, run cache-aware policies (rkv/snapkv/h2o) on the fast backend "
        "(default on): their attention is recomputed from hidden states, so eager is not needed. "
        "--no-fast-observe keeps them on eager, as runs before 2026-11-13 did.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--problem-batch-size", type=int, default=None)
    parser.add_argument("--prompt-bucket-size", type=int, default=None)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Shard attention groups and problems across torchrun ranks.",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=120,
        help="Timeout for distributed control-plane synchronization, default 120 minutes.",
    )
    parser.add_argument("--only-ids", default="")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--debug-topk", type=int, default=None)
    parser.add_argument(
        "--policy-param",
        action="append",
        default=[],
        help="Override strategy parameter as policy.key=value, e.g. rkv.redundancy_lambda=0.7",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--cache-impl",
        choices=["growable", "dynamic"],
        default=None,
        help="KV cache storage: growable (default, in-place) or HF dynamic (torch.cat per step).",
    )
    parser.add_argument(
        "--sync-timing",
        action="store_true",
        default=None,
        help="Synchronize CUDA around every timed phase so prefill/forward/eviction "
        "timings are exact (slightly lowers throughput).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start over instead of resuming an existing --out file.",
    )
    parser.add_argument(
        "--log-mode",
        choices=["full", "brief"],
        default=None,
        help="Result detail level: full keeps generations/events; brief writes summary only.",
    )
    args = parser.parse_args()

    try:
        settings = resolve_settings(
            load_run_config(args.model_config, args.data_config, args.strategy_config), args
        )
    except (ConfigError, ValueError, TypeError) as exc:
        raise SystemExit(f"config error: {exc}") from None
    if args.print_config:
        print(yaml.safe_dump(settings.resolved(), sort_keys=False, allow_unicode=True), end="")
        return
    groups, vllm_arms = settings.attn_groups()
    eval_kwargs = settings.eval_kwargs
    out_path, log_mode = settings.out_path, settings.log_mode
    eval_kwargs = {**eval_kwargs, "resume": not args.overwrite}
    if settings.model_path != settings.model_name:
        print(f"[model] {settings.model_name} -> {settings.model_path}", flush=True)

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or env_world_size > 1
    if distributed and vllm_arms:
        raise SystemExit("engine=vllm is not supported with --distributed yet")
    if distributed:
        if not torch.distributed.is_available():
            raise SystemExit("torch.distributed is not available in this PyTorch build")
        if env_world_size <= 1:
            raise SystemExit("--distributed must be launched with torchrun and WORLD_SIZE > 1")
        if args.distributed_timeout_minutes < 1:
            raise SystemExit("--distributed-timeout-minutes must be at least 1")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # Ranks run independent inference jobs; collectives are control-plane barriers only.
        # Gloo avoids tying those long, imbalanced waits to NCCL's GPU watchdog.
        dist_backend = "gloo"
        torch.distributed.init_process_group(
            backend=dist_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        assignments = allocate_group_ranks(groups, world_size)
        attn_backend, group_rank, group_world_size = assignments[rank]
        group_arms = groups[attn_backend]
        shard_out = distributed_shard_path(out_path, rank, attn_backend)
        # Resume: skip problems already finished in any shard of this backend.
        backend_shards = [
            distributed_shard_path(out_path, shard_rank, backend)
            for shard_rank, (backend, _, _) in enumerate(assignments)
            if backend == attn_backend and shard_rank != rank
        ]
        stale = sorted(
            str(path) for path in shard_out.parent.glob(f"{out_path.stem}.rank*{out_path.suffix}")
            if path not in {distributed_shard_path(out_path, r, b) for r, (b, _, _) in enumerate(assignments)}
        ) if eval_kwargs["resume"] and shard_out.parent.exists() else []
        if stale:
            raise SystemExit(
                f"shard files from a run with a different world size or grouping: {', '.join(stale)}; "
                "rerun with the same number of ranks or pass --overwrite"
            )
        store = torch.distributed.distributed_c10d._get_default_store()
        queue_key = f"kvbench:next_problem:{attn_backend}"

        if torch.cuda.is_available():
            distributed_device_map: str | dict = {"": f"cuda:{local_rank}"}
        else:
            distributed_device_map = "cpu"

        try:
            # Populate the shared dataset cache once before the other ranks read it.
            if rank == 0:
                for backend in groups:
                    store.set(f"kvbench:next_problem:{backend}", "0")
                load_problems(eval_kwargs["dataset"], eval_kwargs["n"])
            torch.distributed.barrier()

            def claim_problem_index() -> int:
                return claim_next_problem(store, queue_key)

            print(
                f"\n=== rank={rank}/{world_size} local_rank={local_rank} "
                f"control_backend={dist_backend} "
                f"timeout_min={args.distributed_timeout_minutes} "
                f"scheduler=dynamic "
                f"attn={attn_backend} group_shard={group_rank}/{group_world_size} "
                f"arms={', '.join(arm.name for arm in group_arms)} ===",
                flush=True,
            )
            model, tokenizer = load_causal_lm(
                settings.model_path,
                dtype=settings.dtype,
                device_map=distributed_device_map,
                attn_implementation=attn_backend,
            )
            run_generation_eval(
                model,
                tokenizer,
                arms=group_arms,
                out_path=shard_out,
                claim_problem_index=claim_problem_index,
                progress_prefix=f"[rank {rank}] ",
                resume_from=[path for path in backend_shards if path.exists()],
                **eval_kwargs,
            )
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            torch.distributed.barrier()
            if rank == 0:
                shard_payloads = []
                for shard_rank, (backend, _, _) in enumerate(assignments):
                    shard_path = distributed_shard_path(out_path, shard_rank, backend)
                    shard_payloads.append(json.loads(shard_path.read_text()))
                payload = merge_payloads(shard_payloads, out_path, log_mode=log_mode)
                print("\n=== Distributed KV eviction eval ===")
                print(json.dumps(payload["summary"], ensure_ascii=False, indent=1))
                print(f"\nMerged {world_size} rank shards into {out_path}")
            torch.distributed.barrier()
        finally:
            torch.distributed.destroy_process_group()
        return

    payloads = []
    if vllm_arms:
        vllm_out = out_path
        if groups:
            vllm_out = out_path.with_name(f"{out_path.stem}.vllm{out_path.suffix}")
        payloads.append(
            run_vllm_generation_eval(
                model_name=settings.model_path,
                dataset=eval_kwargs["dataset"],
                n=eval_kwargs["n"],
                arms=vllm_arms,
                out_path=vllm_out,
                max_new=eval_kwargs["max_new"],
                do_sample=eval_kwargs["do_sample"],
                temperature=eval_kwargs["temperature"],
                top_p=eval_kwargs["top_p"],
                seed=eval_kwargs["seed"],
                num_return_sequences=eval_kwargs["num_return_sequences"],
                seed_offset=eval_kwargs["seed_offset"],
                log_mode=log_mode,
                debug_dir=eval_kwargs["debug_dir"],
                only_ids=eval_kwargs["only_ids"],
                vllm_config=settings.cfg.get("vllm", {}),
                prompt_template=eval_kwargs["prompt_template"],
                answer_extraction=eval_kwargs["answer_extraction"],
            )
        )
    for attn_backend, group_arms in groups.items():
        print(
            f"\n=== Loading model with attn={attn_backend} for arms: "
            f"{', '.join(arm.name for arm in group_arms)} ===",
            flush=True,
        )
        model, tokenizer = load_causal_lm(
            settings.model_path,
            dtype=settings.dtype,
            device_map=settings.device_map,
            attn_implementation=attn_backend,
        )
        group_out = out_path
        if len(groups) > 1:
            group_out = out_path.with_name(f"{out_path.stem}.{attn_backend}{out_path.suffix}")
        payloads.append(
            run_generation_eval(
                model,
                tokenizer,
                arms=group_arms,
                out_path=group_out,
                **eval_kwargs,
            )
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = (
        payloads[0]
        if len(payloads) == 1
        else merge_payloads(payloads, out_path, log_mode=log_mode)
    )
    print("\n=== KV eviction eval ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        main()
    except ResumeMismatch as exc:
        raise SystemExit(f"resume error: {exc}") from None
