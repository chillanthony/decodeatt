"""Run a generation-time KV eviction policy benchmark."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from kvbench.evaluator import run_generation_eval
from kvbench.models import load_causal_lm
from kvbench.policies import parse_arms
from kvbench.metrics import summarize_accuracy


_FAST_ATTN_BACKENDS = {"fullkv", "streamingllm", "window", "random"}


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_config(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in {"policy_params", "policy_defaults"}:
            existing = merged.get("policy_defaults", merged.get("policy_params", {}))
            params = {name: dict(values or {}) for name, values in existing.items()}
            for policy, values in (value or {}).items():
                params.setdefault(policy, {}).update(values or {})
            merged["policy_defaults" if key == "policy_defaults" else "policy_params"] = params
        else:
            merged[key] = value
    return merged


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_policy_param_overrides(overrides: list[str]) -> dict:
    params: dict[str, dict] = {}
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(
                f"invalid --policy-param {item!r}; expected policy.key=value, "
                "e.g. rkv.redundancy_lambda=0.7"
            )
        left, value = item.split("=", 1)
        policy, key = left.split(".", 1)
        params.setdefault(policy, {})[key] = _parse_scalar(value)
    return params


def _policy_defaults(cfg: dict) -> dict:
    return {
        name: dict(values or {})
        for name, values in (cfg.get("policy_defaults") or cfg.get("policy_params") or {}).items()
    }


def _nested_get(cfg: dict, path: tuple[str, ...]):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _config_value(cfg: dict, flat_name: str, nested_path: tuple[str, ...], default=None):
    if flat_name in cfg and not isinstance(cfg.get(flat_name), dict):
        return cfg[flat_name]
    value = _nested_get(cfg, nested_path)
    return default if value is None else value


def _model_name(cfg: dict):
    model = cfg.get("model")
    if isinstance(model, dict):
        return model.get("name")
    return model


def _arm_attn_backend(backend: str, requested_attn: str, fast_attn: str) -> str:
    if requested_attn != "auto":
        return requested_attn
    return fast_attn if backend in _FAST_ATTN_BACKENDS else "eager"


def _merge_payloads(payloads: list[dict], out_path: str | Path) -> dict:
    rows_by_id: dict[str, dict] = {}
    order: list[str] = []
    for payload in payloads:
        for row in payload["records"]:
            problem_id = row["id"]
            if problem_id not in rows_by_id:
                rows_by_id[problem_id] = {
                    "id": problem_id,
                    "gold": row["gold"],
                    "arms": {},
                }
                order.append(problem_id)
            rows_by_id[problem_id]["arms"].update(row["arms"])
    records = [rows_by_id[problem_id] for problem_id in order]
    merged = {"records": records, "summary": summarize_accuracy(records)}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/math500_official_b1024.yaml")
    parser.add_argument(
        "--strategy-config",
        default=None,
        help="Optional overlay config; formal runs should prefer one complete configs/experiments/*.yaml",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None, choices=["sample", "aime", "math500", "mix"])
    parser.add_argument("--n", type=int, default=None)
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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
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
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.strategy_config:
        cfg = _merge_config(cfg, _load_config(args.strategy_config))

    def pick(name, default=None, nested_path: tuple[str, ...] | None = None):
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None:
            return value
        return _config_value(cfg, name, nested_path or (name,), default)

    model_name = args.model or _model_name(cfg)
    if not model_name:
        raise SystemExit("missing --model or model in config")

    policy_defaults = _policy_defaults(cfg)
    policy_overrides = _parse_policy_param_overrides(args.policy_param)
    raw_arms = args.arms or cfg.get(
        "arms",
        "fullkv,snapkv@1024,h2o@1024,streamingllm@1024,rkv@1024",
    )
    arms = parse_arms(raw_arms, policy_defaults=policy_defaults, policy_overrides=policy_overrides)
    requested_attn = pick("attn", "auto", ("model", "attn"))
    fast_attn = pick("fast_attn", "sdpa", ("model", "fast_attn"))
    out_path = Path(pick("out", "results/eval.json", ("experiment", "out")))
    config_only_ids = _config_value(cfg, "only_ids", ("data", "only_ids"), None)
    only_ids = args.only_ids or config_only_ids or ""
    eval_kwargs = dict(
        dataset=pick("dataset", "sample", ("data", "dataset")),
        n=pick("n", 2, ("data", "n")),
        max_new=pick("max_new", 4096, ("generation", "max_new")),
        recent=pick("recent", 64, ("eviction", "recent")),
        sink=pick("sink", 8, ("eviction", "sink")),
        evict_every=pick("evict_every", 128, ("eviction", "evict_every")),
        obs_window=pick("obs_window", 16, ("eviction", "obs_window")),
        obs_decay=pick("obs_decay", 0.9, ("eviction", "obs_decay")),
        do_sample=not args.greedy and bool(
            _config_value(cfg, "do_sample", ("generation", "do_sample"), True)
        ),
        temperature=pick("temperature", 0.6, ("generation", "temperature")),
        top_p=pick("top_p", 0.95, ("generation", "top_p")),
        seed=pick("seed", 0, ("generation", "seed")),
        only_ids=set(only_ids.split(",")) if only_ids else None,
        debug_dir=args.debug_dir or _config_value(cfg, "debug_dir", ("debug", "dir"), None),
        debug_topk=pick("debug_topk", 0, ("debug", "topk")),
        policy_params=policy_defaults,
    )

    groups: dict[str, list] = {}
    for arm in arms:
        groups.setdefault(_arm_attn_backend(arm.backend, requested_attn, fast_attn), []).append(arm)

    payloads = []
    for attn_backend, group_arms in groups.items():
        print(
            f"\n=== Loading model with attn={attn_backend} for arms: "
            f"{', '.join(arm.name for arm in group_arms)} ===",
            flush=True,
        )
        model, tokenizer = load_causal_lm(
            model_name,
            dtype=pick("dtype", "bfloat16", ("model", "dtype")),
            device_map=pick("device_map", "cuda", ("model", "device_map")),
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

    payload = payloads[0] if len(payloads) == 1 else _merge_payloads(payloads, out_path)
    print("\n=== KV eviction eval ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
