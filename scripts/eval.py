"""Run a generation-time KV eviction policy benchmark."""
from __future__ import annotations

import argparse
import json

import yaml

from kvbench.evaluator import run_generation_eval
from kvbench.models import load_causal_lm
from kvbench.policies import parse_arms


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_config(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "policy_params":
            params = {name: dict(values or {}) for name, values in merged.get(key, {}).items()}
            for policy, values in (value or {}).items():
                params.setdefault(policy, {}).update(values or {})
            merged[key] = params
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


def _merge_policy_params(config_params: dict | None, overrides: list[str]) -> dict:
    params = {name: dict(values or {}) for name, values in (config_params or {}).items()}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--strategy-config",
        default=None,
        help="Optional strategy config, e.g. configs/strategies/all_supported.yaml",
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
    parser.add_argument("--attn", default=None)
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

    def pick(name, default=None):
        value = getattr(args, name.replace("-", "_"), None)
        return cfg.get(name, default) if value is None else value

    model_name = args.model or cfg.get("model")
    if not model_name:
        raise SystemExit("missing --model or model in config")

    arms = parse_arms(args.arms or cfg.get("arms", "fullkv,snapkv@1024,h2o@1024,streamingllm@1024,rkv@1024"))
    policy_params = _merge_policy_params(cfg.get("policy_params"), args.policy_param)
    model, tokenizer = load_causal_lm(
        model_name,
        dtype=pick("dtype", "bfloat16"),
        device_map=pick("device_map", "cuda"),
        attn_implementation=pick("attn", "eager"),
    )
    payload = run_generation_eval(
        model,
        tokenizer,
        dataset=pick("dataset", "sample"),
        n=pick("n", 2),
        arms=arms,
        out_path=pick("out", "results/eval.json"),
        max_new=pick("max_new", 4096),
        recent=pick("recent", 64),
        sink=pick("sink", 8),
        evict_every=pick("evict_every", 128),
        obs_window=pick("obs_window", 16),
        obs_decay=pick("obs_decay", 0.9),
        do_sample=not args.greedy and bool(cfg.get("do_sample", True)),
        temperature=pick("temperature", 0.6),
        top_p=pick("top_p", 0.95),
        seed=pick("seed", 0),
        only_ids=set(args.only_ids.split(",")) if args.only_ids else None,
        debug_dir=args.debug_dir,
        debug_topk=pick("debug_topk", 0),
        policy_params=policy_params,
    )
    print("\n=== KV eviction eval ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
