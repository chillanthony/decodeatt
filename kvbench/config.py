"""Run configuration: split-YAML loading, schema validation and CLI resolution.

A run combines three YAML files (model, data, strategy).  Every section and
key is checked against ``SCHEMA`` so a misspelled field fails before any model
or dataset is loaded instead of being silently ignored.  CLI flags take
precedence over YAML, which takes precedence over the defaults below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from kvbench.arms import EvalArm, _parse_anchor, parse_arms
from kvbench.metrics import ANSWER_EXTRACTIONS
from kvbench.models import DEFAULT_PROMPT_TEMPLATE, format_prompt
from kveviction.strategies.anchors import ANCHOR_MODES
from kveviction.strategies.token import get_policy, policy_names

_NUM = (int, float)

# section -> key -> accepted types
SCHEMA: dict[str, dict[str, tuple]] = {
    "model": {
        "name": (str,), "dtype": (str,), "device_map": (str, dict),
        "attn": (str,), "fast_attn": (str,), "fast_observe": (bool,),
    },
    "data": {
        "dataset": (str,), "n": (int,), "only_ids": (str, list),
        "prompt_template": (str,), "answer_extraction": (str,),
    },
    "generation": {
        "max_new": (int,), "do_sample": (bool,), "temperature": _NUM, "top_p": _NUM,
        "seed": (int,), "batch_size": (int,), "num_return_sequences": (int,),
        "seed_offset": (int,), "problem_batch_size": (int,), "prompt_bucket_size": (int,),
    },
    "eviction": {
        "recent": (int,), "sink": (int,), "evict_every": (int,),
        "obs_window": (int,), "obs_decay": _NUM,
    },
    "experiment": {"out": (str,), "log_mode": (str,), "sync_timing": (bool,)},
}
ARM_KEYS = {
    "id": (str,), "name": (str,), "policy": (str,), "backend": (str,), "budget": (int,),
    "params": (dict,), "anchor_mode": (str,), "anchor_frac": _NUM, "sig": (dict,),
    "engine": (str,), "runtime": (str,),
}
# Which top-level sections each split config file may define.
FILE_SECTIONS = {
    # ``vllm`` holds engine kwargs forwarded verbatim to ``vllm.LLM`` (engine=vllm arms).
    "model": {"model", "vllm"},
    "data": {"data", "generation"},
    "strategy": {"arms", "policy_defaults", "policy_params", "eviction", "experiment"},
}
DEFAULTS = {
    ("data", "dataset"): "sample", ("data", "n"): 2,
    ("data", "prompt_template"): DEFAULT_PROMPT_TEMPLATE, ("data", "answer_extraction"): "boxed",
    ("generation", "max_new"): 4096, ("generation", "temperature"): 0.6,
    ("generation", "top_p"): 0.95, ("generation", "seed"): 0,
    ("generation", "batch_size"): 1, ("generation", "num_return_sequences"): 1,
    ("generation", "seed_offset"): 0, ("generation", "problem_batch_size"): 1,
    ("generation", "prompt_bucket_size"): 0,
    ("eviction", "recent"): 64, ("eviction", "sink"): 8, ("eviction", "evict_every"): 128,
    ("eviction", "obs_window"): 16, ("eviction", "obs_decay"): 0.9,
    ("model", "dtype"): "bfloat16", ("model", "device_map"): "cuda",
    ("model", "attn"): "auto", ("model", "fast_attn"): "sdpa", ("model", "fast_observe"): False,
    ("experiment", "out"): "results/eval.json", ("experiment", "log_mode"): "full",
    ("experiment", "sync_timing"): False,
}
DEFAULT_ARMS = "fullkv,snapkv@1024,h2o@1024,streamingllm@1024,rkv@1024"


class ConfigError(ValueError):
    """Raised for malformed run configuration."""


# --------------------------------------------------------------------------
# Loading and merging
# --------------------------------------------------------------------------

def _merge_dicts(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_config(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in {"policy_params", "policy_defaults"}:
            existing = merged.get("policy_defaults", merged.get("policy_params", {}))
            params = {name: dict(values or {}) for name, values in existing.items()}
            for policy, values in (value or {}).items():
                params.setdefault(policy, {}).update(values or {})
            merged["policy_defaults" if key == "policy_defaults" else "policy_params"] = params
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    with Path(path).resolve().open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_run_config(model_config: str | None, data_config: str | None,
                    strategy_config: str | None) -> dict:
    """Load, validate and merge the three split config files."""
    if not model_config or not data_config or not strategy_config:
        raise ConfigError("a run requires --model-config, --data-config, and --strategy-config")
    files = {"model": model_config, "data": data_config, "strategy": strategy_config}
    loaded = {}
    for kind, path in files.items():
        cfg = load_yaml(path)
        extra = set(cfg) - FILE_SECTIONS[kind]
        if extra:
            allowed = ", ".join(sorted(FILE_SECTIONS[kind]))
            raise ConfigError(f"{path}: unknown section(s) {sorted(extra)}; {kind} config may define: {allowed}")
        try:
            validate_config(cfg)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from None
        loaded[kind] = cfg
    cfg = merge_config({}, loaded["data"])
    cfg = merge_config(cfg, loaded["strategy"])
    return merge_config(cfg, loaded["model"])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _check_type(where: str, value, types: tuple) -> None:
    if value is None:
        return
    # bool is an int subclass; only accept it where bool is declared.
    if isinstance(value, bool) and bool not in types:
        raise ConfigError(f"{where} must be {'/'.join(t.__name__ for t in types)}, got bool")
    if not isinstance(value, types):
        raise ConfigError(
            f"{where} must be {'/'.join(t.__name__ for t in types)}, got {type(value).__name__}"
        )


def _check_keys(where: str, mapping: dict, schema: dict) -> None:
    unknown = set(mapping) - set(schema)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in {where}; expected: {', '.join(sorted(schema))}"
        )
    for key, value in mapping.items():
        _check_type(f"{where}.{key}", value, schema[key])


def validate_config(cfg: dict) -> None:
    """Check sections, keys, value types, policy names and anchor modes."""
    for section, schema in SCHEMA.items():
        value = cfg.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"config section {section!r} must be a mapping")
        _check_keys(section, value, schema)

    vllm = cfg.get("vllm")
    if vllm is not None and not isinstance(vllm, dict):
        raise ConfigError("config section 'vllm' must be a mapping of vllm.LLM keyword arguments")

    for section in ("policy_defaults", "policy_params"):
        value = cfg.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"config section {section!r} must be a mapping")
        for policy, params in value.items():
            if policy not in policy_names():
                raise ConfigError(
                    f"unknown policy {policy!r} in {section}; expected one of: {', '.join(policy_names())}"
                )
            _check_type(f"{section}.{policy}", params, (dict,))

    arms = cfg.get("arms")
    if isinstance(arms, list):
        for index, item in enumerate(arms):
            where = f"arms[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{where} must be a mapping")
            _check_keys(where, item, ARM_KEYS)
            mode = item.get("anchor_mode", "none")
            if mode not in ANCHOR_MODES:
                raise ConfigError(f"{where}.anchor_mode {mode!r} must be one of {ANCHOR_MODES}")
            policy = item.get("policy") or item.get("backend")
            if policy in policy_names():
                try:
                    _parse_anchor(policy, item)
                except ValueError as exc:
                    raise ConfigError(f"{where}: {exc}") from None
    elif arms is not None and not isinstance(arms, str):
        raise ConfigError("arms must be a comma-separated string or a list of mappings")


# --------------------------------------------------------------------------
# CLI resolution
# --------------------------------------------------------------------------

def parse_scalar(value: str):
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


def parse_policy_param_overrides(overrides: list[str]) -> dict:
    params: dict[str, dict] = {}
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ConfigError(
                f"invalid --policy-param {item!r}; expected policy.key=value, "
                "e.g. rkv.redundancy_lambda=0.7"
            )
        left, value = item.split("=", 1)
        policy, key = left.split(".", 1)
        params.setdefault(policy, {})[key] = parse_scalar(value)
    return params


def policy_defaults(cfg: dict) -> dict:
    return {
        name: dict(values or {})
        for name, values in (cfg.get("policy_defaults") or cfg.get("policy_params") or {}).items()
    }


def _section_value(cfg: dict, section: str, key: str):
    value = (cfg.get(section) or {}).get(key)
    return DEFAULTS.get((section, key)) if value is None else value


def arm_attn_backend(
    policy: str, requested_attn: str, fast_attn: str, fast_observe: bool = False
) -> str:
    """Pick the attention backend for ``--attn auto``.

    Policies that never read attention use ``fast_attn``.  Cache-aware
    policies (``needs_attn_history``: rkv/snapkv/h2o) recompute their attention
    rows from hidden states, so with ``fast_observe`` they also use
    ``fast_attn``; by default they stay on eager so results match earlier
    eager baselines bit for bit.  Policies that read softmax probabilities
    from the forward pass (window and its anchors) always need eager.
    """
    if requested_attn != "auto":
        return requested_attn
    spec = get_policy(policy)
    if not spec.needs_attention or (fast_observe and spec.needs_attn_history):
        return fast_attn
    return "eager"


@dataclass
class RunSettings:
    """Everything ``script/eval.py`` needs after YAML + CLI resolution."""

    cfg: dict
    model_name: str
    arms: list[EvalArm]
    dtype: str
    device_map: object
    requested_attn: str
    fast_attn: str
    fast_observe: bool
    out_path: Path
    log_mode: str
    eval_kwargs: dict

    def attn_groups(self) -> tuple[dict[str, list[EvalArm]], list[EvalArm]]:
        """Split HF arms by attention backend; return ``(groups, vllm_arms)``."""
        groups: dict[str, list[EvalArm]] = {}
        vllm_arms = []
        for arm in self.arms:
            if arm.engine == "vllm":
                vllm_arms.append(arm)
            else:
                backend = arm_attn_backend(
                    arm.backend, self.requested_attn, self.fast_attn, self.fast_observe
                )
                groups.setdefault(backend, []).append(arm)
        return groups, vllm_arms

    def resolved(self) -> dict:
        """Canonical effective config, as printed by ``--print-config``."""
        cfg, kw = self.cfg, self.eval_kwargs
        resolved = {
            "experiment": {
                **(cfg.get("experiment") or {}), "out": str(self.out_path), "log_mode": self.log_mode,
                "sync_timing": kw["sync_timing"],
            },
            "model": {**(cfg.get("model") or {}), "name": self.model_name},
            "data": {
                **(cfg.get("data") or {}), "dataset": kw["dataset"], "n": kw["n"],
                "prompt_template": kw["prompt_template"],
                "answer_extraction": kw["answer_extraction"],
            },
            "generation": {**(cfg.get("generation") or {})},
            "eviction": {**(cfg.get("eviction") or {})},
            "policy_defaults": policy_defaults(cfg),
            "arms": [],
            "debug": {},
        }
        for key in SCHEMA["generation"]:
            resolved["generation"][key] = kw[key]
        for key in SCHEMA["eviction"]:
            resolved["eviction"][key] = kw[key]
        resolved["data"]["only_ids"] = sorted(kw["only_ids"]) if kw["only_ids"] else None
        resolved["debug"] = {"dir": kw["debug_dir"], "topk": kw["debug_topk"]}
        resolved["model"].update(
            dtype=self.dtype, device_map=self.device_map,
            attn=self.requested_attn, fast_attn=self.fast_attn, fast_observe=self.fast_observe,
        )
        if cfg.get("vllm"):
            resolved["vllm"] = dict(cfg["vllm"])
        for arm in self.arms:
            resolved["arms"].append({
                "id": arm.id, "policy": arm.policy, "budget": arm.budget,
                "params": arm.params, "anchor_mode": arm.anchor_mode,
                "anchor_frac": arm.anchor_frac, "engine": arm.engine,
            })
            if arm.sig is not None:
                resolved["arms"][-1]["sig"] = {"weights": list(arm.sig[0]), "bias": arm.sig[1]}
        return resolved


def resolve_settings(cfg: dict, args) -> RunSettings:
    """Apply CLI precedence over the merged YAML config."""

    def pick(name: str, section: str, key: str | None = None):
        value = getattr(args, name, None)
        return value if value is not None else _section_value(cfg, section, key or name)

    model_name = args.model or (cfg.get("model") or {}).get("name")
    if not model_name:
        raise ConfigError("missing --model or model.name in config")

    overrides = parse_policy_param_overrides(args.policy_param)
    for policy in overrides:
        if policy not in policy_names():
            raise ConfigError(
                f"unknown policy {policy!r} in --policy-param; expected one of: {', '.join(policy_names())}"
            )
    defaults = policy_defaults(cfg)
    raw_arms = args.arms or cfg.get("arms", DEFAULT_ARMS)
    if not raw_arms:
        raise ConfigError("config must define at least one arm (or pass --arms)")
    arms = parse_arms(raw_arms, policy_defaults=defaults, policy_overrides=overrides)
    if len(arms) != 1:
        raise ConfigError("a run with split configs must contain exactly one strategy arm")

    log_mode = pick("log_mode", "experiment")
    if log_mode not in {"full", "brief"}:
        raise ConfigError(f"invalid log_mode {log_mode!r}; expected 'full' or 'brief'")
    only_ids = args.only_ids or (cfg.get("data") or {}).get("only_ids") or ""
    if isinstance(only_ids, list):
        only_ids = ",".join(map(str, only_ids))

    answer_extraction = pick("answer_extraction", "data")
    if answer_extraction not in ANSWER_EXTRACTIONS:
        raise ConfigError(
            f"invalid answer_extraction {answer_extraction!r}; expected one of {ANSWER_EXTRACTIONS}"
        )
    prompt_template = pick("prompt_template", "data")
    try:
        format_prompt("", prompt_template)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None

    eval_kwargs = dict(
        dataset=pick("dataset", "data"),
        n=pick("n", "data"),
        prompt_template=prompt_template,
        answer_extraction=answer_extraction,
        **{key: pick(key, "eviction") for key in SCHEMA["eviction"]},
        **{
            key: pick(key, "generation")
            for key in SCHEMA["generation"]
            if key != "do_sample"
        },
        only_ids=set(only_ids.split(",")) if only_ids else None,
        debug_dir=args.debug_dir,
        debug_topk=args.debug_topk if args.debug_topk is not None else 0,
        policy_params=defaults,
        log_mode=log_mode,
        sync_timing=bool(pick("sync_timing", "experiment")),
    )
    do_sample = (cfg.get("generation") or {}).get("do_sample")
    eval_kwargs["do_sample"] = not args.greedy and bool(True if do_sample is None else do_sample)

    return RunSettings(
        cfg=cfg,
        model_name=model_name,
        arms=arms,
        dtype=pick("dtype", "model"),
        device_map=pick("device_map", "model"),
        requested_attn=pick("attn", "model"),
        fast_attn=pick("fast_attn", "model"),
        fast_observe=bool(pick("fast_observe", "model")),
        out_path=Path(pick("out", "experiment")),
        log_mode=log_mode,
        eval_kwargs=eval_kwargs,
    )


__all__ = [
    "SCHEMA", "ARM_KEYS", "ConfigError", "RunSettings", "arm_attn_backend",
    "load_run_config", "merge_config", "parse_policy_param_overrides",
    "parse_scalar", "policy_defaults", "resolve_settings", "validate_config",
]
