"""Policy arm definitions for generation-time KV eviction benchmarks."""
from __future__ import annotations

from dataclasses import dataclass

from kveviction.strategies.anchors import ANCHOR_MODES
from kveviction.strategies.token import get_policy, policy_names


@dataclass(frozen=True)
class EvalArm:
    id: str
    policy: str
    budget: int
    params: dict
    anchor_mode: str = "none"
    engine: str = "hf"
    # Anchor protection (kveviction/strategies/anchors.py): rescue
    # int(anchor_frac * budget) evicted slots per compaction.  ``sig`` is the
    # (weights[4], bias) signature used by anchor_mode="sig".
    anchor_frac: float = 0.0
    sig: tuple | None = None

    @property
    def name(self) -> str:
        return self.id

    @property
    def backend(self) -> str:
        return self.policy

    @property
    def is_full(self) -> bool:
        return self.budget >= 10**8


def _merge_params(policy: str, defaults: dict | None, overrides: dict | None = None) -> dict:
    params = dict((defaults or {}).get(policy, {}) or {})
    params.update((overrides or {}).get(policy, {}) or {})
    return params


def _parse_engine(value) -> str:
    engine = str(value or "hf").strip().lower()
    if engine not in {"hf", "vllm"}:
        raise ValueError(f"unknown execution engine {engine!r}; expected 'hf' or 'vllm'")
    return engine


def _parse_anchor(policy: str, item: dict) -> tuple[str, float, tuple | None]:
    """Validate the anchor settings of a structured arm."""
    mode = str(item.get("anchor_mode", "none"))
    if mode not in ANCHOR_MODES:
        raise ValueError(f"anchor_mode {mode!r} must be one of {ANCHOR_MODES}")
    frac = float(item.get("anchor_frac", 0.0))
    if frac < 0:
        raise ValueError("anchor_frac cannot be negative")
    raw_sig = item.get("sig")
    if mode == "none":
        if frac > 0 or raw_sig is not None:
            raise ValueError("anchor_frac/sig are set but anchor_mode is 'none'")
        return mode, 0.0, None
    if frac <= 0:
        raise ValueError(f"anchor_mode {mode!r} requires anchor_frac > 0")
    base = get_policy(policy)
    if base.never_evict or base.cache_aware:
        raise ValueError(
            f"anchor modes need a slot-level evicting policy (window, random, "
            f"streamingllm); {policy!r} does not support them"
        )
    sig = None
    if mode == "sig":
        if not isinstance(raw_sig, dict) or set(raw_sig) != {"weights", "bias"}:
            raise ValueError("anchor_mode 'sig' requires sig: {weights: [4 floats], bias: float}")
        weights = [float(w) for w in raw_sig["weights"]]
        if len(weights) != 4:
            raise ValueError("sig.weights must have 4 entries (entropy, cum_attn, position, concentration)")
        if not base.needs_attention:
            raise ValueError(
                f"sig anchors use attention statistics, which {policy!r} never observes; use window"
            )
        sig = (tuple(weights), float(raw_sig["bias"]))
    elif raw_sig is not None:
        raise ValueError("sig is only used with anchor_mode 'sig'")
    return mode, frac, sig


def parse_arm(
    spec: str,
    policy_defaults: dict | None = None,
    policy_overrides: dict | None = None,
) -> EvalArm:
    """Parse arm specs such as fullkv, snapkv@1024, rkv@512."""
    spec = spec.strip()
    if not spec:
        raise ValueError("empty arm spec")
    if spec == "fullkv":
        return EvalArm(
            id="fullkv",
            policy="fullkv",
            budget=10**9,
            params=_merge_params("fullkv", policy_defaults, policy_overrides),
            engine="hf",
        )
    if "@" not in spec:
        raise ValueError(f"arm {spec!r} must be fullkv or backend@budget")
    backend, budget_text = spec.split("@", 1)
    if backend not in policy_names():
        valid = ", ".join(policy_names())
        raise ValueError(f"unknown arm backend {backend!r}; expected one of: {valid}")
    budget = int(budget_text)
    return EvalArm(
        id=f"{backend}@{budget}",
        policy=backend,
        budget=budget,
        params=_merge_params(backend, policy_defaults, policy_overrides),
        engine="hf",
    )


def parse_structured_arm(
    item: dict,
    policy_defaults: dict | None = None,
    policy_overrides: dict | None = None,
) -> EvalArm:
    if not isinstance(item, dict):
        raise TypeError(f"structured arm must be a mapping, got {type(item).__name__}")
    policy = str(item.get("policy") or item.get("backend") or "").strip()
    if not policy:
        raise ValueError(f"structured arm is missing policy/backend: {item!r}")
    if policy not in policy_names():
        valid = ", ".join(policy_names())
        raise ValueError(f"unknown arm policy {policy!r}; expected one of: {valid}")

    anchor_mode, anchor_frac, sig = _parse_anchor(policy, item)
    if policy == "fullkv":
        budget = int(item.get("budget", 10**9))
        default_id = "fullkv"
    else:
        if "budget" not in item:
            raise ValueError(f"structured arm {policy!r} is missing budget")
        budget = int(item["budget"])
        default_id = f"{policy}@{budget}"
    if anchor_mode != "none":
        default_id += f"+{anchor_mode}{anchor_frac:g}"
    arm_id = str(item.get("id") or item.get("name") or default_id)

    params = dict((policy_defaults or {}).get(policy, {}) or {})
    params.update(item.get("params") or {})
    params.update((policy_overrides or {}).get(policy, {}) or {})
    return EvalArm(
        id=arm_id,
        policy=policy,
        budget=budget,
        params=params,
        anchor_mode=anchor_mode,
        engine=_parse_engine(item.get("engine", item.get("runtime", "hf"))),
        anchor_frac=anchor_frac,
        sig=sig,
    )


def parse_arms(
    specs,
    policy_defaults: dict | None = None,
    policy_overrides: dict | None = None,
) -> list[EvalArm]:
    if isinstance(specs, str):
        return [
            parse_arm(part, policy_defaults=policy_defaults, policy_overrides=policy_overrides)
            for part in specs.split(",")
            if part.strip()
        ]
    if isinstance(specs, list):
        return [
            parse_structured_arm(
                item,
                policy_defaults=policy_defaults,
                policy_overrides=policy_overrides,
            )
            for item in specs
        ]
    raise TypeError(f"arms must be a comma-separated string or list, got {type(specs).__name__}")
