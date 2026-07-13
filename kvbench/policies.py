"""Policy arm definitions for generation-time KV eviction benchmarks."""
from __future__ import annotations

from dataclasses import dataclass

from kv_eviction.strategies.token import policy_names


@dataclass(frozen=True)
class EvalArm:
    id: str
    policy: str
    budget: int
    params: dict
    anchor_mode: str = "none"

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

    if policy == "fullkv":
        budget = int(item.get("budget", 10**9))
        arm_id = str(item.get("id") or item.get("name") or "fullkv")
    else:
        if "budget" not in item:
            raise ValueError(f"structured arm {policy!r} is missing budget")
        budget = int(item["budget"])
        arm_id = str(item.get("id") or item.get("name") or f"{policy}@{budget}")

    params = dict((policy_defaults or {}).get(policy, {}) or {})
    params.update(item.get("params") or {})
    params.update((policy_overrides or {}).get(policy, {}) or {})
    return EvalArm(
        id=arm_id,
        policy=policy,
        budget=budget,
        params=params,
        anchor_mode=str(item.get("anchor_mode", "none")),
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
