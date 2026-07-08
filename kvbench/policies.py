"""Policy arm definitions for generation-time KV eviction benchmarks."""
from __future__ import annotations

from dataclasses import dataclass

from rkv.policies import policy_names


@dataclass(frozen=True)
class EvalArm:
    name: str
    backend: str
    budget: int
    anchor_mode: str = "none"

    @property
    def is_full(self) -> bool:
        return self.budget >= 10**8


def parse_arm(spec: str) -> EvalArm:
    """Parse arm specs such as full, snapkv@1024, rkv@512."""
    spec = spec.strip()
    if not spec:
        raise ValueError("empty arm spec")
    if spec == "full":
        return EvalArm(name="full", backend="snapkv", budget=10**9)
    if "@" not in spec:
        raise ValueError(f"arm {spec!r} must be full or backend@budget")
    backend, budget_text = spec.split("@", 1)
    if backend not in policy_names():
        valid = ", ".join(["full", *policy_names()])
        raise ValueError(f"unknown arm backend {backend!r}; expected one of: {valid}")
    budget = int(budget_text)
    return EvalArm(name=f"{backend}@{budget}", backend=backend, budget=budget)


def parse_arms(specs: str) -> list[EvalArm]:
    return [parse_arm(part) for part in specs.split(",") if part.strip()]
