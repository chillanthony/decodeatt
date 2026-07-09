"""Token-level KV eviction strategy registry."""
from __future__ import annotations

from .base import HeadwiseSelection, SelectionContext, TokenEvictionPolicy
from .defensivekv import CriticalKVPolicy, DefensiveKVPolicy
from .full import FullKVPolicy
from .h2o import H2OPolicy
from .random import RandomPolicy
from .rkv import RKVPolicy
from .snapkv import SnapKVPolicy
from .streamingllm import StreamingLLMPolicy
from .window import WindowPolicy


_POLICIES: dict[str, TokenEvictionPolicy] = {
    policy.name: policy
    for policy in (
        FullKVPolicy(),
        SnapKVPolicy(),
        CriticalKVPolicy(),
        DefensiveKVPolicy(),
        RKVPolicy(),
        H2OPolicy(),
        StreamingLLMPolicy(),
        WindowPolicy(),
        RandomPolicy(),
    )
}


def get_policy(name: str) -> TokenEvictionPolicy:
    try:
        return _POLICIES[name]
    except KeyError as exc:
        valid = ", ".join(policy_names())
        raise ValueError(f"unknown eviction policy {name!r}; expected one of: {valid}") from exc


def policy_names() -> list[str]:
    return sorted(_POLICIES)
