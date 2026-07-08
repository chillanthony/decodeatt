"""Token-level KV eviction strategy registry."""
from __future__ import annotations

from .base import SelectionContext, TokenEvictionPolicy
from .h2o import H2OPolicy
from .random import RandomPolicy
from .rkv import RKVPaperAliasPolicy, RKVPolicy
from .snapkv import SnapKVPolicy
from .window import WindowPolicy


_POLICIES: dict[str, TokenEvictionPolicy] = {
    policy.name: policy
    for policy in (
        SnapKVPolicy(),
        RKVPolicy(),
        RKVPaperAliasPolicy(),
        H2OPolicy(),
        WindowPolicy(),
        RandomPolicy(),
    )
}


def get_policy(name: str) -> TokenEvictionPolicy:
    try:
        return _POLICIES[name]
    except KeyError as exc:
        valid = ", ".join(["full", *_POLICIES])
        raise ValueError(f"unknown eviction policy {name!r}; expected one of: {valid}") from exc


def policy_names() -> list[str]:
    return sorted(_POLICIES)
