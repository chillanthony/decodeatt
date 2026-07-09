"""Small checks for official baseline strategy registry semantics."""
from __future__ import annotations

import torch

from kvbench.policies import parse_arm
from kv_eviction.strategies.base import SelectionContext
from kv_eviction.strategies.token import get_policy, policy_names


def _ctx(n: int, budget: int, recent: int = 4, sink: int = 2, params: dict | None = None):
    values = torch.arange(n, dtype=torch.float32)
    return SelectionContext(
        importance=values,
        cumulative_attention=values,
        window_attention=values,
        key_reps=None,
        slot_pos=torch.arange(n),
        budget=budget,
        recent=recent,
        sink=sink,
        params=params or {},
    )


def test_registry_contains_official_baselines():
    names = set(policy_names())
    for name in ["fullkv", "snapkv", "h2o", "streamingllm", "rkv"]:
        assert name in names
    assert get_policy("full") is get_policy("fullkv")


def test_fullkv_keeps_everything_and_full_alias_parses():
    policy = get_policy("fullkv")
    idx = policy.select_keep(_ctx(n=12, budget=4))
    torch.testing.assert_close(idx, torch.arange(12))
    arm = parse_arm("full")
    assert arm.backend == "fullkv"
    assert arm.is_full


def test_streamingllm_keeps_sink_and_recent_budget():
    policy = get_policy("streamingllm")
    idx = policy.select_keep(_ctx(n=12, budget=6, sink=2, recent=4))
    torch.testing.assert_close(idx, torch.tensor([0, 1, 8, 9, 10, 11]))


def test_streamingllm_uses_budget_when_recent_is_smaller():
    policy = get_policy("streamingllm")
    idx = policy.select_keep(_ctx(n=12, budget=8, sink=2, recent=4))
    torch.testing.assert_close(idx, torch.tensor([0, 1, 6, 7, 8, 9, 10, 11]))


if __name__ == "__main__":
    test_registry_contains_official_baselines()
    test_fullkv_keeps_everything_and_full_alias_parses()
    test_streamingllm_keeps_sink_and_recent_budget()
    test_streamingllm_uses_budget_when_recent_is_smaller()
    print("Strategy baseline tests passed")
