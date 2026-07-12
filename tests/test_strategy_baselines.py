"""Small checks for official baseline strategy registry semantics."""
from __future__ import annotations

import torch

from kvbench.policies import parse_arm
from kv_eviction.runner_token import _should_collect_observation, _token_samples
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
    assert "full" not in names


def test_fullkv_keeps_everything_and_parses():
    policy = get_policy("fullkv")
    idx = policy.select_keep(_ctx(n=12, budget=4))
    torch.testing.assert_close(idx, torch.arange(12))
    arm = parse_arm("fullkv")
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


def test_strategy_observation_windows_are_minimal():
    assert get_policy("h2o").observation_window({}, 16) == 1
    assert get_policy("rkv").observation_window({"alpha": 6}, 16) == 6
    assert get_policy("snapkv").observation_window({"window_size": 12}, 16) == 12
    assert get_policy("fullkv").observation_window({}, 16) == 0
    assert get_policy("streamingllm").observation_window({}, 16) == 0
    assert get_policy("random").observation_window({}, 16) == 0


def test_observation_waits_until_next_evict_can_trigger():
    policy = get_policy("rkv")
    should_observe, active_window = _should_collect_observation(
        policy,
        track=True,
        policy_params={"alpha": 2},
        obs_window=16,
        evict_every=4,
        step=2,
        cache_len=7,
        budget=10,
        anchor_k=0,
    )
    assert active_window == 2
    assert not should_observe

    should_observe, active_window = _should_collect_observation(
        policy,
        track=True,
        policy_params={"alpha": 2},
        obs_window=16,
        evict_every=4,
        step=2,
        cache_len=8,
        budget=10,
        anchor_k=0,
    )
    assert active_window == 2
    assert should_observe


def test_debug_token_samples_can_be_disabled():
    class _Tokenizer:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("decode should not be called when limit is 0")

    samples = _token_samples(
        _Tokenizer(),
        torch.tensor([11, 12, 13]),
        torch.tensor([0, 1, 2]),
        limit=0,
    )
    assert samples == []


if __name__ == "__main__":
    test_registry_contains_official_baselines()
    test_fullkv_keeps_everything_and_parses()
    test_streamingllm_keeps_sink_and_recent_budget()
    test_streamingllm_uses_budget_when_recent_is_smaller()
    test_strategy_observation_windows_are_minimal()
    test_observation_waits_until_next_evict_can_trigger()
    test_debug_token_samples_can_be_disabled()
    print("Strategy baseline tests passed")
