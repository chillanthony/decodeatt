"""Small checks for official baseline strategy registry semantics."""
from __future__ import annotations

import torch

from kvbench.policies import parse_arm, parse_arms
from kv_eviction.runner_token import _cache_length_summary, _compact_cache_update, _should_collect_observation, _token_samples
from kv_eviction.strategies.base import SelectionContext
from kv_eviction.strategies.token import get_policy, policy_names
from scripts.eval import _allocate_group_ranks, _arm_attn_backend, _claim_next_problem


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


def test_legacy_arm_parsing_attaches_policy_defaults_and_overrides():
    arms = parse_arms(
        "rkv@1024,rkv@2048",
        policy_defaults={"rkv": {"lambda": 0.1, "alpha": 8}},
        policy_overrides={"rkv": {"lambda": 0.2}},
    )
    assert [arm.name for arm in arms] == ["rkv@1024", "rkv@2048"]
    assert [arm.budget for arm in arms] == [1024, 2048]
    assert arms[0].params == {"lambda": 0.2, "alpha": 8}
    assert arms[1].params == {"lambda": 0.2, "alpha": 8}


def test_structured_arm_parsing_allows_same_policy_with_different_params():
    arms = parse_arms(
        [
            {"id": "rkv_b1024_lam01", "policy": "rkv", "budget": 1024},
            {
                "id": "rkv_b1024_lam02",
                "policy": "rkv",
                "budget": 1024,
                "params": {"lambda": 0.2},
            },
        ],
        policy_defaults={"rkv": {"lambda": 0.1, "alpha": 8}},
    )
    assert [arm.name for arm in arms] == ["rkv_b1024_lam01", "rkv_b1024_lam02"]
    assert [arm.backend for arm in arms] == ["rkv", "rkv"]
    assert arms[0].params == {"lambda": 0.1, "alpha": 8}
    assert arms[1].params == {"lambda": 0.2, "alpha": 8}


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


def test_legacy_tuple_cache_can_be_summarized_and_compacted():
    keys = torch.arange(10, dtype=torch.float32).view(1, 2, 5, 1)
    values = keys + 100
    cache = ((keys, values),)
    summary = _cache_length_summary(cache, 5)
    assert summary["mean_effective_cache_len"] == 5.0
    assert summary["effective_kv_tokens_per_layer_head"] == [[5, 5]]

    new_cache, new_len = _compact_cache_update(cache, torch.tensor([3, 1, 4]))
    assert new_len == 3
    torch.testing.assert_close(new_cache[0][0][0, :, :, 0], torch.tensor([[3.0, 1.0, 4.0], [8.0, 6.0, 9.0]]))
    torch.testing.assert_close(new_cache[0][1][0, :, :, 0], torch.tensor([[103.0, 101.0, 104.0], [108.0, 106.0, 109.0]]))


def test_auto_attention_backend_selection():
    assert _arm_attn_backend("fullkv", "auto", "sdpa") == "sdpa"
    assert _arm_attn_backend("streamingllm", "auto", "sdpa") == "sdpa"
    assert _arm_attn_backend("window", "auto", "sdpa") == "sdpa"
    assert _arm_attn_backend("random", "auto", "sdpa") == "sdpa"
    assert _arm_attn_backend("snapkv", "auto", "sdpa") == "eager"
    assert _arm_attn_backend("h2o", "auto", "sdpa") == "eager"
    assert _arm_attn_backend("rkv", "auto", "sdpa") == "eager"
    assert _arm_attn_backend("fullkv", "eager", "sdpa") == "eager"


def test_distributed_rank_allocation_balances_attention_groups():
    groups = {
        "sdpa": ["fullkv", "streamingllm"],
        "eager": ["snapkv", "h2o", "rkv"],
    }
    assert _allocate_group_ranks(groups, 8) == [
        ("sdpa", 0, 3),
        ("sdpa", 1, 3),
        ("sdpa", 2, 3),
        ("eager", 0, 5),
        ("eager", 1, 5),
        ("eager", 2, 5),
        ("eager", 3, 5),
        ("eager", 4, 5),
    ]


def test_distributed_problem_claims_are_atomic_counter_indices():
    class _Store:
        value = 0

        def add(self, _key, increment):
            self.value += increment
            return self.value

    store = _Store()
    assert [_claim_next_problem(store, "queue") for _ in range(4)] == [0, 1, 2, 3]


if __name__ == "__main__":
    test_registry_contains_official_baselines()
    test_fullkv_keeps_everything_and_parses()
    test_legacy_arm_parsing_attaches_policy_defaults_and_overrides()
    test_structured_arm_parsing_allows_same_policy_with_different_params()
    test_streamingllm_keeps_sink_and_recent_budget()
    test_streamingllm_uses_budget_when_recent_is_smaller()
    test_strategy_observation_windows_are_minimal()
    test_observation_waits_until_next_evict_can_trigger()
    test_debug_token_samples_can_be_disabled()
    test_legacy_tuple_cache_can_be_summarized_and_compacted()
    test_auto_attention_backend_selection()
    test_distributed_rank_allocation_balances_attention_groups()
    test_distributed_problem_claims_are_atomic_counter_indices()
    print("Strategy baseline tests passed")
