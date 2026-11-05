"""Selection contracts for every eviction policy, scheduling and the anchor wrapper (CPU).

Bit-exact agreement with the official implementations lives in test_parity_*.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from kveviction.selection import select_tokens_batch, should_collect_observation
from kveviction.slot_features import SlotFeatures
from kveviction.strategies.base import SelectionContext
from kveviction.strategies.h2o import select_h2o
from kveviction.strategies.rkv_official import select_rkv_layers
from kveviction.strategies.snapkv import select_snapkv
from kveviction.strategies.token import get_policy, policy_names


ATTENTION_FREE = ["window", "random", "streamingllm"]


CACHE_AWARE = ["snapkv", "h2o", "rkv"]


EVICTING = sorted(set(policy_names()) - {"fullkv"})


@dataclass
class _Layer:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class _Cache:
    layers: list[_Layer]


def _ctx(n: int, budget: int, *, recent: int = 3, sink: int = 2, seed: int = 0) -> SelectionContext:
    gen = torch.Generator().manual_seed(seed)
    score = torch.rand(n, generator=gen)
    return SelectionContext(
        importance=score,
        cumulative_attention=score.clone(),
        window_attention=score.clone(),
        slot_pos=torch.arange(n),
        budget=budget,
        recent=recent,
        sink=sink,
        params={},
    )


def test_every_registered_policy_is_covered_here():
    assert set(EVICTING) == set(ATTENTION_FREE) | set(CACHE_AWARE)
    assert {"fullkv", "snapkv", "h2o", "streamingllm", "rkv"} <= set(policy_names())


@pytest.mark.parametrize("backend", ATTENTION_FREE)
@pytest.mark.parametrize("n", [5, 11, 12, 13, 40])
def test_attention_free_keep_size_and_uniqueness(backend, n):
    budget = 12
    torch.manual_seed(0)
    keep = get_policy(backend).select_keep(_ctx(n, budget))
    assert keep.numel() == min(n, budget)
    assert keep.unique().numel() == keep.numel()
    assert int(keep.min()) >= 0 and int(keep.max()) < n
    if n <= budget:
        torch.testing.assert_close(keep, torch.arange(n))


@pytest.mark.parametrize("backend", ["window", "random"])
def test_scored_policies_always_keep_sink_and_recent(backend):
    n, budget, recent, sink = 40, 12, 3, 2
    torch.manual_seed(0)
    keep = set(get_policy(backend).select_keep(_ctx(n, budget, recent=recent, sink=sink)).tolist())
    assert set(range(sink)) <= keep
    assert set(range(n - recent, n)) <= keep


def test_window_picks_highest_scoring_free_slots():
    n, budget, recent, sink = 20, 8, 2, 1
    ctx = _ctx(n, budget, recent=recent, sink=sink)
    keep = get_policy("window").select_keep(ctx).tolist()
    free = [i for i in range(sink, n - recent)]
    expected = sorted(free, key=lambda i: -float(ctx.window_attention[i]))[: budget - recent - sink]
    assert set(keep) == set(range(sink)) | set(range(n - recent, n)) | set(expected)


_CACHE_AWARE_PARAMS = {
    "snapkv": {"window_size": 4, "kernel_size": 3, "pooling": "maxpool"},
    "h2o": {},
    "rkv": {"alpha": 4, "pool_kernel": 3, "lambda": 0.1, "redundancy_chunk": 8},
}


def _gqa_inputs(n: int, *, layers: int = 2, q_heads: int = 4, kv_heads: int = 2, steps: int = 4):
    gen = torch.Generator().manual_seed(5)
    cache = _Cache([
        _Layer(torch.randn(1, kv_heads, n, 8, generator=gen), torch.randn(1, kv_heads, n, 8, generator=gen))
        for _ in range(layers)
    ])
    # Query-head histories exercise aggregate_gqa_attention (q_heads -> kv_heads).
    history = [
        [torch.randn(q_heads, n, generator=gen) for _ in range(layers)]
        for _ in range(steps)
    ]
    return cache, history


def _trailing_window(backend: str) -> int:
    params = _CACHE_AWARE_PARAMS[backend]
    if backend == "h2o":
        return 1
    return int(params["window_size"] if backend == "snapkv" else params["alpha"])


@pytest.mark.parametrize("backend", CACHE_AWARE)
@pytest.mark.parametrize("n", [10, 12, 30])
def test_cache_aware_headwise_selection_contract(backend, n):
    budget, kv_heads = 12, 2
    cache, history = _gqa_inputs(n, kv_heads=kv_heads)
    idx = get_policy(backend).select_from_cache(
        cache, history, n, budget, dict(_CACHE_AWARE_PARAMS[backend])
    )
    assert isinstance(idx, list) and len(idx) == len(cache.layers)
    kept = min(n, budget)
    window = _trailing_window(backend)
    for layer_idx in idx:
        assert layer_idx.shape == (kv_heads, kept)
        for head in layer_idx:
            assert head.unique().numel() == kept
            assert int(head.min()) >= 0 and int(head.max()) < n
            # The trailing observation window is never evicted.
            assert set(range(n - window, n)) <= set(head.tolist())


def _case(batch_size: int = 3):
    torch.manual_seed(11)
    n = 14
    heads = 2
    dim = 6
    alpha = 4
    keys = torch.randn(batch_size, heads, n, dim)
    values = torch.randn_like(keys)
    cache = _Cache([_Layer(keys.clone(), values.clone())])
    history = [
        [torch.randn(batch_size, heads, n)]
        for _ in range(alpha)
    ]
    params = {
        "alpha": alpha,
        "lambda": 0.1,
        "pool_kernel": 7,
        "similarity_threshold": 0.5,
        "retain_ratio": 0.1,
        "retain_direction": "last",
        "redundancy_chunk": 3,
    }
    return cache, history, n, 9, params


_BATCHED_SELECTORS = {
    "rkv": (select_rkv_layers, None),
    "snapkv": (select_snapkv, {"window_size": 4, "kernel_size": 3, "pooling": "avgpool"}),
    "h2o": (select_h2o, {}),
}


@pytest.mark.parametrize("backend", sorted(_BATCHED_SELECTORS))
def test_batched_cache_aware_selection_matches_independent_requests(backend):
    """Each batch row has its own cache and history; selection must not mix rows."""
    cache, history, n, budget, rkv_params = _case()
    selector, params = _BATCHED_SELECTORS[backend]
    params = rkv_params if params is None else params
    actual = selector(cache, history, n, budget, params)
    assert actual[0].shape == (3, 2, budget)
    for batch_idx in range(3):
        single_cache = _Cache([_Layer(
            cache.layers[0].keys[batch_idx:batch_idx + 1].clone(),
            cache.layers[0].values[batch_idx:batch_idx + 1].clone(),
        )])
        single_history = [[step[0][batch_idx].clone()] for step in history]
        expected = selector(single_cache, single_history, n, budget, params)
        torch.testing.assert_close(actual[0][batch_idx], expected[0])


def test_masked_batched_rkv_matches_unpadded_requests():
    torch.manual_seed(17)
    batch_size, heads, padded_n, short_n, dim = 2, 2, 14, 10, 6
    pad = padded_n - short_n
    keys = torch.randn(batch_size, heads, padded_n, dim)
    values = torch.randn_like(keys)
    valid = torch.ones(batch_size, padded_n, dtype=torch.bool)
    valid[1, :pad] = False
    history = [[torch.randn(batch_size, heads, padded_n)] for _ in range(4)]
    params = {
        "alpha": 4,
        "lambda": 0.1,
        "pool_kernel": 3,
        "similarity_threshold": 0.5,
        "retain_ratio": 0.1,
        "retain_direction": "last",
        "redundancy_chunk": 3,
        "_valid_mask": valid,
    }
    cache = _Cache([_Layer(keys.clone(), values.clone())])
    actual = select_rkv_layers(cache, history, padded_n, 9, params)[0]
    assert valid[0, actual[0]].all()
    assert valid[1, actual[1]].all()

    for batch_idx, start in enumerate([0, pad]):
        logical_n = padded_n - start
        single_cache = _Cache([_Layer(
            keys[batch_idx:batch_idx + 1, :, start:].clone(),
            values[batch_idx:batch_idx + 1, :, start:].clone(),
        )])
        single_history = [[step[0][batch_idx, :, start:].clone()] for step in history]
        single_params = {key: value for key, value in params.items() if key != "_valid_mask"}
        expected = select_rkv_layers(single_cache, single_history, logical_n, 9, single_params)[0]
        torch.testing.assert_close(actual[batch_idx] - start, expected)


def test_batched_attention_free_selection_pads_finished_rows():
    """A finished row (masked filler slots) must not change the batched length.

    Regression: rows with invalid slots were padded to ``budget + anchor_k``
    while fully valid rows kept ``budget`` slots, so stacking failed.
    """
    torch.manual_seed(3)
    batch, n, budget = 2, 12, 6
    feats = SlotFeatures((batch, n), torch.device("cpu"), track_valid=True)
    feats.imp[:] = torch.rand(batch, n)
    feats.win_imp[:] = feats.imp
    feats.slot_pos[:] = torch.arange(n)
    feats.valid[:] = True
    feats.valid[1, -3:] = False
    for backend in ["window", "streamingllm", "random"]:
        idx = select_tokens_batch(
            get_policy(backend), feats, n, budget=budget, recent=2, sink=1,
            policy_params={}, cache=None, attn_history=None,
        )
        assert idx.shape == (batch, budget), backend
        assert feats.valid[0, idx[0]].all(), backend
        assert int(feats.valid[1, idx[1]].sum()) == budget, backend


def test_observation_windows_are_minimal_and_match_needs_attention():
    assert get_policy("h2o").observation_window({}, 16) == 1
    assert get_policy("rkv").observation_window({"alpha": 6}, 16) == 6
    assert get_policy("snapkv").observation_window({"window_size": 12}, 16) == 12
    for name in ("fullkv", "streamingllm", "random"):
        assert get_policy(name).observation_window({}, 16) == 0
    # Policies that never open an observation window must not force eager attention.
    for name in policy_names():
        policy = get_policy(name)
        assert policy.needs_attention == (not policy.never_evict and policy.observation_window({}, 8) > 0)


def test_observation_waits_until_next_evict_can_trigger():
    policy = get_policy("rkv")
    should_observe, active_window = should_collect_observation(
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

    should_observe, active_window = should_collect_observation(
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


def _anchor_ctx(n=30, budget=10):
    ctx = _ctx(n, budget, recent=2, sink=1)
    entropy = torch.arange(n, dtype=torch.float32).flip(0)  # slot 0 has the highest entropy
    return SelectionContext(**{**ctx.__dict__, "entropy": entropy})


@pytest.mark.parametrize("mode", ["anchor", "lowent", "random"])
def test_anchor_wrapper_rescues_k_dropped_slots(mode):
    from kveviction.strategies.anchors import with_anchors

    ctx = _anchor_ctx()
    base_keep = set(get_policy("window").select_keep(ctx).tolist())
    torch.manual_seed(0)
    keep, rescue, _ = with_anchors(get_policy("window"), mode, 3).select_with_rescue(ctx)
    assert rescue.numel() == 3
    assert not set(rescue.tolist()) & base_keep
    assert set(keep.tolist()) == base_keep | set(rescue.tolist())
    assert keep.tolist() == sorted(keep.tolist())
    dropped = sorted(set(range(ctx.slot_pos.numel())) - base_keep)
    if mode == "anchor":
        assert sorted(rescue.tolist()) == dropped[:3]   # highest entropy = lowest slot ids
    elif mode == "lowent":
        assert sorted(rescue.tolist()) == dropped[-3:]


def test_sig_anchor_scores_and_debug_split():
    from kveviction.selection import select_tokens
    from kveviction.strategies.anchors import with_anchors

    ctx = _anchor_ctx()
    weights = (torch.tensor([1.0, 0.0, 0.0, 0.0]), 0.0)  # signature == entropy
    policy = with_anchors(get_policy("window"), "sig", 2, weights)
    idx, debug = select_tokens(policy, ctx, return_debug=True)
    torch.testing.assert_close(debug["sig_score"], ctx.entropy)
    assert debug["anchor_extra"].numel() == 2
    # backend_keep is the base choice; final = base + rescued.
    assert set(idx.tolist()) == set(debug["backend_keep"].tolist()) | set(debug["anchor_extra"].tolist())
    assert debug["backend_keep"].numel() == ctx.budget


def test_anchor_wrapper_rejects_invalid_configs():
    from kveviction.strategies.anchors import with_anchors

    assert with_anchors(get_policy("window"), "none", 5).name == "window"
    assert with_anchors(get_policy("rkv"), "anchor", 0) is get_policy("rkv")
    with pytest.raises(ValueError, match="head-wise"):
        with_anchors(get_policy("rkv"), "anchor", 2)
    with pytest.raises(ValueError, match="sig"):
        with_anchors(get_policy("window"), "sig", 2)
    with pytest.raises(ValueError, match="unknown anchor mode"):
        with_anchors(get_policy("window"), "bogus", 2)


def test_only_entropy_anchors_request_entropy():
    from kveviction.strategies.anchors import with_anchors
    from kveviction.strategies.token import get_policy, policy_names

    assert not any(get_policy(name).needs_entropy for name in policy_names())
    window = get_policy("window")
    assert with_anchors(window, "anchor", 4).needs_entropy
    assert with_anchors(window, "lowent", 4).needs_entropy
    assert with_anchors(window, "sig", 4, sig=(torch.ones(4), 0.0)).needs_entropy
    assert not with_anchors(window, "random", 4).needs_entropy
