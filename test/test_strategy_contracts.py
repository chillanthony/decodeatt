"""CPU contract tests for every eviction policy and the HF decoding loops.

Run from the repository root:

    python -m pytest -q test/test_strategy_contracts.py

The end-to-end cases build a tiny randomly initialized Llama (GQA: 4 query
heads, 2 KV heads) so they need no downloads or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from kveviction.strategies.base import SelectionContext
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
        key_reps=None,
        slot_pos=torch.arange(n),
        budget=budget,
        recent=recent,
        sink=sink,
        params={},
    )


def test_every_registered_policy_is_covered_here():
    assert set(EVICTING) == set(ATTENTION_FREE) | set(CACHE_AWARE)


# --------------------------------------------------------------------------
# Attention-free policies: SelectionContext -> kept slot indices
# --------------------------------------------------------------------------

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


def test_streamingllm_keeps_first_tokens_and_contiguous_tail():
    n, budget = 30, 10
    keep = get_policy("streamingllm").select_keep(_ctx(n, budget, sink=3)).tolist()
    assert keep == [0, 1, 2] + list(range(n - (budget - 3), n))


# --------------------------------------------------------------------------
# Cache-aware policies: dense cache + attention history -> head-wise indices
# --------------------------------------------------------------------------

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


@pytest.mark.parametrize("backend", CACHE_AWARE)
def test_cache_aware_selection_is_deterministic(backend):
    n, budget = 30, 12
    cache, history = _gqa_inputs(n)
    params = _CACHE_AWARE_PARAMS[backend]
    first = get_policy(backend).select_from_cache(cache, history, n, budget, dict(params))
    second = get_policy(backend).select_from_cache(cache, history, n, budget, dict(params))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------
# End-to-end decoding on a tiny random Llama (CPU)
# --------------------------------------------------------------------------

class _Tokenizer:
    eos_token_id = None
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


@pytest.fixture(scope="module")
def tiny_llama():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg).eval()
    prompt = torch.randint(1, 64, (1, 20), generator=torch.Generator().manual_seed(1))
    return model, prompt


_E2E = dict(budget=16, recent=4, sink=2, evict_every=4, max_new=40, do_sample=False)


@pytest.mark.parametrize("backend", sorted(policy_names()))
def test_end_to_end_generation_respects_budget(tiny_llama, backend):
    from kveviction.runner_token import generate_token_evict

    model, prompt = tiny_llama
    result = generate_token_evict(model, _Tokenizer(), prompt, backend=backend, **_E2E)
    assert len(result["gen_ids"]) == _E2E["max_new"]
    if backend == "fullkv":
        assert result["n_evict"] == 0
        assert result["final_cache_len"] == prompt.shape[1] + _E2E["max_new"]
        return
    assert result["n_evict"] > 0
    assert len(result["evict_events"]) == result["n_evict"]
    for event in result["evict_events"]:
        assert event["cache_len_after"] == _E2E["budget"]
        assert event["cache_len_before"] > event["cache_len_after"]
        assert event["step"] % _E2E["evict_every"] == 0
    assert result["final_cache_len"] <= _E2E["budget"] + _E2E["evict_every"]
    assert result["cache_len_curve"] == [_E2E["budget"]] * result["n_evict"]


@pytest.mark.parametrize("backend", sorted(set(policy_names()) - {"random"}))
def test_greedy_batch_matches_single_request(tiny_llama, backend):
    """The single and batched loops are separate code paths; keep them in sync."""
    from kveviction.runner_token import generate_token_evict, generate_token_evict_batch

    model, prompt = tiny_llama
    single = generate_token_evict(model, _Tokenizer(), prompt, backend=backend, **_E2E)
    batch = generate_token_evict_batch(
        model, _Tokenizer(), prompt, 2, backend=backend, seeds=[0, 0], **_E2E
    )
    for candidate in batch["candidates"]:
        assert candidate["gen_ids"] == single["gen_ids"]
        assert candidate["n_evict"] == single["n_evict"]
        assert candidate["cache_len_curve"] == single["cache_len_curve"]


def test_trace_scoring_runs_with_eviction(tiny_llama):
    from kveviction.trace_scoring import score_trace_token

    model, prompt = tiny_llama
    trace = torch.randint(1, 64, (1, 30), generator=torch.Generator().manual_seed(2))
    full = torch.cat([prompt, trace], dim=1)
    result = score_trace_token(
        model, _Tokenizer(), full, prompt.shape[1], [3, 15], "none",
        backend="rkv", budget=16, recent=4, sink=2, evict_every=4, span_len=4,
        policy_params={"alpha": 4}, debug=True,
    )
    assert result["n_corr"] + result["n_noncorr"] == trace.shape[1]
    assert result["n_corr"] == 8
    assert result["per_token_nll"].shape == (trace.shape[1],)
    assert result["eviction_events"]
    assert result["final_cache_len"] <= 16 + 4
