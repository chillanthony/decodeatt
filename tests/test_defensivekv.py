"""Small tensor checks for CriticalKV/DefensiveKV scoring ports."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from kv_eviction.strategies.defensivekv import (
    critical_attention_scores,
    defensive_attention_scores,
    select_defensivekv,
)
from kv_eviction.strategies.base import HeadwiseSelection
from kv_eviction.runner_token import _append_valid_token, _compact_cache, _set_headwise_valid_masks


@dataclass
class _Layer:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class _Cache:
    layers: list[_Layer]


class _FakeAttention(torch.nn.Module):
    def __init__(self, q_heads: int, head_dim: int):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=q_heads)
        self.o_proj = torch.nn.Linear(q_heads * head_dim, q_heads * head_dim, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(q_heads * head_dim))


class _FakeLayer(torch.nn.Module):
    def __init__(self, q_heads: int, head_dim: int):
        super().__init__()
        self.self_attn = _FakeAttention(q_heads, head_dim)


def _history_from_rows(rows: list[torch.Tensor]):
    return [[row] for row in rows]


def test_defensive_aggregation_happens_before_window_meaning_collapses():
    rows = _history_from_rows([
        torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
    ])
    mean_scores, _ = critical_attention_scores(
        rows, layer_idx=0, kv_heads=1, n_cand=3, n_total=5, pool_kernel=1, device="cpu"
    )
    defensive_scores, _ = defensive_attention_scores(
        rows, layer_idx=0, kv_heads=1, n_cand=3, n_total=5, pool_kernel=1, device="cpu"
    )

    assert defensive_scores[0, 0] > mean_scores[0, 0]
    assert defensive_scores[0, 2] > mean_scores[0, 2]
    assert defensive_scores[0, 1] > mean_scores[0, 1]


def test_select_defensivekv_returns_rectangular_headwise_indices():
    torch.manual_seed(5)
    n = 7
    budget = 5
    window = 2
    kv_heads = 2
    head_dim = 2
    keys = torch.randn(1, kv_heads, n, head_dim)
    values = torch.randn(1, kv_heads, n, head_dim)
    cache = _Cache([_Layer(keys=keys, values=values)])
    history = _history_from_rows([
        torch.randn(kv_heads, n),
        torch.randn(kv_heads, n),
    ])

    idx = select_defensivekv(
        cache,
        history,
        n,
        budget,
        {
            "window_size": window,
            "kernel_size": 1,
            "_layers": [_FakeLayer(q_heads=kv_heads, head_dim=head_dim)],
        },
    )

    assert len(idx) == 1
    assert idx[0].shape == (kv_heads, budget)
    torch.testing.assert_close(
        idx[0][:, -window:],
        torch.arange(n - window, n).expand(kv_heads, -1),
    )


def test_select_defensivekv_variable_head_budget_returns_valid_masks():
    torch.manual_seed(13)
    n = 8
    budget = 5
    window = 2
    kv_heads = 2
    head_dim = 2
    keys = torch.randn(1, kv_heads, n, head_dim)
    values = torch.randn(1, kv_heads, n, head_dim)
    cache = _Cache([_Layer(keys=keys, values=values)])
    history = _history_from_rows([
        torch.tensor([[9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0]]),
    ])

    idx = select_defensivekv(
        cache,
        history,
        n,
        budget,
        {
            "window_size": window,
            "kernel_size": 1,
            "variable_head_budget": True,
            "_layers": [_FakeLayer(q_heads=kv_heads, head_dim=head_dim)],
        },
    )

    assert isinstance(idx, HeadwiseSelection)
    assert idx.valid_masks is not None
    lengths = idx.valid_masks[0].sum(dim=-1)
    assert lengths.unique().numel() > 1
    assert int(lengths.sum()) == kv_heads * budget


def test_headwise_valid_mask_runs_through_tiny_llama_forward():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    out = model(input_ids=input_ids, use_cache=True, attention_mask=torch.ones(1, 4, dtype=torch.long))
    cache = out.past_key_values

    selection = HeadwiseSelection(
        indices=[torch.tensor([[0, 2, 3], [1, 2, 3]])],
        valid_masks=[torch.tensor([[True, False, True], [True, True, True]])],
    )
    k, valid_masks = _compact_cache(cache, selection, return_valid_masks=True)
    assert k == 3
    assert valid_masks is not None
    assert valid_masks[0].sum(dim=-1).tolist() == [2, 3]

    forward_masks = _append_valid_token(valid_masks)
    _set_headwise_valid_masks(model, forward_masks)
    out = model(
        input_ids=torch.tensor([[5]]),
        past_key_values=cache,
        use_cache=True,
        attention_mask=torch.ones(1, k + 1, dtype=torch.long),
        position_ids=torch.tensor([[4]]),
        cache_position=torch.tensor([k]),
    )
    _set_headwise_valid_masks(model, None)
    assert out.logits.shape == (1, 1, 32)


if __name__ == "__main__":
    test_defensive_aggregation_happens_before_window_meaning_collapses()
    test_select_defensivekv_returns_rectangular_headwise_indices()
    test_select_defensivekv_variable_head_budget_returns_valid_masks()
    test_headwise_valid_mask_runs_through_tiny_llama_forward()
    print("DefensiveKV tests passed")
