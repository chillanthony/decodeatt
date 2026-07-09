"""Parity checks for official R-KV HuggingFace baseline selection semantics."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from kv_eviction.runner_token import _attention_logit_rows, _compact_cache
from kv_eviction.strategies.base import SelectionContext
from kv_eviction.strategies.h2o import select_h2o
from kv_eviction.strategies.rkv import select_rkv
from kv_eviction.strategies.snapkv import select_snapkv
from kv_eviction.strategies.token import get_policy


@dataclass
class _Layer:
    keys: torch.Tensor
    values: torch.Tensor | None = None


@dataclass
class _Cache:
    layers: list[_Layer]


def _attention_scores(query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    group = q_heads // kv_heads
    if group == 1:
        return torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    query_states = query_states.view(batch_size, kv_heads, group, q_len, head_dim)
    key_states = key_states.unsqueeze(2)
    return (
        torch.matmul(query_states, key_states.transpose(3, 4)) / math.sqrt(head_dim)
    ).max(dim=2).values


def _similarity(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
) -> torch.Tensor:
    _, _, seq_len, _ = key_states.shape
    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)
    mask = similarity_cos > threshold
    k = max(1, int(seq_len * retain_ratio))
    indices = torch.where(
        mask,
        torch.arange(seq_len, device=key_states.device).view(1, 1, 1, seq_len),
        torch.zeros_like(mask, dtype=torch.long),
    )
    if retain_direction == "last":
        retain = torch.max(indices, dim=-1).values
    elif retain_direction == "first":
        retain = torch.min(indices, dim=-1).values
    elif retain_direction == "last_percent":
        retain = torch.topk(indices, k=k, dim=-1).values[:, :, 0]
    elif retain_direction == "first_percent":
        retain = torch.topk(indices, k=k, dim=-1, largest=False).values[:, :, -1]
    else:
        raise ValueError("retain_direction not supported")
    similarity_cos.scatter_(-1, retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


def _history(query_states: torch.Tensor, key_states: torch.Tensor, window: int):
    scores = _attention_scores(query_states, key_states)
    return [[scores[0, :, -window + row, :].clone()] for row in range(window)]


def _case(kv_heads: int = 2):
    torch.manual_seed(11)
    n_total = 13
    head_dim = 5
    key_states = torch.randn(1, kv_heads, n_total, head_dim)
    value_states = torch.randn(1, kv_heads, n_total, head_dim)
    query_states = torch.randn(1, kv_heads, n_total, head_dim)
    cache = _Cache([_Layer(keys=key_states.clone(), values=value_states.clone())])
    return cache, key_states, value_states, query_states


def test_snapkv_matches_official_update_indices():
    cache, key_states, _, query_states = _case()
    budget = 8
    window = 3
    kernel = 5
    actual = select_snapkv(
        cache,
        _history(query_states, key_states, window),
        key_states.shape[-2],
        budget,
        {"window_size": window, "kernel_size": kernel},
    )[0]

    scores = _attention_scores(query_states, key_states)
    attn = torch.softmax(scores[:, :, -window:, :-window], dim=-1, dtype=torch.float32).mean(dim=-2)
    pooled = torch.nn.functional.max_pool1d(
        attn, kernel_size=kernel, padding=kernel // 2, stride=1
    )[:, :, : key_states.shape[-2] - window]
    selected = pooled.topk(budget - window, dim=-1).indices[0]
    recent = torch.arange(key_states.shape[-2] - window, key_states.shape[-2]).expand(
        key_states.shape[1], -1
    )
    expected = torch.cat([selected, recent], dim=-1)
    torch.testing.assert_close(actual, expected)


def test_h2o_matches_official_shared_head_indices():
    cache, key_states, _, query_states = _case()
    budget = 7
    actual = select_h2o(
        cache,
        _history(query_states, key_states, 1),
        key_states.shape[-2],
        budget,
        {},
    )[0]

    scores = _attention_scores(query_states[:, :, -1:, :], key_states).squeeze(2)
    importance = torch.softmax(scores[:, :, :-1], dim=-1, dtype=torch.float32).mean(dim=-2)
    selected = importance.topk(budget - 1, dim=-1).indices[0]
    expected = torch.cat([selected, torch.tensor([key_states.shape[-2] - 1])])
    expected = expected.unsqueeze(0).expand(key_states.shape[1], -1)
    torch.testing.assert_close(actual, expected)


def test_streamingllm_matches_official_first_plus_local_window():
    policy = get_policy("streamingllm")
    ctx = SelectionContext(
        importance=torch.zeros(12),
        cumulative_attention=torch.zeros(12),
        window_attention=torch.zeros(12),
        key_reps=None,
        slot_pos=torch.arange(12),
        budget=7,
        recent=0,
        sink=3,
        params={"first_tokens": 3},
    )
    torch.testing.assert_close(policy.select_keep(ctx), torch.tensor([0, 1, 2, 8, 9, 10, 11]))


def test_rkv_layer_selector_matches_official_update_indices():
    cache, key_states, _, query_states = _case()
    budget = 8
    window = 3
    kernel = 5
    mix_lambda = 0.1
    params = {
        "alpha": window,
        "pool_kernel": kernel,
        "lambda": mix_lambda,
        "retain_ratio": 0.1,
        "retain_direction": "last",
        "redundancy_chunk": 4,
    }
    actual = select_rkv(
        cache, _history(query_states, key_states, window), key_states.shape[-2], budget, params
    )[0]

    scores = _attention_scores(query_states, key_states)
    attn = torch.softmax(scores[:, :, -window:, :-window], dim=-1, dtype=torch.float32).mean(dim=-2)
    pooled = torch.nn.functional.max_pool1d(
        attn, kernel_size=kernel, padding=kernel // 2, stride=1
    )[:, :, : key_states.shape[-2] - window]
    similarity = _similarity(
        key_states,
        retain_ratio=params["retain_ratio"],
        retain_direction=params["retain_direction"],
    )[:, :, :-window]
    final_score = pooled * mix_lambda - similarity * (1.0 - mix_lambda)
    selected = final_score.topk(budget - window, dim=-1).indices[0]
    recent = torch.arange(key_states.shape[-2] - window, key_states.shape[-2]).expand(
        key_states.shape[1], -1
    )
    expected = torch.cat([selected, recent], dim=-1)
    torch.testing.assert_close(actual, expected)


def test_headwise_cache_compaction_gathers_each_head_independently():
    keys = torch.arange(10, dtype=torch.float32).view(1, 2, 5, 1)
    values = keys + 100
    cache = _Cache([_Layer(keys=keys.clone(), values=values.clone())])
    idx = [torch.tensor([[3, 1, 4], [0, 2, 4]])]
    k = _compact_cache(cache, idx)
    assert k == 3
    torch.testing.assert_close(cache.layers[0].keys[0, :, :, 0], torch.tensor([[3.0, 1.0, 4.0], [5.0, 7.0, 9.0]]))
    torch.testing.assert_close(cache.layers[0].values[0, :, :, 0], torch.tensor([[103.0, 101.0, 104.0], [105.0, 107.0, 109.0]]))


class _IdentityProj(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeAttention:
    head_dim = 2

    def __init__(self):
        self.q_proj = _IdentityProj()


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeAttention()


class _FakeBase:
    def __init__(self):
        self.layers = [_FakeLayer()]

    def rotary_emb(self, hidden, position_ids):
        shape = (*hidden.shape[:-1], 2)
        return torch.ones(shape, device=hidden.device), torch.zeros(shape, device=hidden.device)


class _FakeModel:
    def __init__(self):
        self.model = _FakeBase()


@dataclass
class _Outputs:
    hidden_states: tuple[torch.Tensor, ...]


def test_runner_attention_logit_rows_match_official_scores_without_attentions():
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    keys = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    cache = _Cache([_Layer(keys=keys)])
    rows = _attention_logit_rows(_FakeModel(), _Outputs((hidden,)), cache, torch.tensor([[0]]), 3)
    q = hidden.view(1, 1, 2, 2).transpose(1, 2)
    expected = _attention_scores(q, keys)[0, :, -1, :]
    assert rows is not None
    torch.testing.assert_close(rows[0], expected)


if __name__ == "__main__":
    test_snapkv_matches_official_update_indices()
    test_h2o_matches_official_shared_head_indices()
    test_streamingllm_matches_official_first_plus_local_window()
    test_rkv_layer_selector_matches_official_update_indices()
    test_headwise_cache_compaction_gathers_each_head_independently()
    test_runner_attention_logit_rows_match_official_scores_without_attentions()
    print("Official baseline parity tests passed")
