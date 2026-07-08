"""CPU parity checks for the R-KV reference selection semantics.

Run from the repository root:

    python tests/test_rkv_parity.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from kv_eviction.strategies.rkv_paper import (
    rkv_paper_importance,
    rkv_paper_redundancy,
    select_rkv_paper,
)


@dataclass
class _Layer:
    keys: torch.Tensor


@dataclass
class _Cache:
    layers: list[_Layer]


def _reference_similarity(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
) -> torch.Tensor:
    """Copy of upstream R-KV ``cal_similarity`` for tiny CPU tensors."""
    _, _, seq_len, _ = key_states.shape
    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    sim.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)

    mask = sim > threshold
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

    sim.scatter_(-1, retain.unsqueeze(-1), 0)
    return sim.mean(dim=-2).softmax(dim=-1)


def _reference_attention_scores(query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
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


def _reference_rkv_indices(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    total_budget: int,
    window_size: int,
    kernel_size: int,
    mix_lambda: float,
    retain_ratio: float,
    retain_direction: str,
) -> torch.Tensor:
    attn_weights = _reference_attention_scores(query_states, key_states)
    attn_weights_sum = torch.softmax(
        attn_weights[:, :, -window_size:, :-window_size], dim=-1, dtype=torch.float32
    ).mean(dim=-2)
    attn_cache = torch.nn.functional.max_pool1d(
        attn_weights_sum,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    )
    similarity = _reference_similarity(
        key_states, retain_ratio=retain_ratio, retain_direction=retain_direction
    )[:, :, :-window_size]
    final_score = attn_cache * mix_lambda - similarity * (1.0 - mix_lambda)
    selected = final_score.topk(total_budget - window_size, dim=-1).indices
    obs = torch.arange(
        key_states.shape[-2] - window_size,
        key_states.shape[-2],
        device=key_states.device,
    ).view(1, 1, window_size)
    return torch.cat([selected, obs.expand_as(selected[:, :, :window_size])], dim=-1)


def _case():
    torch.manual_seed(7)
    n_total = 14
    window = 4
    total_budget = 9
    head_dim = 6
    key_states = torch.randn(1, 1, n_total, head_dim)
    value_states = torch.randn(1, 1, n_total, head_dim)
    query_states = torch.randn(1, 1, n_total, head_dim)
    cache = _Cache([_Layer(keys=key_states.clone())])
    params = {
        "alpha": window,
        "lambda": 0.1,
        "pool_kernel": 7,
        "similarity_threshold": 0.5,
        "retain_ratio": 0.1,
        "retain_direction": "last",
        "redundancy_chunk": 3,
    }
    return cache, key_states, value_states, query_states, total_budget, params


def _attn_history_from_full_probs(query_states: torch.Tensor, key_states: torch.Tensor, window: int):
    scores = _reference_attention_scores(query_states, key_states)
    probs = torch.softmax(scores[:, :, -window:, :], dim=-1, dtype=torch.float32)
    return [[probs[0, :, row, :].clone()] for row in range(window)]


def test_importance_matches_reference_candidate_softmax():
    _, key_states, _, query_states, _, params = _case()
    window = params["alpha"]
    n_total = key_states.shape[-2]
    n_cand = n_total - window
    history = _attn_history_from_full_probs(query_states, key_states, window)

    actual = rkv_paper_importance(
        history, 0, 1, n_cand, n_total, params["pool_kernel"], key_states.device
    )
    scores = _reference_attention_scores(query_states, key_states)
    expected_rows = torch.softmax(
        scores[:, :, -window:, :n_cand], dim=-1, dtype=torch.float32
    ).mean(dim=-2)[0]
    expected = torch.nn.functional.max_pool1d(
        expected_rows.unsqueeze(1),
        kernel_size=params["pool_kernel"],
        padding=params["pool_kernel"] // 2,
        stride=1,
    ).squeeze(1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_redundancy_matches_reference_cal_similarity():
    _, key_states, _, _, _, params = _case()
    n_cand = key_states.shape[-2] - params["alpha"]
    actual = rkv_paper_redundancy(
        key_states[0, :, :n_cand, :],
        params["similarity_threshold"],
        params["retain_ratio"],
        params["retain_direction"],
        1e-8,
        params["redundancy_chunk"],
    )
    expected = _reference_similarity(
        key_states[:, :, :n_cand, :],
        threshold=params["similarity_threshold"],
        retain_ratio=params["retain_ratio"],
        retain_direction=params["retain_direction"],
    )[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_select_rkv_paper_matches_reference_single_layer_single_head():
    cache, key_states, _, query_states, total_budget, params = _case()
    history = _attn_history_from_full_probs(query_states, key_states, params["alpha"])

    actual = select_rkv_paper(
        cache, history, key_states.shape[-2], total_budget, params
    )
    reference = _reference_rkv_indices(
        query_states,
        key_states,
        total_budget=total_budget,
        window_size=params["alpha"],
        kernel_size=params["pool_kernel"],
        mix_lambda=params["lambda"],
        retain_ratio=params["retain_ratio"],
        retain_direction=params["retain_direction"],
    )[0, 0].sort().values

    assert actual.numel() == total_budget
    torch.testing.assert_close(actual, reference)


def test_select_rkv_paper_debug_scores_are_aligned():
    cache, key_states, _, query_states, total_budget, params = _case()
    history = _attn_history_from_full_probs(query_states, key_states, params["alpha"])
    idx, debug = select_rkv_paper(
        cache, history, key_states.shape[-2], total_budget, params, return_debug=True
    )

    n_cand = key_states.shape[-2] - params["alpha"]
    assert torch.isfinite(debug["policy_score"][:n_cand]).all()
    assert torch.isinf(debug["policy_score"][n_cand:]).all()
    torch.testing.assert_close(debug["backend_keep"], idx)
    assert debug["anchor_extra"].numel() == 0
    assert debug["sig_score"] is None


def test_select_rkv_paper_rejects_budget_not_greater_than_alpha():
    cache, key_states, _, query_states, _, params = _case()
    history = _attn_history_from_full_probs(query_states, key_states, params["alpha"])
    try:
        select_rkv_paper(cache, history, key_states.shape[-2], params["alpha"], params)
    except ValueError as exc:
        assert "budget must be greater than alpha" in str(exc)
    else:
        raise AssertionError("expected ValueError for budget <= alpha")


if __name__ == "__main__":
    test_importance_matches_reference_candidate_softmax()
    test_redundancy_matches_reference_cal_similarity()
    test_select_rkv_paper_matches_reference_single_layer_single_head()
    test_select_rkv_paper_debug_scores_are_aligned()
    test_select_rkv_paper_rejects_budget_not_greater_than_alpha()
    print("R-KV parity tests passed")
