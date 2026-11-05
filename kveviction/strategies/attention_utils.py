"""Attention-score helpers shared by attention-based eviction strategies."""
from __future__ import annotations

import math

import torch

__all__ = [
    "compute_attention_scores", "aggregate_gqa_attention", "pool_importance",
    "max_pool_importance", "observation_importance",
]


def compute_attention_scores(query_states: torch.Tensor, key_states: torch.Tensor, pooling: str = "max") -> torch.Tensor:
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads
    if query_group_size == 1:
        return torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    query_states = query_states.view(batch_size, kv_heads, query_group_size, q_len, head_dim)
    key_states = key_states.unsqueeze(2)
    attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) / math.sqrt(head_dim)
    if pooling == "mean":
        return attn_weights.mean(dim=2)
    if pooling == "max":
        return attn_weights.max(dim=2).values
    raise ValueError("Pooling method not supported")


def aggregate_gqa_attention(attn: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Map query-head attention rows to KV heads.

    This is used only for compatibility histories that are not already grouped
    by ``compute_attention_scores``.
    """
    q_heads = attn.shape[-2]
    if q_heads == kv_heads:
        return attn
    if q_heads % kv_heads == 0:
        group = q_heads // kv_heads
        return attn.reshape(*attn.shape[:-2], kv_heads, group, attn.shape[-1]).max(-2).values
    return attn.mean(-2, keepdim=True).expand(*attn.shape[:-2], kv_heads, attn.shape[-1])


def pool_importance(attn: torch.Tensor, kernel: int, pooling: str = "max") -> torch.Tensor:
    if kernel <= 1 or attn.numel() == 0:
        return attn
    pad = kernel // 2
    original_shape = attn.shape
    values = attn.reshape(-1, 1, original_shape[-1])
    pooling = pooling.lower()
    if pooling == "avgpool":
        pooling = "avg"
    elif pooling == "maxpool":
        pooling = "max"
    if pooling == "avg":
        pooled = torch.nn.functional.avg_pool1d(
            values, kernel_size=kernel, stride=1, padding=pad
        ).squeeze(1)
    elif pooling == "max":
        pooled = torch.nn.functional.max_pool1d(
            values, kernel_size=kernel, stride=1, padding=pad
        ).squeeze(1)
    else:
        raise ValueError(f"unsupported importance pooling {pooling!r}")
    return pooled[:, : attn.shape[-1]].reshape(*original_shape[:-1], attn.shape[-1])


def max_pool_importance(attn: torch.Tensor, kernel: int) -> torch.Tensor:
    return pool_importance(attn, kernel, "max")


def observation_importance(
    attn_history: list[list[torch.Tensor]],
    layer_idx: int,
    kv_heads: int,
    n_cand: int,
    n_total: int,
    pool_kernel: int,
    device,
    pooling: str = "max",
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax-normalized, window-averaged, pooled importance for one layer.

    Shared by SnapKV, H2O and R-KV: ``attn_history`` holds per-step,
    per-layer pre-softmax rows; the trailing observation window
    (``n_total - n_cand`` slots) is sliced off before softmax, as in the
    official implementations.
    """
    rows = []
    for step_rows in attn_history:
        if layer_idx >= len(step_rows):
            continue
        row = step_rows[layer_idx].to(device=device, dtype=torch.float32)
        row = aggregate_gqa_attention(row, kv_heads)
        if row.shape[-1] < n_total:
            row = torch.nn.functional.pad(row, (0, n_total - row.shape[-1]))
        # Official R-KV/SnapKV/H2O slice out the trailing observation window
        # before softmax.
        cand = row[..., :n_cand]
        if valid_mask is not None:
            mask = valid_mask[..., :n_cand].to(device=device, dtype=torch.bool)
            while mask.ndim < cand.ndim:
                mask = mask.unsqueeze(-2)
            cand = cand.masked_fill(~mask, torch.finfo(cand.dtype).min)
        rows.append(torch.softmax(cand, dim=-1, dtype=torch.float32))
    if not rows:
        return torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=device)
    attn = torch.stack(rows, dim=-2).mean(-2)
    return pool_importance(attn, pool_kernel, pooling)
