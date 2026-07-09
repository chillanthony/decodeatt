"""Official HuggingFace R-KV token selection utilities.

This module intentionally contains only tensor/cache selection logic. The
generation loop, tokenizer debug output, and cache mutation live in
``kv_eviction.runner_token``.
"""
from __future__ import annotations

import math

import torch


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
    q_heads = attn.shape[0]
    if q_heads == kv_heads:
        return attn
    if q_heads % kv_heads == 0:
        group = q_heads // kv_heads
        return attn.view(kv_heads, group, attn.shape[-1]).max(1).values
    return attn.mean(0, keepdim=True).expand(kv_heads, attn.shape[-1])


def max_pool_importance(attn: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1 or attn.numel() == 0:
        return attn
    pad = kernel // 2
    pooled = torch.nn.functional.max_pool1d(
        attn.unsqueeze(1), kernel_size=kernel, stride=1, padding=pad
    ).squeeze(1)
    return pooled[:, : attn.shape[-1]]


def rkv_importance(
    attn_history: list[list[torch.Tensor]],
    layer_idx: int,
    kv_heads: int,
    n_cand: int,
    n_total: int,
    pool_kernel: int,
    device,
) -> torch.Tensor:
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
        cand = row[:, :n_cand]
        rows.append(torch.softmax(cand, dim=-1, dtype=torch.float32))
    if not rows:
        return torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=device)
    attn = torch.stack(rows, dim=1).mean(1)
    return max_pool_importance(attn, pool_kernel)


def rkv_redundancy(
    keys: torch.Tensor,
    threshold: float,
    retain_ratio: float,
    retain_direction: str,
    eps: float,
    chunk_size: int,
) -> torch.Tensor:
    """Reference-compatible R-KV key redundancy per KV head.

    This matches upstream ``cal_similarity`` while chunking over query rows to
    avoid materializing every row chunk at once.
    """
    kv_heads, n_cand, _ = keys.shape
    if n_cand == 0:
        return torch.empty(kv_heads, 0, dtype=torch.float32, device=keys.device)
    keys = keys.float() / (keys.float().norm(dim=-1, keepdim=True) + eps)
    col_ids = torch.arange(n_cand, device=keys.device).view(1, 1, n_cand)
    col_sum = torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=keys.device)
    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        sim = keys[:, start:end] @ keys.transpose(1, 2)
        local_rows = torch.arange(end - start, device=keys.device)
        global_rows = torch.arange(start, end, device=keys.device)
        sim[:, local_rows, global_rows] = 0.0

        mask = sim > threshold
        indices = torch.where(mask, col_ids, torch.zeros_like(mask, dtype=torch.long))
        k = max(1, int(n_cand * retain_ratio))
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
        col_sum += sim.sum(dim=1)
    return torch.softmax(col_sum / n_cand, dim=-1)


def select_rkv_global(cache, attn_history, n, budget, params, return_debug=False):
    """Return token slots kept by the paper-faithful R-KV selection rule.

    ``budget`` matches upstream ``R1KV.budget``: the final compacted cache
    length, including the last alpha observation tokens.
    """
    device = cache.layers[0].keys.device
    alpha = int(params.get("alpha", 8))
    if budget - alpha <= 0:
        raise ValueError("R-KV budget must be greater than alpha")
    lam = float(params.get("lambda", params.get("redundancy_lambda", 0.1)))
    pool_kernel = int(params.get("pool_kernel", 7))
    threshold = float(params.get("similarity_threshold", 0.5))
    retain_ratio = float(params.get("retain_ratio", 0.1))
    retain_direction = str(params.get("retain_direction", "last"))
    eps = float(params.get("eps", 1e-8))
    chunk_size = int(params.get("redundancy_chunk", 512))

    obs = min(alpha, n)
    n_cand = n - obs
    if n <= budget:
        idx = torch.arange(n, device=device)
        if return_debug:
            score = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
            return idx, {
                "backend_keep": idx,
                "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
                "sig_score": None,
                "policy_score": score,
            }
        return idx

    agg_score = torch.zeros(n_cand, dtype=torch.float32, device=device)
    n_heads_total = 0
    for layer_idx, layer in enumerate(cache.layers):
        keys = layer.keys[0, :, :n, :]
        kv_heads = keys.shape[0]
        importance = rkv_importance(
            attn_history, layer_idx, kv_heads, n_cand, n, pool_kernel, device
        )
        redundancy = rkv_redundancy(
            keys, threshold, retain_ratio, retain_direction, eps, chunk_size
        )[:, :n_cand]
        score = lam * importance - (1.0 - lam) * redundancy
        agg_score += score.sum(0)
        n_heads_total += kv_heads
    agg_score /= max(n_heads_total, 1)

    n_select = max(0, budget - obs)
    selected = torch.argsort(agg_score, descending=True)[:n_select]
    obs_idx = torch.arange(n_cand, n, device=device)
    idx = torch.cat([selected, obs_idx]).sort().values
    if return_debug:
        policy_score = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
        policy_score[:n_cand] = agg_score
        return idx, {
            "backend_keep": idx,
            "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
            "sig_score": None,
            "policy_score": policy_score,
        }
    return idx


def select_rkv_layers(cache, attn_history, n, budget, params, return_debug=False):
    """Return official R-KV per-layer/per-KV-head kept slot indices.

    Each returned tensor has shape ``[num_kv_heads, budget]`` and preserves the
    upstream gather order: top-scoring candidate slots followed by the trailing
    observation window.
    """
    device = cache.layers[0].keys.device
    alpha = int(params.get("alpha", params.get("window_size", 8)))
    if budget - alpha <= 0:
        raise ValueError("R-KV budget must be greater than alpha")
    lam = float(params.get("lambda", params.get("mix_lambda", params.get("redundancy_lambda", 0.1))))
    pool_kernel = int(params.get("pool_kernel", params.get("kernel_size", 7)))
    threshold = float(params.get("similarity_threshold", 0.5))
    retain_ratio = float(params.get("retain_ratio", 0.1))
    retain_direction = str(params.get("retain_direction", "last"))
    eps = float(params.get("eps", 1e-8))
    chunk_size = int(params.get("redundancy_chunk", 512))

    if n < budget:
        idx = [
            torch.arange(n, device=device).expand(layer.keys.shape[1], -1)
            for layer in cache.layers
        ]
        if return_debug:
            return idx, {
                "backend_keep": idx[0][0],
                "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
                "sig_score": None,
                "policy_score": torch.full((n,), float("inf"), dtype=torch.float32, device=device),
            }
        return idx

    n_cand = n - alpha
    per_layer = []
    score_acc = torch.zeros(n, dtype=torch.float32, device=device)
    score_count = 0
    for layer_idx, layer in enumerate(cache.layers):
        keys = layer.keys[0, :, :n, :]
        kv_heads = keys.shape[0]
        importance = rkv_importance(
            attn_history[-alpha:], layer_idx, kv_heads, n_cand, n, pool_kernel, device
        )
        redundancy = rkv_redundancy(
            keys, threshold, retain_ratio, retain_direction, eps, chunk_size
        )[:, :n_cand]
        final_score = importance * lam - redundancy * (1.0 - lam)
        selected = torch.topk(final_score, k=budget - alpha, dim=-1).indices
        recent = torch.arange(n_cand, n, device=device).expand(kv_heads, -1)
        per_layer.append(torch.cat([selected, recent], dim=-1))
        score_acc[:n_cand] += final_score.sum(0)
        score_count += kv_heads

    policy_score = score_acc / max(score_count, 1)
    policy_score[n_cand:] = float("inf")
    if return_debug:
        return per_layer, {
            "backend_keep": per_layer[0][0],
            "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
            "sig_score": None,
            "policy_score": policy_score,
        }
    return per_layer
