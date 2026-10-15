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


def rkv_importance(
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


def rkv_redundancy(
    keys: torch.Tensor,
    threshold: float,
    retain_ratio: float,
    retain_direction: str,
    eps: float,
    chunk_size: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference-compatible R-KV key redundancy per KV head.

    This matches upstream ``cal_similarity`` while chunking over query rows to
    avoid materializing every row chunk at once.
    """
    original_shape = keys.shape
    if keys.ndim not in {3, 4}:
        raise ValueError(f"R-KV keys must have shape [H,N,D] or [B,H,N,D], got {tuple(keys.shape)}")
    prefix = original_shape[:-2]
    n_cand = original_shape[-2]
    keys = keys.reshape(-1, n_cand, original_shape[-1])
    flat_heads = keys.shape[0]
    if n_cand == 0:
        return torch.empty(*prefix, 0, dtype=torch.float32, device=keys.device)
    keys = keys.float() / (keys.float().norm(dim=-1, keepdim=True) + eps)
    flat_valid = None
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=keys.device, dtype=torch.bool)
        if len(prefix) == 2:
            if valid_mask.ndim == 1:
                valid_mask = valid_mask.unsqueeze(0)
            flat_valid = valid_mask.unsqueeze(1).expand(prefix[0], prefix[1], n_cand).reshape(
                flat_heads, n_cand
            )
        else:
            flat_valid = valid_mask.reshape(1, n_cand).expand(flat_heads, -1)
    col_ids = torch.arange(n_cand, device=keys.device).view(1, 1, n_cand)
    col_sum = torch.zeros(flat_heads, n_cand, dtype=torch.float32, device=keys.device)
    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        sim = keys[:, start:end] @ keys.transpose(1, 2)
        local_rows = torch.arange(end - start, device=keys.device)
        global_rows = torch.arange(start, end, device=keys.device)
        sim[:, local_rows, global_rows] = 0.0
        if flat_valid is not None:
            sim = sim * flat_valid[:, start:end].unsqueeze(-1)
            sim = sim * flat_valid.unsqueeze(1)

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
    denominator = (
        flat_valid.sum(-1, keepdim=True).clamp_min(1)
        if flat_valid is not None
        else n_cand
    )
    redundancy = col_sum / denominator
    if flat_valid is not None:
        redundancy = redundancy.masked_fill(~flat_valid, torch.finfo(redundancy.dtype).min)
    return torch.softmax(redundancy, dim=-1).reshape(*prefix, n_cand)


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
    valid_mask = params.get("_valid_mask")

    batch_size = int(cache.layers[0].keys.shape[0])
    batched = batch_size > 1
    if n < budget:
        idx = [
            torch.arange(n, device=device).expand(
                (batch_size, layer.keys.shape[1], n) if batched else (layer.keys.shape[1], n)
            )
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
    score_acc = torch.zeros((batch_size, n) if batched else (n,), dtype=torch.float32, device=device)
    score_count = 0
    for layer_idx, layer in enumerate(cache.layers):
        keys = layer.keys[:, :, :n, :] if batched else layer.keys[0, :, :n, :]
        kv_heads = keys.shape[1] if batched else keys.shape[0]
        importance = rkv_importance(
            attn_history[-alpha:], layer_idx, kv_heads, n_cand, n, pool_kernel, device,
            valid_mask=valid_mask,
        )
        redundancy = rkv_redundancy(
            keys, threshold, retain_ratio, retain_direction, eps, chunk_size,
            valid_mask=valid_mask,
        )[..., :n_cand]
        final_score = importance * lam - redundancy * (1.0 - lam)
        if valid_mask is not None:
            candidate_valid = valid_mask[..., :n_cand].to(device=device, dtype=torch.bool)
            if batched:
                final_score = final_score.masked_fill(
                    ~candidate_valid.unsqueeze(1), torch.finfo(final_score.dtype).min
                )
            else:
                final_score = final_score.masked_fill(
                    ~candidate_valid, torch.finfo(final_score.dtype).min
                )
        selected = torch.topk(final_score, k=budget - alpha, dim=-1).indices
        recent_shape = (batch_size, kv_heads, alpha) if batched else (kv_heads, alpha)
        recent = torch.arange(n_cand, n, device=device).expand(recent_shape)
        per_layer.append(torch.cat([selected, recent], dim=-1))
        if batched:
            score_acc[:, :n_cand] += final_score.sum(1)
        else:
            score_acc[:n_cand] += final_score.sum(0)
        score_count += kv_heads

    policy_score = score_acc / max(score_count, 1)
    policy_score[..., n_cand:] = float("inf")
    if return_debug:
        return per_layer, {
            "backend_keep": per_layer[0][0] if per_layer[0].ndim == 2 else per_layer[0][0, 0],
            "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
            "sig_score": None,
            "policy_score": policy_score,
        }
    return per_layer
