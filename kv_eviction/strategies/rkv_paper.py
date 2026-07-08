"""Paper-faithful R-KV token selection utilities.

This module intentionally contains only tensor/cache selection logic. The
generation loop, tokenizer debug output, and cache mutation live in
``kv_eviction.runner_token``.
"""
from __future__ import annotations

import torch


def aggregate_gqa_attention(attn: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Map query-head attention rows to KV heads.

    HF returns attention probabilities per query head. For GQA models, several
    query heads share one KV head; we approximate the paper's group max-pooling
    using the returned attention probabilities.
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


def rkv_paper_importance(
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
        # The reference R-KV implementation slices out the trailing observation
        # window before softmax. HF exposes post-softmax attention over the full
        # cache, so re-normalize the candidate prefix to recover that semantics.
        cand = row[:, :n_cand]
        rows.append(cand / cand.sum(-1, keepdim=True).clamp_min(1e-12))
    if not rows:
        return torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=device)
    attn = torch.stack(rows, dim=1).mean(1)
    return max_pool_importance(attn, pool_kernel)


def rkv_paper_redundancy(
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


def select_rkv_paper(cache, attn_history, n, budget, params, return_debug=False):
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
        importance = rkv_paper_importance(
            attn_history, layer_idx, kv_heads, n_cand, n, pool_kernel, device
        )
        redundancy = rkv_paper_redundancy(
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
