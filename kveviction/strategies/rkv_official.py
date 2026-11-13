"""Official HuggingFace R-KV token selection utilities.

This module intentionally contains only the R-KV-specific selection logic
(redundancy scoring and global/per-layer selection) kept in parity with the
official implementation.  Attention scoring shared with SnapKV/H2O lives in
``kveviction.strategies.attention_utils``; the generation loop lives in
``kveviction.runner_token``.
"""
from __future__ import annotations

import torch

from .attention_utils import (  # noqa: F401  (re-exported for compatibility)
    aggregate_gqa_attention,
    compute_attention_scores,
    max_pool_importance,
    observation_importance,
    pool_importance,
)

# R-KV's importance term is the shared observation-window importance.
rkv_importance = observation_importance

# Target size of the per-head-group similarity temporaries in ``rkv_redundancy``.
_GROUP_BYTES = 256 * 2 ** 20


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
    if retain_direction not in {"last", "first", "last_percent", "first_percent"}:
        raise ValueError("retain_direction not supported")
    col_ids = torch.arange(n_cand, device=keys.device).view(1, 1, n_cand)
    col_sum = torch.zeros(flat_heads, n_cand, dtype=torch.float32, device=keys.device)
    k = max(1, int(n_cand * retain_ratio))
    rows = min(chunk_size, n_cand)
    # Heads are independent: process them in groups so the [heads, rows, n]
    # temporaries stay near _GROUP_BYTES however large the batch is.
    group = max(1, _GROUP_BYTES // (rows * n_cand * 4))
    for h0 in range(0, flat_heads, group):
        h1 = min(h0 + group, flat_heads)
        raw = keys[h0:h1].float()
        normed = raw / (raw.norm(dim=-1, keepdim=True) + eps)
        del raw
        valid = None if flat_valid is None else flat_valid[h0:h1]
        for start in range(0, n_cand, chunk_size):
            end = min(start + chunk_size, n_cand)
            sim = normed[:, start:end] @ normed.transpose(1, 2)
            local_rows = torch.arange(end - start, device=keys.device)
            global_rows = torch.arange(start, end, device=keys.device)
            sim[:, local_rows, global_rows] = 0.0
            if valid is not None:
                sim.mul_(valid[:, start:end].unsqueeze(-1))   # exact: multiplies by 0 or 1
                sim.mul_(valid.unsqueeze(1))

            mask = sim > threshold
            if retain_direction == "last":
                # Largest column above the threshold, else 0 (as max over where(mask, col, 0)),
                # without materializing an int64 index matrix.
                last = (n_cand - 1) - mask.flip(-1).to(torch.uint8).argmax(dim=-1)
                retain = torch.where(mask.any(dim=-1), last, torch.zeros_like(last))
            else:
                indices = torch.where(mask, col_ids, torch.zeros_like(mask, dtype=torch.long))
                if retain_direction == "first":
                    retain = torch.min(indices, dim=-1).values
                elif retain_direction == "last_percent":
                    retain = torch.topk(indices, k=k, dim=-1).values[:, :, 0]
                else:
                    retain = torch.topk(indices, k=k, dim=-1, largest=False).values[:, :, -1]
                del indices
            del mask
            sim.scatter_(-1, retain.unsqueeze(-1), 0)
            col_sum[h0:h1] += sim.sum(dim=1)
            del sim
        del normed
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
