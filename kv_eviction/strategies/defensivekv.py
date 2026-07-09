"""CriticalKV and DefensiveKV selectors adapted to this runner.

The upstream DefensiveKV implementation is built on KVPRESS' packed per-head
cache.  This module ports the scoring rule while keeping this repository's
rectangular per-layer/per-head cache contract: every head keeps the same number
of slots, followed by the observation window.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .base import HeadwiseSelection, SelectionContext, TokenEvictionPolicy
from .rkv_official import aggregate_gqa_attention


def _layer_modules(params: dict):
    layers = params.get("_layers")
    if layers is None:
        model = params.get("_model")
        base = getattr(model, "model", getattr(model, "transformer", model))
        layers = getattr(base, "layers", getattr(base, "h", None))
    return layers


def _attention_score_rows(
    attn_history: list[list[torch.Tensor]],
    layer_idx: int,
    kv_heads: int,
    n_cand: int,
    n_total: int,
    pool_kernel: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for step_rows in attn_history:
        if layer_idx >= len(step_rows):
            continue
        row = step_rows[layer_idx].to(device=device, dtype=torch.float32)
        row = aggregate_gqa_attention(row, kv_heads)
        if row.shape[-1] < n_total:
            row = F.pad(row, (0, n_total - row.shape[-1]))
        rows.append(torch.softmax(row[:, :n_cand], dim=-1, dtype=torch.float32))

    if not rows:
        empty = torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=device)
        return empty, empty

    scores = torch.stack(rows, dim=1)  # [kv_heads, window, n_cand]
    ave_attn = scores.mean(dim=1)
    if pool_kernel > 1 and n_cand > 0:
        base = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = F.avg_pool1d(
            scores,
            kernel_size=pool_kernel,
            padding=pool_kernel // 2,
            stride=1,
        )[:, :, :n_cand]
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12) * base
    return scores, ave_attn


def critical_attention_scores(
    attn_history: list[list[torch.Tensor]],
    layer_idx: int,
    kv_heads: int,
    n_cand: int,
    n_total: int,
    pool_kernel: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, ave_attn = _attention_score_rows(
        attn_history, layer_idx, kv_heads, n_cand, n_total, pool_kernel, device
    )
    return rows.mean(dim=1), ave_attn


def defensive_attention_scores(
    attn_history: list[list[torch.Tensor]],
    layer_idx: int,
    kv_heads: int,
    n_cand: int,
    n_total: int,
    pool_kernel: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, ave_attn = _attention_score_rows(
        attn_history, layer_idx, kv_heads, n_cand, n_total, pool_kernel, device
    )
    max_scores = rows.max(dim=1).values
    defended = max_scores.clamp(min=max_scores.mean(dim=-1, keepdim=True))
    return defended, ave_attn


def _projected_value_norm(values: torch.Tensor, layer_module, eps: float) -> torch.Tensor:
    attn = getattr(layer_module, "self_attn", layer_module)
    o_proj = getattr(attn, "o_proj", None)
    if o_proj is None:
        raise ValueError("CriticalKV requires layer.self_attn.o_proj for value-norm scoring")

    kv_heads, n_cand, head_dim = values.shape
    config = getattr(attn, "config", getattr(layer_module, "config", None))
    q_heads = getattr(attn, "num_heads", None)
    if q_heads is None and config is not None:
        q_heads = getattr(config, "num_attention_heads", None)
    if q_heads is None:
        q_heads = kv_heads
    if q_heads % kv_heads != 0:
        raise ValueError("number of query heads must be divisible by KV heads")

    groups = q_heads // kv_heads
    weight = o_proj.weight.detach().to(device=values.device, dtype=torch.float32)
    wo = weight.transpose(0, 1).contiguous().view(q_heads, head_dim, -1)
    values = values.float()
    norm = torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=values.device)
    for q_head in range(q_heads):
        kv_head = q_head // groups
        projected = values[kv_head].matmul(wo[q_head])
        norm[kv_head] += projected.norm(p=1, dim=-1)
    norm /= groups
    return norm / norm.sum(dim=-1, keepdim=True).clamp_min(eps)


def _apply_critical_stage(
    scores: torch.Tensor,
    ave_attn: torch.Tensor,
    values: torch.Tensor,
    layer_module,
    n_select: int,
    threshold: float,
    eps: float,
) -> torch.Tensor:
    value_norm = _projected_value_norm(values, layer_module, eps)
    scores = scores * value_norm

    normalized = ave_attn / ave_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    sorted_scores, sorted_indices = torch.sort(normalized, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_scores, dim=-1)
    counts = torch.argmax((cumsum >= threshold).to(torch.int32), dim=-1)
    counts.clamp_(max=n_select)

    if int(counts.max()) > 0:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for head_idx in range(scores.shape[0]):
            k = int(counts[head_idx])
            if k > 0:
                mask[head_idx, sorted_indices[head_idx, :k]] = True
        scores = torch.where(mask, scores.max(), scores)
    return scores


def _variable_head_indices(scores: torch.Tensor, n_select: int, n_total: int) -> tuple[torch.Tensor, torch.Tensor]:
    kv_heads, n_cand = scores.shape
    total = max(0, min(kv_heads * n_select, scores.numel()))
    if total == 0:
        recent = torch.arange(n_cand, n_total, device=scores.device)
        idx = recent.expand(kv_heads, -1)
        return idx, torch.ones_like(idx, dtype=torch.bool)

    flat = torch.topk(scores.reshape(-1), k=total).indices
    head_ids = flat // n_cand
    token_ids = flat % n_cand
    buckets = [token_ids[head_ids == head] for head in range(kv_heads)]
    window = n_total - n_cand
    max_len = max((int(bucket.numel()) + window for bucket in buckets), default=window)
    idx = torch.zeros(kv_heads, max_len, dtype=torch.long, device=scores.device)
    valid = torch.zeros(kv_heads, max_len, dtype=torch.bool, device=scores.device)
    recent = torch.arange(n_cand, n_total, device=scores.device)
    for head, bucket in enumerate(buckets):
        selected = bucket
        if selected.numel() > 1:
            selected = selected[torch.argsort(scores[head, selected], descending=True)]
        values = torch.cat([selected, recent])
        idx[head, : values.numel()] = values
        valid[head, : values.numel()] = True
        if values.numel() < max_len and values.numel() > 0:
            idx[head, values.numel():] = values[-1]
    return idx, valid


def _select_layers(cache, attn_history, n, budget, params, *, defensive: bool, return_debug=False):
    device = cache.layers[0].keys.device
    window = int(params.get("window_size", params.get("alpha", 32)))
    if budget - window <= 0:
        raise ValueError("CriticalKV/DefensiveKV budget must be greater than window_size")
    kernel = int(params.get("kernel_size", params.get("pool_kernel", 5)))
    threshold = float(params.get("critical_threshold", 0.9))
    eps = float(params.get("eps", 1e-8))
    variable_heads = bool(params.get("variable_head_budget", params.get("adaptive_heads", False)))
    layers = _layer_modules(params)
    if layers is None:
        raise ValueError("CriticalKV/DefensiveKV requires the model layers in runtime params")

    if n < budget:
        idx = [
            torch.arange(n, device=device).expand(layer.keys.shape[1], -1)
            for layer in cache.layers
        ]
        if return_debug:
            return idx, _debug(idx, torch.full((n,), float("inf"), dtype=torch.float32, device=device))
        return idx

    n_cand = n - window
    n_select = budget - window
    per_layer = []
    per_layer_valid = []
    score_acc = torch.zeros(n, dtype=torch.float32, device=device)
    score_count = 0
    for layer_idx, layer in enumerate(cache.layers):
        kv_heads = layer.keys.shape[1]
        if defensive:
            scores, ave_attn = defensive_attention_scores(
                attn_history, layer_idx, kv_heads, n_cand, n, kernel, device
            )
        else:
            scores, ave_attn = critical_attention_scores(
                attn_history, layer_idx, kv_heads, n_cand, n, kernel, device
            )
        if layer.values is None:
            raise ValueError("CriticalKV/DefensiveKV requires cache values")
        values = layer.values[0, :, :n_cand, :]
        scores = _apply_critical_stage(
            scores, ave_attn, values, layers[layer_idx], n_select, threshold, eps
        )
        if variable_heads:
            layer_idx, layer_valid = _variable_head_indices(scores, n_select, n)
            per_layer.append(layer_idx)
            per_layer_valid.append(layer_valid)
        else:
            selected = torch.topk(scores, k=n_select, dim=-1).indices
            recent = torch.arange(n_cand, n, device=device).expand(kv_heads, -1)
            per_layer.append(torch.cat([selected, recent], dim=-1))
        score_acc[:n_cand] += scores.sum(0)
        score_count += kv_heads

    policy_score = score_acc / max(score_count, 1)
    policy_score[n_cand:] = float("inf")
    result = HeadwiseSelection(per_layer, per_layer_valid) if variable_heads else per_layer
    if return_debug:
        return result, _debug(per_layer, policy_score)
    return result


def _debug(idx, policy_score):
    return {
        "backend_keep": idx[0][0],
        "anchor_extra": torch.empty(0, dtype=torch.long, device=policy_score.device),
        "sig_score": None,
        "policy_score": policy_score,
    }


def select_criticalkv(cache, attn_history, n, budget, params, return_debug=False):
    return _select_layers(cache, attn_history, n, budget, params, defensive=False, return_debug=return_debug)


def select_defensivekv(cache, attn_history, n, budget, params, return_debug=False):
    return _select_layers(cache, attn_history, n, budget, params, defensive=True, return_debug=return_debug)


class CriticalKVPolicy(TokenEvictionPolicy):
    name = "criticalkv"
    needs_cache = True
    needs_attn_history = True

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.importance

    def observation_window(self, params: dict, default: int) -> int:
        return int(params.get("window_size", params.get("alpha", default)))

    def select_from_cache(
        self,
        cache,
        attn_history,
        n: int,
        budget: int,
        params: dict,
        *,
        ctx: SelectionContext | None = None,
        return_debug: bool = False,
    ):
        return select_criticalkv(cache, attn_history, n, budget, params, return_debug=return_debug)


class DefensiveKVPolicy(CriticalKVPolicy):
    name = "defensivekv"

    def select_from_cache(
        self,
        cache,
        attn_history,
        n: int,
        budget: int,
        params: dict,
        *,
        ctx: SelectionContext | None = None,
        return_debug: bool = False,
    ):
        return select_defensivekv(cache, attn_history, n, budget, params, return_debug=return_debug)
