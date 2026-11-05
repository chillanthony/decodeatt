"""Cache-length summaries, eviction event records and per-run metric aggregation."""
from __future__ import annotations

import torch

from kveviction.cache_manager import TransformersCacheManager


def mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def take_float(x):
    return [float(v) for v in x.detach().float().cpu().tolist()]


def take_int(x):
    return [int(v) for v in x.detach().cpu().tolist()]


def token_samples(tokenizer, slot_ids, idx, limit=64):
    if limit <= 0:
        return []
    idx = idx[:limit].detach().cpu().tolist()
    ids = slot_ids.detach().cpu().tolist()
    return [{"slot": int(i), "token_id": int(ids[i]),
             "text": tokenizer.decode([int(ids[i])], skip_special_tokens=False)}
            for i in idx]


def cache_length_summary(cache, physical_len: int) -> dict:
    cache = TransformersCacheManager._view(cache)
    if cache is None:
        lengths = torch.tensor([float(physical_len)])
    else:
        lengths = torch.cat([
            torch.full((layer.keys.shape[1],), float(physical_len), device=layer.keys.device)
            for layer in cache.layers
        ])

    total = float(lengths.sum())
    probs = lengths / max(total, 1e-12)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum()) if lengths.numel() else 0.0
    max_len = float(lengths.max()) if lengths.numel() else 0.0
    return {
        "mean_effective_cache_len": float(lengths.mean()) if lengths.numel() else 0.0,
        "max_effective_cache_len": max_len,
        "min_effective_cache_len": float(lengths.min()) if lengths.numel() else 0.0,
        "total_effective_kv_tokens": total,
        "effective_kv_tokens_per_layer_head": [
            [int(physical_len)] * int(layer.keys.shape[1])
            for layer in cache.layers
        ] if cache is not None else [[int(physical_len)]],
        "head_budget_mean": float(lengths.mean()) if lengths.numel() else 0.0,
        "head_budget_std": float(lengths.std(unbiased=False)) if lengths.numel() else 0.0,
        "head_budget_min": float(lengths.min()) if lengths.numel() else 0.0,
        "head_budget_max": max_len,
        "head_budget_entropy": entropy,
        "num_underfilled_heads": int((lengths < max_len).sum()) if lengths.numel() else 0,
    }


def uniform_length_summary(effective_len: int, heads_per_layer: list[int], device=None) -> dict:
    """Cache-length summary when every layer/KV head holds ``effective_len`` tokens."""
    lengths = torch.full((sum(heads_per_layer),), float(effective_len), device=device)
    total = float(lengths.sum())
    probs = lengths / max(total, 1e-12)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    return {
        "mean_effective_cache_len": float(effective_len),
        "max_effective_cache_len": float(effective_len),
        "min_effective_cache_len": float(effective_len),
        "total_effective_kv_tokens": total,
        "effective_kv_tokens_per_layer_head": [
            [effective_len] * heads for heads in heads_per_layer
        ],
        "head_budget_mean": float(effective_len),
        "head_budget_std": 0.0,
        "head_budget_min": float(effective_len),
        "head_budget_max": float(effective_len),
        "head_budget_entropy": entropy,
        "num_underfilled_heads": 0,
    }


def batch_cache_length_summaries(cache, valid_mask: torch.Tensor) -> list[dict]:
    """Summarize the effective cache length of every request in a static batch."""
    cache_view = TransformersCacheManager._view(cache)
    heads_per_layer = (
        [int(layer.keys.shape[1]) for layer in cache_view.layers]
        if cache_view is not None else [1]
    )
    return [
        uniform_length_summary(int(valid_mask[row].sum()), heads_per_layer, valid_mask.device)
        for row in range(valid_mask.shape[0])
    ]


def eviction_event(*, step: int, before_n: int, after_n: int,
                   before: dict, after: dict, eviction_sec: float) -> dict:
    """One compaction record, shared by single-request and batched runners."""
    return {
        "step": step,
        "cache_len_before": int(before_n),
        "cache_len_after": int(after_n),
        "compression_ratio": float(after_n / max(before_n, 1)),
        "evicted": int(before_n - after_n),
        "effective_cache_len_before_mean": before["mean_effective_cache_len"],
        "effective_cache_len_after_mean": after["mean_effective_cache_len"],
        "effective_cache_len_before_max": before["max_effective_cache_len"],
        "effective_cache_len_after_max": after["max_effective_cache_len"],
        "effective_compression_ratio": float(
            after["total_effective_kv_tokens"]
            / max(before["total_effective_kv_tokens"], 1.0)
        ),
        "effective_evicted": float(
            before["total_effective_kv_tokens"] - after["total_effective_kv_tokens"]
        ),
        "head_budget_mean": after["head_budget_mean"],
        "head_budget_std": after["head_budget_std"],
        "head_budget_min": after["head_budget_min"],
        "head_budget_max": after["head_budget_max"],
        "head_budget_entropy": after["head_budget_entropy"],
        "num_underfilled_heads": after["num_underfilled_heads"],
        "eviction_sec": eviction_sec,
    }


def eviction_metrics(events: list[dict]) -> dict:
    """Aggregate a request's eviction events into its result fields."""
    ratios = [event["compression_ratio"] for event in events]
    effective = [event["effective_compression_ratio"] for event in events]
    return {
        "mean_compression_ratio": mean(ratios, 1.0),
        "total_evicted_tokens": int(sum(event["evicted"] for event in events)),
        "total_effective_evicted_tokens": float(
            sum(event["effective_evicted"] for event in events)
        ),
        "mean_evicted_per_event": float(mean([event["evicted"] for event in events])),
        "mean_effective_evicted_per_event": float(
            mean([event["effective_evicted"] for event in events])
        ),
        "min_compression_ratio": min(ratios) if ratios else 1.0,
        "max_compression_ratio": max(ratios) if ratios else 1.0,
        "mean_effective_compression_ratio": mean(effective, 1.0),
        "min_effective_compression_ratio": min(effective) if effective else 1.0,
        "max_effective_compression_ratio": max(effective) if effective else 1.0,
        "cache_len_curve": [event["cache_len_after"] for event in events],
        "effective_cache_len_curve": [
            event["effective_cache_len_after_mean"] for event in events
        ],
    }
