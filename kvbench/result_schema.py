"""Canonical per-candidate generation result shared by the HF and vLLM engines.

Every engine returns one dict per generated candidate with exactly
``CANDIDATE_KEYS``.  The evaluators score and serialize these dicts the same
way, so results from either engine can be compared and merged directly.
Fields an engine cannot measure are ``None`` (summaries skip them) rather
than missing.
"""
from __future__ import annotations

from kveviction.eviction_stats import eviction_metrics, uniform_length_summary

CANDIDATE_KEYS = (
    "text", "gen_ids", "seed", "prompt_len",
    "n_evict", "final_cache_len",
    "elapsed_sec", "tokens_per_sec",
    "batch_elapsed_sec", "batch_tokens_per_sec", "batch_size",
    "peak_memory_bytes", "mean_compression_ratio", "evict_events",
    # cache-length summary (kveviction.eviction_stats)
    "mean_effective_cache_len", "max_effective_cache_len", "min_effective_cache_len",
    "total_effective_kv_tokens", "effective_kv_tokens_per_layer_head",
    "head_budget_mean", "head_budget_std", "head_budget_min", "head_budget_max",
    "head_budget_entropy", "num_underfilled_heads",
    # eviction aggregates (kveviction.eviction_stats.eviction_metrics)
    "total_evicted_tokens", "total_effective_evicted_tokens",
    "mean_evicted_per_event", "mean_effective_evicted_per_event",
    "min_compression_ratio", "max_compression_ratio",
    "mean_effective_compression_ratio", "min_effective_compression_ratio",
    "max_effective_compression_ratio", "cache_len_curve", "effective_cache_len_curve",
    # timing breakdown
    "prefill_sec", "decode_sec", "decode_forward_sec", "attention_observation_sec",
    "eviction_sec_total", "eviction_sec_mean", "other_decode_sec", "timing",
)


def missing_candidate_keys(result: dict) -> set[str]:
    return set(CANDIDATE_KEYS) - set(result)


def check_candidate(result: dict) -> dict:
    """Return ``result`` if it carries every canonical key, else raise."""
    missing = missing_candidate_keys(result)
    if missing:
        raise KeyError(f"generation result is missing canonical fields: {sorted(missing)}")
    return result


def no_eviction_candidate(
    *,
    text: str,
    gen_ids: list[int],
    seed: int,
    prompt_len: int,
    heads_per_layer: list[int],
    elapsed_sec: float,
    batch_elapsed_sec: float,
    batch_tokens_per_sec: float,
    batch_size: int,
    peak_memory_bytes: int | None = None,
) -> dict:
    """Canonical result for an engine that keeps the full KV cache (no eviction).

    The cache holds ``prompt_len + len(gen_ids)`` tokens in every layer/head.
    Only end-to-end time is known, so the per-phase timing fields are ``None``.
    """
    final_len = prompt_len + len(gen_ids)
    metrics = eviction_metrics([])
    mean_compression = metrics.pop("mean_compression_ratio")
    result = {
        "text": text,
        "gen_ids": list(gen_ids),
        "seed": int(seed),
        "prompt_len": int(prompt_len),
        "n_evict": 0,
        "final_cache_len": final_len,
        "elapsed_sec": elapsed_sec,
        "tokens_per_sec": len(gen_ids) / elapsed_sec if elapsed_sec > 0 else 0.0,
        "batch_elapsed_sec": batch_elapsed_sec,
        "batch_tokens_per_sec": batch_tokens_per_sec,
        "batch_size": int(batch_size),
        "peak_memory_bytes": peak_memory_bytes,
        "mean_compression_ratio": mean_compression,
        "evict_events": [],
        **uniform_length_summary(final_len, heads_per_layer),
        **metrics,
        "prefill_sec": None,
        "decode_sec": None,
        "decode_forward_sec": None,
        "attention_observation_sec": 0.0,
        "eviction_sec_total": 0.0,
        "eviction_sec_mean": 0.0,
        "other_decode_sec": None,
        "timing": {},
    }
    return check_candidate(result)


__all__ = ["CANDIDATE_KEYS", "check_candidate", "missing_candidate_keys", "no_eviction_candidate"]
