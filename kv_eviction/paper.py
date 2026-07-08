"""Backward-compatible imports for paper R-KV strategy utilities."""

from kv_eviction.strategies.rkv_paper import (
    aggregate_gqa_attention,
    max_pool_importance,
    rkv_paper_importance,
    rkv_paper_redundancy,
    select_rkv_paper,
)

__all__ = [
    "aggregate_gqa_attention",
    "max_pool_importance",
    "rkv_paper_importance",
    "rkv_paper_redundancy",
    "select_rkv_paper",
]
