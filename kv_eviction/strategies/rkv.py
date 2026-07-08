"""R-KV strategy registry entries.

The paper-faithful selector lives in ``kv_eviction.strategies.rkv_paper`` because it
needs cache keys and observation-window attention rows in addition to the
generic ``SelectionContext``.
"""
from __future__ import annotations

from .base import SelectionContext, TokenEvictionPolicy


class RKVPolicy(TokenEvictionPolicy):
    name = "rkv"
    needs_key_reps = True

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.importance

    def select_keep(self, ctx: SelectionContext) -> torch.Tensor:
        """Generic fallback; paper R-KV uses ``kv_eviction.strategies.rkv_paper``."""
        return super().select_keep(ctx)


class RKVPaperAliasPolicy(TokenEvictionPolicy):
    """Backward-compatible alias for the paper-faithful R-KV implementation."""

    name = "rkv-paper"
    needs_key_reps = True

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.importance
