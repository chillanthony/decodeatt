"""SnapKV-style observation-window importance strategy."""
from __future__ import annotations

from .base import SelectionContext, TokenEvictionPolicy


class SnapKVPolicy(TokenEvictionPolicy):
    name = "snapkv"

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.importance
