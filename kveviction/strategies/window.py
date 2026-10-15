"""Observation-window attention strategy."""
from __future__ import annotations

from .base import SelectionContext, TokenEvictionPolicy


class WindowPolicy(TokenEvictionPolicy):
    name = "window"

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.window_attention
