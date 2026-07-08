"""H2O-style cumulative attention strategy."""
from __future__ import annotations

from .base import SelectionContext, TokenEvictionPolicy


class H2OPolicy(TokenEvictionPolicy):
    name = "h2o"

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.cumulative_attention
