"""Official HuggingFace R-KV strategy implementation and registry entries."""
from __future__ import annotations

from .base import SelectionContext, TokenEvictionPolicy
from .rkv_official import (
    aggregate_gqa_attention,
    max_pool_importance,
    rkv_importance,
    rkv_redundancy,
    select_rkv_global,
    select_rkv_layers,
)

select_rkv = select_rkv_layers


class RKVPolicy(TokenEvictionPolicy):
    name = "rkv"
    needs_cache = True
    needs_attn_history = True

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.importance

    def observation_window(self, params: dict, default: int) -> int:
        return int(params.get("alpha", 8))

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
        return select_rkv(cache, attn_history, n, budget, params, return_debug=return_debug)
