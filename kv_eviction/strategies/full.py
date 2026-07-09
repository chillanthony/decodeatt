"""FullKV no-eviction baseline."""
from __future__ import annotations

import torch

from .base import SelectionContext, TokenEvictionPolicy


class FullKVPolicy(TokenEvictionPolicy):
    name = "fullkv"
    never_evict = True

    def observation_window(self, params: dict, default: int) -> int:
        return 0

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return torch.ones_like(ctx.importance)

    def select_keep(self, ctx: SelectionContext) -> torch.Tensor:
        return torch.arange(ctx.slot_pos.numel(), device=ctx.slot_pos.device)
