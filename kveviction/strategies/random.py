"""Random middle-token retention strategy."""
from __future__ import annotations

import torch

from .base import SelectionContext, TokenEvictionPolicy


class RandomPolicy(TokenEvictionPolicy):
    name = "random"

    def observation_window(self, params: dict, default: int) -> int:
        return 0

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return torch.zeros_like(ctx.importance)

    def _pick_free(self, ctx: SelectionContext, free: torch.Tensor, n_pick: int) -> torch.Tensor:
        order = torch.randperm(free.numel(), device=free.device)
        return free[order[:n_pick]]
