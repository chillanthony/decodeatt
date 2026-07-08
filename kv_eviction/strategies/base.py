"""Shared interfaces for token-level KV eviction strategies."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectionContext:
    importance: torch.Tensor
    cumulative_attention: torch.Tensor
    window_attention: torch.Tensor
    key_reps: torch.Tensor | None
    slot_pos: torch.Tensor
    budget: int
    recent: int
    sink: int
    params: dict


class TokenEvictionPolicy:
    name = "base"
    needs_key_reps = False

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        raise NotImplementedError

    def select_keep(self, ctx: SelectionContext) -> torch.Tensor:
        n = ctx.slot_pos.numel()
        dev = ctx.slot_pos.device
        if n <= ctx.budget:
            return torch.arange(n, device=dev)

        keep = torch.zeros(n, dtype=torch.bool, device=dev)
        if ctx.sink > 0:
            keep[: min(ctx.sink, n)] = True
        if ctx.recent > 0:
            keep[max(0, n - ctx.recent):] = True

        free = (~keep).nonzero(as_tuple=True)[0]
        n_pick = max(0, ctx.budget - int(keep.sum()))
        if n_pick > 0 and free.numel() > 0:
            picked = self._pick_free(ctx, free, n_pick)
            keep[picked] = True
        return keep.nonzero(as_tuple=True)[0]

    def _pick_free(self, ctx: SelectionContext, free: torch.Tensor, n_pick: int) -> torch.Tensor:
        score = self.scores(ctx)
        return free[torch.argsort(score[free], descending=True)[:n_pick]]
