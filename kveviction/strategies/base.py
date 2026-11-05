"""Shared interfaces for token-level KV eviction strategies."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectionContext:
    """Per-slot features for one request, aligned with the current cache slots."""

    importance: torch.Tensor
    cumulative_attention: torch.Tensor
    window_attention: torch.Tensor
    slot_pos: torch.Tensor
    budget: int
    recent: int
    sink: int
    params: dict
    # Anchor features (see strategies/anchors.py); None when not tracked.
    entropy: torch.Tensor | None = None
    cumulative_mean: torch.Tensor | None = None
    concentration: torch.Tensor | None = None


class TokenEvictionPolicy:
    """Base class for token eviction policies.

    Capability flags tell the runner what to compute:

    - ``needs_attention``: attention observations must be collected inside the
      observation window (forces an attention backend that returns them).
    - ``needs_attn_history``: keep per-step, per-layer attention rows.
    - ``needs_cache``: selection reads cache keys and may be head-wise
      (``select_from_cache`` returns a list of per-layer index tensors).
    - ``never_evict``: the runner never compacts (FullKV).
    """

    name = "base"
    needs_attention = True
    needs_cache = False
    needs_attn_history = False
    never_evict = False

    @property
    def cache_aware(self) -> bool:
        return self.needs_cache or self.needs_attn_history

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        raise NotImplementedError

    def observation_window(self, params: dict, default: int) -> int:
        return default

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
        if ctx is None:
            raise ValueError(f"{self.name} requires SelectionContext")
        idx = self.select_keep(ctx)
        if return_debug:
            return idx, {
                "backend_keep": idx,
                "anchor_extra": torch.empty(0, dtype=torch.long, device=idx.device),
                "sig_score": None,
                "policy_score": self.scores(ctx),
            }
        return idx
