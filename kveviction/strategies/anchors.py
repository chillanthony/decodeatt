"""Anchor protection: rescue a few evicted slots on top of any token policy.

``AnchoredPolicy`` wraps a slot-level (non head-wise) policy.  After the base
policy picks its ``budget`` slots, up to ``k`` of the dropped slots are
rescued according to ``mode``:

- ``anchor``: highest next-token entropy (forking tokens)
- ``lowent``: lowest entropy (control)
- ``random``: uniformly random dropped slots (control)
- ``sig``:    linear signature ``[entropy, cum_attn, position, concentration] @ w + b``
"""
from __future__ import annotations

import torch

from .base import SelectionContext, TokenEvictionPolicy

ANCHOR_MODES = ("none", "anchor", "lowent", "random", "sig")


class AnchoredPolicy(TokenEvictionPolicy):
    def __init__(self, base: TokenEvictionPolicy, mode: str, k: int, sig=None):
        if mode not in ANCHOR_MODES or mode == "none":
            raise ValueError(f"unknown anchor mode {mode!r}; expected one of {ANCHOR_MODES[1:]}")
        if base.needs_cache or base.needs_attn_history:
            raise ValueError("anchor modes are not supported for head-wise cache-aware strategies")
        if mode == "sig" and sig is None:
            raise ValueError("anchor_mode='sig' requires sig=(weights[4], bias)")
        self.base = base
        self.mode = mode
        self.k = int(k)
        self.sig = sig
        self.name = base.name
        self.never_evict = base.never_evict
        self.needs_attention = base.needs_attention

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return self.base.scores(ctx)

    def observation_window(self, params: dict, default: int) -> int:
        return self.base.observation_window(params, default)

    def anchor_scores(self, ctx: SelectionContext) -> torch.Tensor | None:
        """Per-slot rescue score for ``sig`` mode (None for other modes)."""
        if self.mode != "sig":
            return None
        entropy = _feature(ctx.entropy, ctx)
        position = ctx.slot_pos.float() / max(int(ctx.slot_pos.max()), 1)
        features = torch.stack([
            entropy,
            _feature(ctx.cumulative_mean, ctx),
            position,
            _feature(ctx.concentration, ctx),
        ], 1)
        weights, bias = self.sig
        return features @ weights + bias

    def select_with_rescue(self, ctx: SelectionContext):
        """Return ``(keep, rescued, anchor_score)`` slot indices for ``ctx``."""
        n = ctx.slot_pos.numel()
        dev = ctx.slot_pos.device
        keep_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        keep_mask[self.base.select_keep(ctx)] = True
        drop = (~keep_mask).nonzero(as_tuple=True)[0]
        rescue = torch.empty(0, dtype=torch.long, device=dev)
        anchor_score = self.anchor_scores(ctx)
        if self.k > 0 and drop.numel():
            entropy = _feature(ctx.entropy, ctx)
            if self.mode == "lowent":
                order = torch.argsort(entropy[drop])
            elif self.mode == "sig":
                order = torch.argsort(anchor_score[drop], descending=True)
            elif self.mode == "random":
                order = torch.randperm(drop.numel(), device=drop.device)
            else:
                order = torch.argsort(entropy[drop], descending=True)
            rescue = drop[order[: self.k]]
            keep_mask[rescue] = True
        return keep_mask.nonzero(as_tuple=True)[0], rescue, anchor_score

    def select_keep(self, ctx: SelectionContext) -> torch.Tensor:
        return self.select_with_rescue(ctx)[0]


def _feature(value, ctx: SelectionContext) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(ctx.slot_pos, dtype=torch.float)
    return value


def with_anchors(policy: TokenEvictionPolicy, mode: str, k: int, sig=None) -> TokenEvictionPolicy:
    """Wrap ``policy`` with anchor rescue, or return it unchanged when disabled."""
    if mode == "none" or k <= 0:
        return policy
    return AnchoredPolicy(policy, mode, k, sig)


__all__ = ["ANCHOR_MODES", "AnchoredPolicy", "with_anchors"]
