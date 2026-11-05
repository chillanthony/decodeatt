"""Policy selection boundary shared by reference and serving runners."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .strategies.anchors import AnchoredPolicy
from .strategies.base import SelectionContext, TokenEvictionPolicy
from .strategies.token import get_policy


@dataclass
class EvictionDecision:
    """Slots to keep plus diagnostics.

    ``keep_indices`` is a sorted ``[k]`` tensor for slot-level policies, or a
    list of per-layer ``[kv_heads, k]`` tensors for head-wise policies.
    ``base_indices`` is the base policy's choice before anchor rescue.
    """

    keep_indices: object
    rescue_indices: torch.Tensor
    base_indices: object = None
    policy_score: torch.Tensor | None = None
    anchor_score: torch.Tensor | None = None


class EvictionController:
    """Turn a selection context (and cache, for cache-aware policies) into a decision."""

    def __init__(self, policy: TokenEvictionPolicy | str):
        self.policy = get_policy(policy) if isinstance(policy, str) else policy

    def decide(self, ctx: SelectionContext, *, cache=None, attn_history=None,
               return_debug: bool = False) -> EvictionDecision:
        policy = self.policy
        n = ctx.slot_pos.numel()
        dev = ctx.slot_pos.device
        no_rescue = torch.empty(0, dtype=torch.long, device=dev)

        if policy.cache_aware:
            result = policy.select_from_cache(
                cache, attn_history, n, ctx.budget, dict(ctx.params),
                ctx=ctx, return_debug=return_debug,
            )
            indices, debug = result if return_debug else (result, None)
            score = debug.get("policy_score") if debug else None
            if not isinstance(indices, list):
                indices = _sorted_unique(indices, n)
                if return_debug and score is None:
                    score = policy.scores(ctx)
            return EvictionDecision(indices, no_rescue, indices, score)

        if isinstance(policy, AnchoredPolicy):
            keep, rescue, anchor_score = policy.select_with_rescue(ctx)
            base = keep[~torch.isin(keep, rescue)] if return_debug else None
        else:
            keep = _sorted_unique(policy.select_keep(ctx), n)
            rescue, anchor_score, base = no_rescue, None, keep
        score = policy.scores(ctx) if return_debug else None
        return EvictionDecision(keep, rescue, base, score, anchor_score)


def _sorted_unique(indices: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    return mask.nonzero(as_tuple=True)[0]


__all__ = ["EvictionController", "EvictionDecision"]
