"""Policy selection boundary shared by reference and serving runners."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .strategies.token import SelectionContext, get_policy


@dataclass
class EvictionDecision:
    keep_indices: object
    rescue_indices: torch.Tensor
    policy_score: torch.Tensor | None = None
    debug: dict | None = None


class EvictionController:
    """Turn observations and cache metadata into a storage-neutral decision."""

    def __init__(self, backend: str, *, params: dict | None = None):
        self.backend = backend
        self.params = dict(params or {})

    def decide(self, context: SelectionContext, *, cache=None, attn_history=None,
               model=None, anchor_mode: str = "none", anchor_k: int = 0,
               anchor_features: dict | None = None, sig=None,
               return_debug: bool = False) -> EvictionDecision:
        policy = get_policy(self.backend)
        params = dict(context.params)
        params.update(self.params)
        if model is not None:
            params["_model"] = model
        if policy.needs_cache or policy.needs_attn_history:
            result = policy.select_from_cache(
                cache, attn_history, context.slot_pos.numel(), context.budget,
                params, ctx=context, return_debug=return_debug,
            )
            if return_debug:
                indices, debug = result
            else:
                indices, debug = result, None
        else:
            indices = policy.select_keep(context)
            debug = {"policy_score": policy.scores(context)} if return_debug else None
        score = debug.get("policy_score") if debug else None
        rescue = torch.empty(0, dtype=torch.long, device=context.slot_pos.device)
        if isinstance(indices, list):
            if anchor_mode != "none" and anchor_k:
                raise ValueError("anchor modes are not supported for head-wise selection")
            keep = indices
        else:
            keep_mask = torch.zeros(context.slot_pos.numel(), dtype=torch.bool,
                                    device=context.slot_pos.device)
            keep_mask[indices] = True
            drop = (~keep_mask).nonzero(as_tuple=True)[0]
            if anchor_mode != "none" and anchor_k > 0 and drop.numel():
                features = anchor_features or {}
                entropy = features.get("entropy", torch.zeros_like(context.slot_pos, dtype=torch.float))
                if anchor_mode == "lowent":
                    order = torch.argsort(entropy[drop])
                elif anchor_mode == "sig" and sig is not None:
                    cumulative = features.get("cumulative", torch.zeros_like(entropy))
                    position = context.slot_pos.float() / max(int(context.slot_pos.max()), 1)
                    concentration = features.get("concentration", torch.zeros_like(entropy))
                    score = torch.stack([entropy, cumulative, position, concentration], 1) @ sig[0] + sig[1]
                    order = torch.argsort(score[drop], descending=True)
                elif anchor_mode == "random":
                    order = torch.randperm(drop.numel(), device=drop.device)
                else:
                    order = torch.argsort(entropy[drop], descending=True)
                rescue = drop[order[:anchor_k]]
                keep_mask[rescue] = True
            keep = keep_mask.nonzero(as_tuple=True)[0]
        return EvictionDecision(
            keep_indices=keep,
            rescue_indices=rescue,
            policy_score=score,
            debug=debug,
        )


__all__ = ["EvictionController", "EvictionDecision"]
