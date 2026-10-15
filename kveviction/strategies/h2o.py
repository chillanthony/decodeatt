"""H2O-style cumulative attention strategy."""
from __future__ import annotations

import torch

from .base import SelectionContext, TokenEvictionPolicy
from .rkv_official import rkv_importance


def select_h2o(cache, attn_history, n, budget, params, return_debug=False):
    device = cache.layers[0].keys.device
    window_size = 1
    valid_mask = params.get("_valid_mask")
    if budget - window_size <= 0:
        raise ValueError("H2O budget must be greater than 1")
    batch_size = int(cache.layers[0].keys.shape[0])
    batched = batch_size > 1
    if n < budget:
        idx = [
            torch.arange(n, device=device).expand(
                (batch_size, layer.keys.shape[1], n) if batched else (layer.keys.shape[1], n)
            )
            for layer in cache.layers
        ]
        if return_debug:
            return idx, _debug(idx, torch.full((n,), float("inf"), device=device))
        return idx

    n_cand = n - window_size
    per_layer = []
    score_acc = torch.zeros((batch_size, n) if batched else (n,), dtype=torch.float32, device=device)
    score_count = 0
    for layer_idx, layer in enumerate(cache.layers):
        kv_heads = layer.keys.shape[1]
        importance = rkv_importance(
            attn_history[-1:],
            layer_idx,
            kv_heads,
            n_cand,
            n,
            1,
            device,
            valid_mask=valid_mask,
        )
        global_importance = importance.mean(-2)
        if valid_mask is not None:
            global_importance = global_importance.masked_fill(
                ~valid_mask[..., :n_cand].to(device=device, dtype=torch.bool),
                torch.finfo(global_importance.dtype).min,
            )
        selected = torch.topk(global_importance, k=budget - window_size, dim=-1).indices
        if batched:
            selected = selected.unsqueeze(1).expand(batch_size, kv_heads, -1)
            recent = torch.arange(n_cand, n, device=device).expand(batch_size, kv_heads, -1)
        else:
            selected = selected.unsqueeze(0).expand(kv_heads, -1)
            recent = torch.arange(n_cand, n, device=device).expand(kv_heads, -1)
        per_layer.append(torch.cat([selected, recent], dim=-1))
        score_acc[..., :n_cand] += global_importance
        score_count += 1
    policy_score = score_acc / max(score_count, 1)
    policy_score[..., n_cand:] = float("inf")
    if return_debug:
        return per_layer, _debug(per_layer, policy_score)
    return per_layer


def _debug(idx, policy_score):
    representative = idx[0][0] if idx[0].ndim == 2 else idx[0][0, 0]
    return {
        "backend_keep": representative,
        "anchor_extra": torch.empty(0, dtype=torch.long, device=policy_score.device),
        "sig_score": None,
        "policy_score": policy_score,
    }


class H2OPolicy(TokenEvictionPolicy):
    name = "h2o"
    needs_cache = True
    needs_attn_history = True

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.cumulative_attention

    def observation_window(self, params: dict, default: int) -> int:
        return 1

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
        return select_h2o(cache, attn_history, n, budget, params, return_debug=return_debug)
