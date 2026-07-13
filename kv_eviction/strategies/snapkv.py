"""SnapKV-style observation-window importance strategy."""
from __future__ import annotations

import torch

from .base import SelectionContext, TokenEvictionPolicy
from .rkv_official import rkv_importance


def select_snapkv(cache, attn_history, n, budget, params, return_debug=False):
    device = cache.layers[0].keys.device
    window_size = int(params.get("window_size", params.get("alpha", 32)))
    kernel_size = int(params.get("kernel_size", params.get("pool_kernel", 7)))
    pooling = str(params.get("pooling", "avgpool"))
    if budget - window_size <= 0:
        raise ValueError("SnapKV budget must be greater than window_size")
    if n < budget:
        idx = [
            torch.arange(n, device=device).expand(layer.keys.shape[1], -1)
            for layer in cache.layers
        ]
        if return_debug:
            return idx, _debug(idx, torch.full((n,), float("inf"), device=device))
        return idx

    n_cand = n - window_size
    per_layer = []
    score_acc = torch.zeros(n, dtype=torch.float32, device=device)
    score_count = 0
    for layer_idx, layer in enumerate(cache.layers):
        kv_heads = layer.keys.shape[1]
        importance = rkv_importance(
            attn_history[-window_size:],
            layer_idx,
            kv_heads,
            n_cand,
            n,
            kernel_size,
            device,
            pooling,
        )
        selected = torch.topk(importance, k=budget - window_size, dim=-1).indices
        recent = torch.arange(n_cand, n, device=device).expand(kv_heads, -1)
        per_layer.append(torch.cat([selected, recent], dim=-1))
        score_acc[:n_cand] += importance.sum(0)
        score_count += kv_heads
    policy_score = score_acc / max(score_count, 1)
    policy_score[n_cand:] = float("inf")
    if return_debug:
        return per_layer, _debug(per_layer, policy_score)
    return per_layer


def _debug(idx, policy_score):
    return {
        "backend_keep": idx[0][0],
        "anchor_extra": torch.empty(0, dtype=torch.long, device=policy_score.device),
        "sig_score": None,
        "policy_score": policy_score,
    }


class SnapKVPolicy(TokenEvictionPolicy):
    name = "snapkv"
    needs_cache = True
    needs_attn_history = True

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return ctx.importance

    def observation_window(self, params: dict, default: int) -> int:
        return int(params.get("window_size", params.get("alpha", 32)))

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
        return select_snapkv(cache, attn_history, n, budget, params, return_debug=return_debug)
