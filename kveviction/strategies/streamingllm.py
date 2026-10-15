"""StreamingLLM sink-token plus recent-window strategy."""
from __future__ import annotations

import torch

from .base import SelectionContext, TokenEvictionPolicy


class StreamingLLMPolicy(TokenEvictionPolicy):
    name = "streamingllm"

    def observation_window(self, params: dict, default: int) -> int:
        return 0

    def scores(self, ctx: SelectionContext) -> torch.Tensor:
        return torch.arange(ctx.slot_pos.numel(), device=ctx.slot_pos.device, dtype=torch.float32)

    def select_keep(self, ctx: SelectionContext) -> torch.Tensor:
        n = ctx.slot_pos.numel()
        dev = ctx.slot_pos.device
        if n <= ctx.budget:
            return torch.arange(n, device=dev)

        first_tokens = int(ctx.params.get("first_tokens", ctx.params.get("sink", ctx.sink)))
        first_tokens = max(0, min(first_tokens, n, ctx.budget))
        local_window_size = ctx.budget - first_tokens

        keep = torch.zeros(n, dtype=torch.bool, device=dev)
        if first_tokens > 0:
            keep[:first_tokens] = True
        if local_window_size > 0:
            keep[n - local_window_size:] = True
        return keep.nonzero(as_tuple=True)[0]
