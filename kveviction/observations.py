"""Observation providers used by eviction controllers.

Providers keep attention extraction out of the decoding loop.  The default
implementation accepts a model output and returns the same dense observations
the reference runner used before this split.
"""
from __future__ import annotations

from typing import Any, Protocol

import torch


class ObservationProvider(Protocol):
    def observe(self, outputs: Any, *, cache, n: int, position_ids, model=None) -> dict[str, Any]:
        ...


class AttentionObservationProvider:
    """Extract attention-window importance and optional attention history."""

    def __init__(self, *, decay: float = 0.9, needs_history: bool = False):
        self.decay = decay
        self.needs_history = needs_history

    def observe(self, outputs, *, cache, n: int, position_ids, model=None):
        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return {"importance": None, "history": None}
        batch = attentions[0].shape[0]
        row = torch.zeros((batch, n), device=attentions[0].device)
        history = []
        for attention in attentions:
            row = row + attention[:, :, -1, :n].mean(1).float()
            if self.needs_history:
                history.append(torch.log(attention[:, :, -1, :n].detach().float().clamp_min(1e-30)))
        row = row / len(attentions)
        if batch == 1:
            row = row[0]
            if history:
                history = [item[0] for item in history]
        return {"importance": row, "history": history or None}


class AttentionLogitObservationProvider:
    """Provider for R-KV q/k logits, injected with model-specific extractor."""

    def __init__(self, extractor):
        self.extractor = extractor

    def observe(self, outputs, *, cache, n: int, position_ids, model=None):
        rows = self.extractor(model, outputs, cache, position_ids, n)
        return {"importance": None, "history": rows}


__all__ = [
    "ObservationProvider", "AttentionObservationProvider",
    "AttentionLogitObservationProvider",
]
