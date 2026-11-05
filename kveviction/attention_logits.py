"""Recompute pre-softmax decode attention rows from hidden states and cache keys.

Cache-aware baselines (R-KV official parity) need attention logits rather than
post-softmax probabilities.  Instead of forcing eager attention every step, the
runner requests hidden states inside the observation window and recomputes the
last query row per layer here.
"""
from __future__ import annotations

import torch

from kveviction.strategies.attention_utils import compute_attention_scores


def base_model(model):
    return getattr(model, "model", getattr(model, "transformer", model))


def model_layers(model):
    base = base_model(model)
    return getattr(base, "layers", getattr(base, "h", None))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_q(query_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query_states * cos) + (_rotate_half(query_states) * sin)


def supports_attention_logits(model) -> bool:
    layers = model_layers(model)
    base = base_model(model)
    if layers is None or not hasattr(base, "rotary_emb"):
        return False
    if len(layers) == 0:
        return False
    attn = getattr(layers[0], "self_attn", None)
    return attn is not None and hasattr(attn, "q_proj") and hasattr(attn, "head_dim")


def attention_logit_rows_batch(model, outputs, cache, position_ids, n):
    """Return one ``[B, H, N]`` pre-softmax attention row per layer, or None."""
    if not getattr(outputs, "hidden_states", None):
        return None
    layers = model_layers(model)
    base = base_model(model)
    if layers is None or not hasattr(base, "rotary_emb"):
        return None

    rows = []
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj") or not hasattr(attn, "head_dim"):
            return None
        hidden = outputs.hidden_states[layer_idx][:, -1:, :]
        q = attn.q_proj(hidden)
        q = q.view(*hidden.shape[:-1], -1, attn.head_dim).transpose(1, 2)
        cos, sin = base.rotary_emb(hidden, position_ids)
        q = _apply_rotary_q(q, cos, sin)
        keys = cache.layers[layer_idx].keys[:, :, :n, :]
        rows.append(compute_attention_scores(q, keys)[:, :, -1, :].detach().float())
    return rows


def attention_logit_rows(model, outputs, cache, position_ids, n):
    """Single-request variant returning one ``[H, N]`` row per layer."""
    rows = attention_logit_rows_batch(model, outputs, cache, position_ids, n)
    return None if rows is None else [row[0] for row in rows]
