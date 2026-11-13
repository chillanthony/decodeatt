"""Grouped-query attention for decode steps without copying the KV cache.

HF's ``eager`` and masked ``sdpa`` paths call ``repeat_kv`` on every layer and
step, materializing the cache ``num_heads / num_kv_heads`` times over (4x for
Llama-8B).  At large batch or long context that copy dominated decode time.

For a decode step (one query token) the query heads that share a KV head are
folded into the query-length axis instead: ``[B, H, 1, d] -> [B, Hkv, g, d]``
attends to the unrepeated ``[B, Hkv, n, d]`` cache, and the decode mask
``[B, 1, 1, n]`` broadcasts over the folded axis unchanged.  The math is the
same; only the GEMM shapes differ, so values can differ in the last bf16 bits.
Prefill (more than one query token) keeps HF's original code path.

``install()`` registers both functions under the existing ``eager`` and
``sdpa`` names, so model configs, checks and fingerprints keep their names.
"""
from __future__ import annotations

import torch
from torch import nn

_INSTALLED = False


def _repeat_kv(hidden: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden
    batch, kv_heads, length, dim = hidden.shape
    return hidden[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(
        batch, kv_heads * groups, length, dim
    )


def _hf_eager(module, query, key, value, attention_mask, scaling, dropout=0.0):
    """HF's ``eager_attention_forward`` (Llama / Qwen2), used for prefill."""
    groups = query.shape[1] // key.shape[1]
    key_states = _repeat_kv(key, groups)
    value_states = _repeat_kv(value, groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def _decode_mask(attention_mask, length):
    """``[B, 1, 1, n]`` slice of a decode mask (broadcasts over the folded heads)."""
    if attention_mask is None:
        return None
    mask = attention_mask[:, :, :, :length]
    if mask.shape[1] != 1:
        raise ValueError(f"expected a head-broadcast attention mask, got shape {tuple(mask.shape)}")
    return mask


def gqa_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if query.shape[2] != 1 or query.shape[1] == key.shape[1]:
        return _hf_eager(module, query, key, value, attention_mask, scaling, dropout)
    batch, heads, _, dim = query.shape
    kv_heads, length = key.shape[1], key.shape[2]
    groups = heads // kv_heads
    folded = query.reshape(batch, kv_heads, groups, dim)
    attn_weights = torch.matmul(folded, key.transpose(2, 3)) * scaling          # [B, Hkv, g, n]
    mask = _decode_mask(attention_mask, length)
    if mask is not None:
        attn_weights = attn_weights + mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)                              # [B, Hkv, g, d]
    attn_output = attn_output.reshape(batch, heads, 1, dim).transpose(1, 2).contiguous()
    return attn_output, attn_weights.reshape(batch, heads, 1, length)


def make_gqa_sdpa(original):
    """Wrap HF's ``sdpa_attention_forward``: decode steps attend without ``repeat_kv``."""

    def gqa_sdpa_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                                   is_causal=None, **kwargs):
        if query.shape[2] != 1 or query.shape[1] == key.shape[1] or kwargs.get("output_attentions"):
            return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                            is_causal=is_causal, **kwargs)
        batch, heads, _, dim = query.shape
        kv_heads, length = key.shape[1], key.shape[2]
        folded = query.reshape(batch, kv_heads, heads // kv_heads, dim)
        mask = _decode_mask(attention_mask, length)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            folded, key, value, attn_mask=mask,
            dropout_p=dropout if module.training else 0.0, scale=scaling, is_causal=False,
        )
        return attn_output.reshape(batch, heads, 1, dim).transpose(1, 2).contiguous(), None

    return gqa_sdpa_attention_forward


def install() -> None:
    """Register the grouped decode paths as transformers' ``eager`` and ``sdpa`` (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    AttentionInterface.register("eager", gqa_eager_attention_forward)
    AttentionInterface.register("sdpa", make_gqa_sdpa(sdpa_attention_forward))
    _INSTALLED = True
