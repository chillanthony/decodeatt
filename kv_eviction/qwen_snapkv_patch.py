"""Qwen2-only integrated SnapKV prototype.

This is intentionally narrow: it monkeypatches the eager Qwen2 attention path
used by ``Qwen/Qwen2.5-0.5B-Instruct`` in Transformers 4.44-style releases.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import MethodType

import torch
from torch import nn


@dataclass
class QwenSnapKVStats:
    evictions: int = 0
    attention_observation_sec: float = 0.0
    eviction_sec_total: float = 0.0
    cache_len: int = 0
    layer_evictions: dict[int, int] = field(default_factory=dict)


class QwenSnapKVController:
    def __init__(
        self,
        budget: int,
        window_size: int = 32,
        kernel_size: int = 7,
        pooling: str = "avgpool",
    ):
        if budget <= window_size:
            raise ValueError("SnapKV budget must be greater than window_size")
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.histories: dict[int, list[torch.Tensor]] = {}
        self.stats = QwenSnapKVStats()
        self.enabled = False

    def reset(self) -> None:
        self.histories.clear()
        self.stats = QwenSnapKVStats()

    def observe_and_compress(self, attn_module, attn_weights: torch.Tensor, cache) -> None:
        if not self.enabled:
            return
        if cache is None or not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
            return
        layer_idx = getattr(attn_module, "layer_idx", None)
        if layer_idx is None or layer_idx >= len(cache.key_cache):
            return

        obs_start = time.perf_counter()
        key_states = cache.key_cache[layer_idx]
        value_states = cache.value_cache[layer_idx]
        kv_len = int(key_states.shape[-2])
        self.stats.cache_len = kv_len

        kv_heads = int(key_states.shape[1])
        q_heads = int(attn_weights.shape[1])
        row = attn_weights[0, :, -1, :kv_len].detach().float()
        if q_heads == kv_heads:
            grouped = row
        elif q_heads % kv_heads == 0:
            grouped = row.view(kv_heads, q_heads // kv_heads, kv_len).max(1).values
        else:
            grouped = row.mean(0, keepdim=True).expand(kv_heads, kv_len)

        history = self.histories.setdefault(layer_idx, [])
        history.append(grouped)
        del history[:-self.window_size]
        self.stats.attention_observation_sec += time.perf_counter() - obs_start

        if kv_len <= self.budget:
            return
        obs = min(self.window_size, kv_len)
        n_cand = kv_len - obs
        n_select = self.budget - obs
        if n_cand <= 0 or n_select <= 0 or len(history) < obs:
            return

        evict_start = time.perf_counter()
        rows = []
        for item in history[-obs:]:
            cand = item[:, :n_cand]
            rows.append(cand / cand.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        importance = torch.stack(rows, dim=1).mean(1)
        importance = _pool_importance(importance, self.kernel_size, self.pooling)
        selected = torch.topk(importance, k=n_select, dim=-1).indices
        recent = torch.arange(n_cand, kv_len, device=key_states.device).expand(kv_heads, -1)
        keep = torch.cat([selected, recent], dim=-1)

        head_dim = int(key_states.shape[-1])
        gather_idx = keep.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, head_dim)
        cache.key_cache[layer_idx] = key_states.gather(2, gather_idx).contiguous()
        cache.value_cache[layer_idx] = value_states.gather(2, gather_idx).contiguous()
        self.histories[layer_idx] = []
        self.stats.cache_len = self.budget
        self.stats.evictions += 1
        self.stats.layer_evictions[layer_idx] = self.stats.layer_evictions.get(layer_idx, 0) + 1
        self.stats.eviction_sec_total += time.perf_counter() - evict_start


def _pool_importance(values: torch.Tensor, kernel_size: int, pooling: str) -> torch.Tensor:
    if kernel_size <= 1 or values.numel() == 0:
        return values
    mode = pooling.lower()
    if mode == "avgpool":
        mode = "avg"
    elif mode == "maxpool":
        mode = "max"
    padded = values.unsqueeze(1)
    if mode == "avg":
        pooled = torch.nn.functional.avg_pool1d(
            padded, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(1)
    elif mode == "max":
        pooled = torch.nn.functional.max_pool1d(
            padded, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(1)
    else:
        raise ValueError(f"unsupported SnapKV pooling {pooling!r}")
    return pooled[:, : values.shape[-1]]


def install_qwen2_snapkv(model, controller: QwenSnapKVController) -> None:
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    except Exception as exc:  # pragma: no cover - depends on optional transformers model
        raise RuntimeError("Qwen2 SnapKV patch requires transformers.models.qwen2") from exc

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Qwen2 SnapKV patch requires attention modules with layer_idx")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if past_key_value is not None:
            self._snapkv_controller.observe_and_compress(self, attn_weights, past_key_value)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise RuntimeError("Qwen2 SnapKV patch expected model.model.layers")
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn._snapkv_controller = controller
        attn.forward = MethodType(forward, attn)
