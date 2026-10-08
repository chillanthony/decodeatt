"""Cache ownership boundary for reference and serving backends.

The reference implementation still uses dense HF-style tensors.  Serving
implementations can replace this class with a paged/block cache manager while
keeping policy code independent of cache storage.
"""
from __future__ import annotations

from typing import Any, Protocol

import torch


def _is_headwise(indices) -> bool:
    return isinstance(indices, list)


def _representative_indices(indices):
    if _is_headwise(indices):
        return indices[0][0]
    return indices


def _representative_indices_batch(indices):
    if _is_headwise(indices):
        layer_indices = indices[0]
        if layer_indices.ndim == 2:
            return layer_indices[0].unsqueeze(0)
        return layer_indices[:, 0, :]
    if indices.ndim == 1:
        return indices.unsqueeze(0)
    return indices


class CacheManager(Protocol):
    """Storage operations needed by the token reference runner."""

    def compact(self, cache, indices):
        ...

    def compact_batch(self, cache, indices):
        ...


class TransformersCacheManager:
    """Dense tensor cache manager for tuple and HF Cache objects."""

    @staticmethod
    def _view(cache):
        if cache is None or hasattr(cache, "layers"):
            return cache
        if isinstance(cache, tuple):
            return _CacheView([_CacheLayerView(keys, values) for keys, values in cache])
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return _CacheView([
                _CacheLayerView(keys, values)
                for keys, values in zip(cache.key_cache, cache.value_cache)
            ])
        return cache

    @staticmethod
    def _compact(cache, indices):
        if _is_headwise(indices):
            for layer, layer_indices in zip(cache.layers, indices):
                layer_indices = layer_indices.to(layer.keys.device)
                head_dim = layer.keys.shape[-1]
                gather_indices = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                    1, -1, -1, head_dim
                )
                layer.keys = layer.keys.gather(2, gather_indices).contiguous()
                layer.values = layer.values.gather(2, gather_indices).contiguous()
            return int(indices[0].shape[-1])

        for layer in cache.layers:
            local_indices = indices.to(layer.keys.device)
            layer.keys = layer.keys.index_select(2, local_indices).contiguous()
            layer.values = layer.values.index_select(2, local_indices).contiguous()
        return int(indices.numel())

    def compact(self, cache, indices):
        if hasattr(cache, "layers"):
            return cache, self._compact(cache, indices)

        if isinstance(cache, tuple):
            compacted = []
            if _is_headwise(indices):
                for (keys, values), layer_indices in zip(cache, indices):
                    layer_indices = layer_indices.to(keys.device)
                    head_dim = keys.shape[-1]
                    gather_indices = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                        1, -1, -1, head_dim
                    )
                    compacted.append((
                        keys.gather(2, gather_indices).contiguous(),
                        values.gather(2, gather_indices).contiguous(),
                    ))
                return tuple(compacted), int(indices[0].shape[-1])

            for keys, values in cache:
                local_indices = indices.to(keys.device)
                compacted.append((
                    keys.index_select(2, local_indices).contiguous(),
                    values.index_select(2, local_indices).contiguous(),
                ))
            return tuple(compacted), int(indices.numel())

        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            if _is_headwise(indices):
                for layer_idx, layer_indices in enumerate(indices):
                    keys = cache.key_cache[layer_idx]
                    values = cache.value_cache[layer_idx]
                    layer_indices = layer_indices.to(keys.device)
                    head_dim = keys.shape[-1]
                    gather_indices = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                        1, -1, -1, head_dim
                    )
                    cache.key_cache[layer_idx] = keys.gather(2, gather_indices).contiguous()
                    cache.value_cache[layer_idx] = values.gather(2, gather_indices).contiguous()
                return cache, int(indices[0].shape[-1])

            for layer_idx, keys in enumerate(cache.key_cache):
                local_indices = indices.to(keys.device)
                cache.key_cache[layer_idx] = keys.index_select(2, local_indices).contiguous()
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx].index_select(
                    2, local_indices
                ).contiguous()
            return cache, int(indices.numel())

        return cache, self._compact(cache, indices)

    def compact_batch(self, cache, indices):
        cache_view = self._view(cache)
        batch_size = cache_view.layers[0].keys.shape[0]
        tuple_layers = [] if isinstance(cache, tuple) else None
        for layer_idx, layer in enumerate(cache_view.layers):
            keys = layer.keys
            values = layer.values
            if _is_headwise(indices):
                layer_indices = indices[layer_idx]
                if layer_indices.ndim == 2:
                    layer_indices = layer_indices.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                layer_indices = indices
                if layer_indices.ndim == 1:
                    layer_indices = layer_indices.unsqueeze(0).expand(batch_size, -1)
                layer_indices = layer_indices.unsqueeze(1).expand(-1, keys.shape[1], -1)
            gather_indices = layer_indices.to(keys.device).unsqueeze(-1).expand(
                -1, -1, -1, keys.shape[-1]
            )
            compact_keys = keys.gather(2, gather_indices).contiguous()
            compact_values = values.gather(2, gather_indices).contiguous()
            if hasattr(cache, "layers"):
                cache.layers[layer_idx].keys = compact_keys
                cache.layers[layer_idx].values = compact_values
            elif isinstance(cache, tuple):
                tuple_layers.append((compact_keys, compact_values))
            else:
                cache.key_cache[layer_idx] = compact_keys
                cache.value_cache[layer_idx] = compact_values
        if tuple_layers is not None:
            cache = tuple(tuple_layers)
        return cache, int(_representative_indices_batch(indices).shape[-1])


class _CacheLayerView:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _CacheView:
    def __init__(self, layers: list[_CacheLayerView]):
        self.layers = layers


__all__ = ["CacheManager", "TransformersCacheManager"]
