"""Cache ownership boundary for reference and serving backends.

The reference implementation uses dense HF ``Cache`` tensors.  Serving
implementations can replace this class with a paged/block cache manager while
keeping policy code independent of cache storage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

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

    def metadata(self, cache):
        ...


class TransformersCacheManager:
    """Dense HF cache manager (transformers >= 5 ``Cache`` objects with ``.layers``).

    Compaction gathers the kept slots per layer and assigns them back to
    ``layer.keys`` / ``layer.values``; ``GrowableLayer`` turns that assignment
    into an in-place write of its buffer prefix.
    """

    @staticmethod
    def _view(cache):
        if cache is not None and not hasattr(cache, "layers"):
            raise TypeError(
                f"unsupported KV cache {type(cache).__name__}; expected a transformers Cache with .layers"
            )
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
        return cache, self._compact(self._view(cache), indices)

    def compact_batch(self, cache, indices):
        cache = self._view(cache)
        batch_size = cache.layers[0].keys.shape[0]
        for layer_idx, layer in enumerate(cache.layers):
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
            layer.keys = keys.gather(2, gather_indices).contiguous()
            layer.values = values.gather(2, gather_indices).contiguous()
        return cache, int(_representative_indices_batch(indices).shape[-1])

    def metadata(self, cache):
        if cache is None or not hasattr(cache, "layers"):
            return {"kind": "unknown"}
        return {
            "kind": "dense",
            "layers": len(cache.layers),
            "batch_size": int(cache.layers[0].keys.shape[0]),
            "heads": int(cache.layers[0].keys.shape[1]),
            "length": int(cache.layers[0].keys.shape[2]),
        }


@dataclass
class PagedCacheState:
    """Backend-neutral paged cache state used by serving adapters."""

    block_ids: list[int]
    valid_tokens: list[int]
    block_size: int
    logical_positions: list[int] = field(default_factory=list)


class PagedBlockCacheManager:
    """Storage-neutral paged cache contract for serving backends.

    This adapter deliberately does not copy dense tensors.  A vLLM adapter can
    implement the same operations using its block allocator and token mapping.
    The reference implementation stores ownership metadata and validates that
    selections are block aligned, making unsupported token-level compaction
    explicit instead of silently falling back to dense gather.
    """

    def __init__(self, block_size: int = 16):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)

    def metadata(self, cache):
        return {
            "kind": "paged",
            "block_size": self.block_size,
            "blocks": list(getattr(cache, "block_ids", [])),
            "valid_tokens": list(getattr(cache, "valid_tokens", [])),
        }

    def compact(self, cache: PagedCacheState, indices):
        """Retain whole blocks and return a logical token-to-block mapping.

        The manager never gathers dense KV tensors.  A serving adapter can use
        the returned block ids to update its allocator asynchronously.
        """
        if not isinstance(cache, PagedCacheState):
            raise TypeError("paged cache requires PagedCacheState")
        flat = [int(i) for i in indices.detach().cpu().reshape(-1).tolist()]
        keep_blocks = []
        for token in flat:
            block = token // cache.block_size
            if block < 0 or block >= len(cache.block_ids):
                raise IndexError(f"token index {token} is outside paged cache")
            if block not in keep_blocks:
                keep_blocks.append(block)
        old_ids = cache.block_ids
        old_valid = cache.valid_tokens
        cache.block_ids = [old_ids[i] for i in keep_blocks]
        cache.valid_tokens = [old_valid[i] for i in keep_blocks]
        cache.logical_positions = flat
        return cache, len(flat)

    def compact_batch(self, cache, indices):
        if not isinstance(cache, (list, tuple)):
            raise TypeError("paged batch cache requires a list of PagedCacheState")
        if len(cache) != len(indices):
            raise ValueError("cache and selection batch sizes differ")
        result = [self.compact(item, row) for item, row in zip(cache, indices)]
        return [item for item, _ in result], max((length for _, length in result), default=0)


class VLLMPagedBlockCacheManager(PagedBlockCacheManager):
    """Adapter around a vLLM worker's block operations.

    vLLM versions expose different worker internals, so callbacks are injected
    by the engine integration.  This keeps policy code independent of those
    private APIs while making allocation, release, and compaction executable.
    """

    def __init__(self, block_size: int, *, retain_blocks: Callable | None = None,
                 release_blocks: Callable | None = None):
        super().__init__(block_size)
        self.retain_blocks = retain_blocks
        self.release_blocks = release_blocks

    def compact(self, cache, indices):
        previous = list(cache.block_ids)
        cache, length = super().compact(cache, indices)
        retained = set(cache.block_ids)
        released = [block for block in previous if block not in retained]
        if self.retain_blocks:
            self.retain_blocks(cache.block_ids, cache.valid_tokens)
        if released and self.release_blocks:
            self.release_blocks(released)
        return cache, length


__all__ = [
    "CacheManager", "TransformersCacheManager", "PagedCacheState",
    "PagedBlockCacheManager", "VLLMPagedBlockCacheManager",
]
