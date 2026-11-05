"""HF KV cache with amortized growth and in-place compaction.

``transformers.DynamicLayer.update`` concatenates every new token onto the
whole cache, so each decode step reallocates and copies all of it: O(n) per
step and O(n^2) over a long generation.  That cost falls almost entirely on
FullKV (eviction arms stay near their budget), which biases HF throughput
comparisons in favour of eviction.

``GrowableLayer`` keeps keys/values in a preallocated buffer and exposes
``keys``/``values`` as views of its filled prefix.  Appending writes into the
spare capacity; assigning a tensor (as cache compaction does) copies it into
the buffer prefix.  The buffer grows geometrically (by 1/8 plus a fixed
chunk), so total copying is linear and the reserved memory exceeds the live
cache by at most ~12.5% + ``_CHUNK`` tokens.  Stored values are bit-identical
to ``DynamicLayer``'s, only their memory layout differs.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

_CHUNK = 256


def _grown_capacity(needed: int, current: int) -> int:
    return max(needed, current + current // 8 + _CHUNK)


class GrowableLayer(DynamicLayer):
    def __init__(self, **kwargs):
        # Keys and values keep separate lengths: compaction assigns ``keys``
        # before reading ``values``, which must still see the old length.
        self._kbuf = self._vbuf = None
        self._klen = self._vlen = 0
        super().__init__(**kwargs)

    # ``keys`` / ``values`` are views of the filled prefix of the buffers.
    @property
    def keys(self):
        return None if self._kbuf is None else self._kbuf[:, :, :self._klen]

    @keys.setter
    def keys(self, tensor):
        self._kbuf, self._klen = self._store(self._kbuf, tensor)

    @property
    def values(self):
        return None if self._vbuf is None else self._vbuf[:, :, :self._vlen]

    @values.setter
    def values(self, tensor):
        self._vbuf, self._vlen = self._store(self._vbuf, tensor)

    @staticmethod
    def _store(buf, tensor):
        """Copy ``tensor`` into the prefix of ``buf`` (reallocating if needed)."""
        if tensor is None or tensor.dim() != 4:
            return None, 0  # (re)initialization with an empty placeholder
        length = tensor.shape[2]
        if (
            buf is None
            or buf.shape[:2] != tensor.shape[:2]
            or buf.shape[3] != tensor.shape[3]
            or buf.dtype != tensor.dtype
            or buf.device != tensor.device
            or buf.shape[2] < length
        ):
            capacity = length if buf is None else _grown_capacity(length, buf.shape[2])
            new = torch.empty(
                (*tensor.shape[:2], capacity, tensor.shape[3]), dtype=tensor.dtype, device=tensor.device
            )
            new[:, :, :length].copy_(tensor)
            return new, length
        if tensor.data_ptr() != buf.data_ptr():
            buf[:, :, :length].copy_(tensor)
        return buf, length

    @staticmethod
    def _append(buf, length, states):
        step = states.shape[2]
        if buf is None:
            buf = torch.empty(
                (*states.shape[:2], step, states.shape[3]), dtype=states.dtype, device=states.device
            )
        elif buf.shape[2] < length + step:
            new = torch.empty(
                (*buf.shape[:2], _grown_capacity(length + step, buf.shape[2]), buf.shape[3]),
                dtype=buf.dtype, device=buf.device,
            )
            new[:, :, :length].copy_(buf[:, :, :length])
            buf = new
        buf[:, :, length:length + step].copy_(states)
        return buf, length + step

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self._kbuf, self._klen = self._append(self._kbuf, self._klen, key_states)
        self._vbuf, self._vlen = self._append(self._vbuf, self._vlen, value_states)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._klen if self.is_initialized else 0

    @property
    def capacity(self) -> int:
        return 0 if self._kbuf is None else int(self._kbuf.shape[2])


class GrowableCache(DynamicCache):
    """Drop-in ``DynamicCache`` replacement built from ``GrowableLayer``s."""

    def __init__(self):
        Cache.__init__(self, layer_class_to_replicate=GrowableLayer)


__all__ = ["GrowableCache", "GrowableLayer"]
