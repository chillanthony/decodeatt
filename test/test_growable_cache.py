"""GrowableCache must hold exactly what DynamicCache holds through appends and compactions."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.cache_utils import DynamicCache  # noqa: E402

from kveviction.cache_manager import TransformersCacheManager  # noqa: E402
from kveviction.growable_cache import GrowableCache  # noqa: E402


def _step(cache, layer, k, v):
    return cache.update(k, v, layer)


@pytest.mark.parametrize("headwise", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_appends_and_compactions_match_dynamic_cache(headwise, batch):
    gen = torch.Generator().manual_seed(0)
    layers, heads, dim = 2, 2, 4
    ref, new = DynamicCache(), GrowableCache()
    manager = TransformersCacheManager()
    length = 0
    for step in range(300):
        q = 7 if step == 0 else 1  # prefill then single-token decode
        for layer in range(layers):
            k = torch.randn(batch, heads, q, dim, generator=gen)
            v = torch.randn(batch, heads, q, dim, generator=gen)
            rk, rv = _step(ref, layer, k, v)
            nk, nv = _step(new, layer, k, v)
            assert torch.equal(rk, nk) and torch.equal(rv, nv)
        length += q
        assert new.get_seq_length() == ref.get_seq_length() == length
        if step % 37 == 36:
            keep = length // 2
            if headwise:
                idx = [
                    torch.stack([torch.randperm(length, generator=gen)[:keep].sort().values for _ in range(heads)])
                    for _ in range(layers)
                ]
            else:
                idx = torch.randperm(length, generator=gen)[:keep].sort().values
            if batch == 1:
                ref, _ = manager.compact(ref, idx)
                new, length = manager.compact(new, idx)
            else:
                ref, _ = manager.compact_batch(ref, idx)
                new, length = manager.compact_batch(new, idx)
            for a, b in zip(ref.layers, new.layers):
                assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    # Amortized growth: capacity stays within ~12.5% + one chunk of the live length peak.
    assert all(layer.capacity <= 2 * 300 + 256 for layer in new.layers)


def test_growth_is_amortized_and_bounded():
    cache = GrowableCache()
    reallocs, last = 0, None
    for _ in range(5000):
        cache.update(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), 0)
        ptr = cache.layers[0]._kbuf.data_ptr()
        reallocs += ptr != last
        last = ptr
    layer = cache.layers[0]
    assert layer.get_seq_length() == 5000
    assert reallocs < 40                      # vs 5000 for torch.cat per step
    assert layer.capacity <= 5000 * 1.125 + 256 + 1
