"""KV cache storage: dense compaction, GrowableCache and the paged-manager contract."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.cache_utils import DynamicCache  # noqa: E402

from kveviction.cache_manager import (  # noqa: E402
    PagedBlockCacheManager,
    PagedCacheState,
    TransformersCacheManager,
    VLLMPagedBlockCacheManager,
)
from kveviction.eviction_stats import cache_length_summary  # noqa: E402
from kveviction.growable_cache import GrowableCache  # noqa: E402


def _cache(batch_size=1, heads=2, length=5, dim=3, growable=False):
    if growable:
        from kveviction.growable_cache import GrowableCache
        cache = GrowableCache()
    else:
        cache = DynamicCache()
    keys = torch.arange(batch_size * heads * length * dim).reshape(batch_size, heads, length, dim)
    for layer in range(2):
        cache.update(keys, keys + 1000, layer)
    return cache


@pytest.mark.parametrize("growable", [False, True])
def test_dense_cache_compaction(growable):
    manager = TransformersCacheManager()
    cache = _cache(growable=growable)
    summary = cache_length_summary(cache, 5)
    assert summary["effective_kv_tokens_per_layer_head"] == [[5, 5], [5, 5]]
    cache, length = manager.compact(cache, torch.tensor([4, 1, 3]))

    assert length == 3
    assert cache.layers[0].keys.shape == (1, 2, 3, 3)
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [12, 3, 9]
    assert cache.layers[0].values[0, 0, :, 0].tolist() == [1012, 1003, 1009]


@pytest.mark.parametrize("growable", [False, True])
def test_headwise_compaction_gathers_each_head_independently(growable):
    manager = TransformersCacheManager()
    cache, length = manager.compact(_cache(growable=growable), [torch.tensor([[3, 1, 4], [0, 2, 4]])] * 2)
    assert length == 3
    # head 1 starts at 5 * 3 = 15 in the arange layout (length 5, dim 3)
    assert cache.layers[0].keys[0, :, :, 0].tolist() == [[9, 3, 12], [15, 21, 27]]
    assert cache.layers[0].values[0, 0, :, 0].tolist() == [1009, 1003, 1012]


@pytest.mark.parametrize("growable", [False, True])
def test_batched_headwise_compaction_uses_per_request_per_head_indices(growable):
    cache = _cache(batch_size=2, growable=growable)
    original = cache.layers[0].keys.clone()
    indices = torch.tensor([[[0, 2, 4], [1, 3, 4]], [[1, 2, 3], [0, 1, 4]]])
    cache, length = TransformersCacheManager().compact_batch(cache, [indices, indices])
    assert length == 3
    for b in range(2):
        for h in range(2):
            torch.testing.assert_close(cache.layers[0].keys[b, h], original[b, h, indices[b, h]])


@pytest.mark.parametrize("growable", [False, True])
def test_batched_dense_cache_compaction(growable):
    manager = TransformersCacheManager()
    cache, length = manager.compact_batch(
        _cache(batch_size=2, growable=growable), torch.tensor([[4, 1, 3], [2, 0, 4]])
    )

    assert length == 3
    assert cache.layers[0].keys.shape == (2, 2, 3, 3)
    assert cache.layers[0].keys[1, 0, :, 0].tolist() == [36, 30, 42]


def test_legacy_tuple_cache_is_rejected():
    keys = torch.zeros(1, 2, 5, 3)
    with pytest.raises(TypeError, match="transformers Cache"):
        TransformersCacheManager().compact(((keys, keys),), torch.tensor([0]))


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


def test_paged_manager_exposes_block_metadata_without_dense_fallback():
    manager = PagedBlockCacheManager(block_size=16)
    cache = PagedCacheState([3, 8], [16, 5], 16)
    assert manager.metadata(cache)["valid_tokens"] == [16, 5]
    cache, length = manager.compact(cache, torch.tensor([0, 17]))
    assert length == 2
    assert cache.block_ids == [3, 8]
    assert cache.logical_positions == [0, 17]


def test_vllm_adapter_releases_dropped_blocks():
    released = []
    manager = VLLMPagedBlockCacheManager(16, release_blocks=released.extend)
    cache = PagedCacheState([3, 8], [16, 5], 16)
    manager.compact(cache, torch.tensor([0]))
    assert released == [8]
