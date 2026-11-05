import pytest
import torch

from kveviction.cache_manager import (
    PagedBlockCacheManager,
    PagedCacheState,
    TransformersCacheManager,
    VLLMPagedBlockCacheManager,
)


def _cache(batch_size=1, heads=2, length=5, dim=3, growable=False):
    transformers = pytest.importorskip("transformers")
    if growable:
        from kveviction.growable_cache import GrowableCache
        cache = GrowableCache()
    else:
        cache = transformers.DynamicCache()
    keys = torch.arange(batch_size * heads * length * dim).reshape(batch_size, heads, length, dim)
    for layer in range(2):
        cache.update(keys, keys + 1000, layer)
    return cache


@pytest.mark.parametrize("growable", [False, True])
def test_dense_cache_compaction(growable):
    manager = TransformersCacheManager()
    cache, length = manager.compact(_cache(growable=growable), torch.tensor([4, 1, 3]))

    assert length == 3
    assert cache.layers[0].keys.shape == (1, 2, 3, 3)
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [12, 3, 9]
    assert cache.layers[0].values[0, 0, :, 0].tolist() == [1012, 1003, 1009]


def test_legacy_tuple_cache_is_rejected():
    keys = torch.zeros(1, 2, 5, 3)
    with pytest.raises(TypeError, match="transformers Cache"):
        TransformersCacheManager().compact(((keys, keys),), torch.tensor([0]))


@pytest.mark.parametrize("growable", [False, True])
def test_batched_dense_cache_compaction(growable):
    manager = TransformersCacheManager()
    cache, length = manager.compact_batch(
        _cache(batch_size=2, growable=growable), torch.tensor([[4, 1, 3], [2, 0, 4]])
    )

    assert length == 3
    assert cache.layers[0].keys.shape == (2, 2, 3, 3)
    assert cache.layers[0].keys[1, 0, :, 0].tolist() == [36, 30, 42]


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
