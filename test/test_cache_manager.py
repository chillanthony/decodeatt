import torch

from kveviction.cache_manager import (
    PagedBlockCacheManager,
    PagedCacheState,
    TransformersCacheManager,
    VLLMPagedBlockCacheManager,
)


def _cache(batch_size=1, heads=2, length=5, dim=3):
    return tuple((
        torch.arange(batch_size * heads * length * dim).reshape(
            batch_size, heads, length, dim
        ),
        torch.arange(batch_size * heads * length * dim).reshape(
            batch_size, heads, length, dim
        ) + 1000,
    ) for _ in range(2))


def test_dense_tuple_cache_compaction():
    manager = TransformersCacheManager()
    cache, length = manager.compact(_cache(), torch.tensor([4, 1, 3]))

    assert length == 3
    assert cache[0][0].shape == (1, 2, 3, 3)
    assert cache[0][0][0, 0, :, 0].tolist() == [12, 3, 9]
    assert cache[0][1][0, 0, :, 0].tolist() == [1012, 1003, 1009]


def test_batched_dense_tuple_cache_compaction():
    manager = TransformersCacheManager()
    cache, length = manager.compact_batch(
        _cache(batch_size=2), torch.tensor([[4, 1, 3], [2, 0, 4]])
    )

    assert length == 3
    assert cache[0][0].shape == (2, 2, 3, 3)
    assert cache[0][0][1, 0, :, 0].tolist() == [36, 30, 42]


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
