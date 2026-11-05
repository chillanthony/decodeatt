"""Distributed scheduling helpers."""
from __future__ import annotations

from kvbench.sharding import allocate_group_ranks as _allocate_group_ranks
from kvbench.sharding import claim_next_problem as _claim_next_problem


def test_distributed_rank_allocation_balances_attention_groups():
    groups = {
        "sdpa": ["fullkv", "streamingllm"],
        "eager": ["snapkv", "h2o", "rkv"],
    }
    assert _allocate_group_ranks(groups, 8) == [
        ("sdpa", 0, 3),
        ("sdpa", 1, 3),
        ("sdpa", 2, 3),
        ("eager", 0, 5),
        ("eager", 1, 5),
        ("eager", 2, 5),
        ("eager", 3, 5),
        ("eager", 4, 5),
    ]


def test_distributed_problem_claims_are_atomic_counter_indices():
    class _Store:
        value = 0

        def add(self, _key, increment):
            self.value += increment
            return self.value

    store = _Store()
    assert [_claim_next_problem(store, "queue") for _ in range(4)] == [0, 1, 2, 3]
