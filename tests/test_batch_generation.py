"""CPU checks for same-prompt candidate batching primitives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from kvbench.models import pad_input_ids
from kvbench.evaluator import _run_cross_problem_eval
from kvbench.policies import parse_arm
from kv_eviction.runner_token import _compact_cache_update_batch, generate_token_evict_batch
from kv_eviction.strategies.h2o import select_h2o
from kv_eviction.strategies.rkv_official import select_rkv_layers
from kv_eviction.strategies.snapkv import select_snapkv


@dataclass
class _Layer:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class _Cache:
    layers: list[_Layer]


def _case(batch_size: int = 3):
    torch.manual_seed(11)
    n = 14
    heads = 2
    dim = 6
    alpha = 4
    keys = torch.randn(batch_size, heads, n, dim)
    values = torch.randn_like(keys)
    cache = _Cache([_Layer(keys.clone(), values.clone())])
    history = [
        [torch.randn(batch_size, heads, n)]
        for _ in range(alpha)
    ]
    params = {
        "alpha": alpha,
        "lambda": 0.1,
        "pool_kernel": 7,
        "similarity_threshold": 0.5,
        "retain_ratio": 0.1,
        "retain_direction": "last",
        "redundancy_chunk": 3,
    }
    return cache, history, n, 9, params


def test_batched_rkv_selection_matches_independent_requests():
    cache, history, n, budget, params = _case()
    actual = select_rkv_layers(cache, history, n, budget, params)
    assert actual[0].shape == (3, 2, budget)

    for batch_idx in range(3):
        single_cache = _Cache([
            _Layer(
                cache.layers[0].keys[batch_idx:batch_idx + 1].clone(),
                cache.layers[0].values[batch_idx:batch_idx + 1].clone(),
            )
        ])
        single_history = [
            [step[0][batch_idx].clone()]
            for step in history
        ]
        expected = select_rkv_layers(single_cache, single_history, n, budget, params)
        torch.testing.assert_close(actual[0][batch_idx], expected[0])


def test_batched_attention_baselines_match_independent_requests():
    cache, history, n, budget, _ = _case()
    cases = [
        (select_snapkv, {"window_size": 4, "kernel_size": 3, "pooling": "avgpool"}),
        (select_h2o, {}),
    ]
    for selector, params in cases:
        actual = selector(cache, history, n, budget, params)
        assert actual[0].shape == (3, 2, budget)
        for batch_idx in range(3):
            single_cache = _Cache([
                _Layer(
                    cache.layers[0].keys[batch_idx:batch_idx + 1].clone(),
                    cache.layers[0].values[batch_idx:batch_idx + 1].clone(),
                )
            ])
            single_history = [
                [step[0][batch_idx].clone()]
                for step in history
            ]
            expected = selector(single_cache, single_history, n, budget, params)
            torch.testing.assert_close(actual[0][batch_idx], expected[0])


def test_batched_cache_compaction_uses_per_request_per_head_indices():
    cache, _, _, _, _ = _case(batch_size=2)
    original_keys = cache.layers[0].keys.clone()
    original_values = cache.layers[0].values.clone()
    indices = [torch.tensor([
        [[0, 2, 4], [1, 3, 5]],
        [[6, 8, 10], [7, 9, 11]],
    ])]

    compacted, length = _compact_cache_update_batch(cache, indices)
    assert length == 3
    for batch_idx in range(2):
        for head_idx in range(2):
            expected_keys = original_keys[batch_idx, head_idx, indices[0][batch_idx, head_idx]]
            expected_values = original_values[batch_idx, head_idx, indices[0][batch_idx, head_idx]]
            torch.testing.assert_close(
                compacted.layers[0].keys[batch_idx, head_idx], expected_keys
            )
            torch.testing.assert_close(
                compacted.layers[0].values[batch_idx, head_idx], expected_values
            )


def test_left_padding_helper_returns_mask_and_lengths():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

    input_ids, mask, lengths = pad_input_ids(
        _Tokenizer(), [torch.tensor([[4, 5, 6]]), torch.tensor([[7, 8]])], torch.device("cpu")
    )
    torch.testing.assert_close(input_ids, torch.tensor([[4, 5, 6], [0, 7, 8]]))
    torch.testing.assert_close(mask, torch.tensor([[1, 1, 1], [0, 1, 1]]))
    torch.testing.assert_close(lengths, torch.tensor([3, 2]))


def test_masked_batched_rkv_matches_unpadded_requests():
    torch.manual_seed(17)
    batch_size, heads, padded_n, short_n, dim = 2, 2, 14, 10, 6
    pad = padded_n - short_n
    keys = torch.randn(batch_size, heads, padded_n, dim)
    values = torch.randn_like(keys)
    valid = torch.ones(batch_size, padded_n, dtype=torch.bool)
    valid[1, :pad] = False
    history = [[torch.randn(batch_size, heads, padded_n)] for _ in range(4)]
    params = {
        "alpha": 4,
        "lambda": 0.1,
        "pool_kernel": 3,
        "similarity_threshold": 0.5,
        "retain_ratio": 0.1,
        "retain_direction": "last",
        "redundancy_chunk": 3,
        "_valid_mask": valid,
    }
    cache = _Cache([_Layer(keys.clone(), values.clone())])
    actual = select_rkv_layers(cache, history, padded_n, 9, params)[0]
    assert valid[0, actual[0]].all()
    assert valid[1, actual[1]].all()

    for batch_idx, start in enumerate([0, pad]):
        logical_n = padded_n - start
        single_cache = _Cache([_Layer(
            keys[batch_idx:batch_idx + 1, :, start:].clone(),
            values[batch_idx:batch_idx + 1, :, start:].clone(),
        )])
        single_history = [[step[0][batch_idx, :, start:].clone()] for step in history]
        single_params = {key: value for key, value in params.items() if key != "_valid_mask"}
        expected = select_rkv_layers(single_cache, single_history, logical_n, 9, single_params)[0]
        torch.testing.assert_close(actual[batch_idx] - start, expected)


def test_heterogeneous_batch_masks_finished_requests():
    class _Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            visible = [token for token in ids if not skip_special_tokens or token != self.eos_token_id]
            return " ".join(map(str, visible))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            logits = torch.full((batch, length, 5), -100.0)
            if self.calls == 0:
                logits[0, -1, 2] = 100.0
                logits[1, -1, 3] = 100.0
            else:
                logits[:, -1, 2] = 100.0
            self.calls += 1
            return SimpleNamespace(logits=logits, past_key_values=None)

    result = generate_token_evict_batch(
        _Model(),
        _Tokenizer(),
        torch.tensor([[0, 4, 5], [6, 7, 8]]),
        input_attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        batch_size=2,
        backend="fullkv",
        budget=32,
        max_new=4,
        do_sample=False,
        seeds=[0, 0],
    )
    assert result["candidates"][0]["gen_ids"] == [2]
    assert result["candidates"][1]["gen_ids"] == [3, 2]
    assert [item["prompt_len"] for item in result["candidates"]] == [2, 3]


def test_batched_generation_reports_eviction_metrics():
    class _Tokenizer:
        eos_token_id = None
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, past_key_values=None, **_kwargs):
            batch, length = input_ids.shape
            if past_key_values is None:
                keys = torch.zeros(batch, 1, length, 1)
                values = torch.zeros_like(keys)
                past_key_values = _Cache([_Layer(keys, values)])
            else:
                layer = past_key_values.layers[0]
                added = torch.zeros(batch, 1, length, 1)
                layer.keys = torch.cat([layer.keys, added], dim=2)
                layer.values = torch.cat([layer.values, added], dim=2)
            logits = torch.zeros(batch, length, 4)
            logits[:, -1, 1] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    result = generate_token_evict_batch(
        _Model(),
        _Tokenizer(),
        torch.tensor([[3, 3, 3]]),
        batch_size=2,
        backend="window",
        budget=4,
        recent=1,
        sink=1,
        evict_every=2,
        obs_window=0,
        max_new=8,
        do_sample=False,
        seeds=[0, 1],
    )
    for candidate in result["candidates"]:
        assert candidate["n_evict"] > 0
        assert candidate["total_evicted_tokens"] > 0
        assert candidate["total_effective_evicted_tokens"] > 0
        assert candidate["mean_effective_cache_len"] > 0
        assert len(candidate["effective_cache_len_curve"]) == candidate["n_evict"]


def test_cross_problem_evaluator_restores_order_and_candidates():
    class _Tokenizer:
        chat_template = None
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, question, return_tensors="pt"):
            length = 2 + len(question)
            return SimpleNamespace(input_ids=torch.arange(3, 3 + length).unsqueeze(0))

        def decode(self, ids, skip_special_tokens=True):
            return ""

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, **_kwargs):
            logits = torch.full((*input_ids.shape, 8), -100.0)
            logits[:, -1, 2] = 100.0
            return SimpleNamespace(logits=logits, past_key_values=None)

    problems = [
        (1, 3, {"id": "short", "question": "a", "answer": "0"}),
        (2, 3, {"id": "long", "question": "abcdefgh", "answer": "0"}),
        (3, 3, {"id": "medium", "question": "abcd", "answer": "0"}),
    ]
    with TemporaryDirectory() as tmp_dir:
        payload = _run_cross_problem_eval(
            model=_Model(),
            tokenizer=_Tokenizer(),
            problem_iter=iter(problems),
            arms=[parse_arm("fullkv")],
            out_path=Path(tmp_dir) / "result.json",
            progress_prefix="",
            problem_batch_size=2,
            prompt_bucket_size=3,
            num_return_sequences=3,
            candidate_batch_size=2,
            seed=10,
            seed_offset=5,
            max_new=2,
            recent=0,
            sink=0,
            evict_every=4,
            obs_window=0,
            obs_decay=0.9,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            policy_params={},
        )
    assert [record["id"] for record in payload["records"]] == ["short", "long", "medium"]
    for record in payload["records"]:
        candidates = record["arms"]["fullkv"]["candidates"]
        assert len(candidates) == 3
        assert [candidate["seed"] for candidate in candidates] == [15, 16, 17]


if __name__ == "__main__":
    test_batched_rkv_selection_matches_independent_requests()
    test_batched_attention_baselines_match_independent_requests()
    test_batched_cache_compaction_uses_per_request_per_head_indices()
    test_left_padding_helper_returns_mask_and_lengths()
    test_masked_batched_rkv_matches_unpadded_requests()
    test_heterogeneous_batch_masks_finished_requests()
    test_cross_problem_evaluator_restores_order_and_candidates()
    print("Batch generation primitive tests passed")
