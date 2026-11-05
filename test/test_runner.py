"""End-to-end decoding on a tiny random Llama (CPU): budgets, batching, anchors, traces."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kvbench.models import pad_input_ids
from kvbench.result_schema import CANDIDATE_KEYS
from kveviction.attention_logits import require_eager_attention
from kveviction.eviction_stats import token_samples
from kveviction.runner_token import generate_token_evict, generate_token_evict_batch
from kveviction.strategies.token import get_policy, policy_names


class _Tokenizer:
    eos_token_id = None
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


_E2E = dict(budget=16, recent=4, sink=2, evict_every=4, max_new=40, do_sample=False)


@pytest.mark.parametrize("backend", sorted(policy_names()))
def test_end_to_end_generation_respects_budget(tiny_llama, backend):
    model, prompt = tiny_llama
    result = generate_token_evict(model, _Tokenizer(), prompt, backend=backend, **_E2E)
    assert len(result["gen_ids"]) == _E2E["max_new"]
    if backend == "fullkv":
        assert result["n_evict"] == 0
        assert result["final_cache_len"] == prompt.shape[1] + _E2E["max_new"]
        return
    assert result["n_evict"] > 0
    assert len(result["evict_events"]) == result["n_evict"]
    for event in result["evict_events"]:
        assert event["cache_len_after"] == _E2E["budget"]
        assert event["cache_len_before"] > event["cache_len_after"]
        assert event["step"] % _E2E["evict_every"] == 0
    assert result["final_cache_len"] <= _E2E["budget"] + _E2E["evict_every"]
    assert result["cache_len_curve"] == [_E2E["budget"]] * result["n_evict"]


@pytest.mark.parametrize("do_sample", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize("backend", sorted(policy_names()))
def test_batch_matches_single_request(tiny_llama, backend, do_sample):
    """The single and batched loops are separate code paths; keep them in sync.

    A batch row seeded with ``s`` must reproduce a single request seeded with
    ``s``, including sampling and the random policy's slot choices.
    """
    model, prompt = tiny_llama
    cfg = {**_E2E, "do_sample": do_sample}
    single = generate_token_evict(model, _Tokenizer(), prompt, backend=backend, seed=3, **cfg)
    batch = generate_token_evict_batch(
        model, _Tokenizer(), prompt, 2, backend=backend, seeds=[3, 3], **cfg
    )
    for candidate in batch["candidates"]:
        assert candidate["gen_ids"] == single["gen_ids"]
        assert candidate["n_evict"] == single["n_evict"]
        assert candidate["cache_len_curve"] == single["cache_len_curve"]
        assert candidate["total_evicted_tokens"] == single["total_evicted_tokens"]
        assert set(candidate) == set(CANDIDATE_KEYS)
    assert set(single) == set(CANDIDATE_KEYS)
    assert single["batch_size"] == 1 and batch["candidates"][0]["batch_size"] == 2


@pytest.mark.parametrize("mode", ["anchor", "lowent", "random", "sig"])
def test_batch_matches_single_request_with_anchors(tiny_llama, mode):
    model, prompt = tiny_llama
    sig = ([1.0, -0.5, 0.3, 0.2], 0.1) if mode == "sig" else None
    cfg = {**_E2E, "do_sample": True, "anchor_mode": mode, "anchor_frac": 0.2, "sig": sig}
    single = generate_token_evict(model, _Tokenizer(), prompt, backend="window", seed=5, **cfg)
    batch = generate_token_evict_batch(
        model, _Tokenizer(), prompt, 2, backend="window", seeds=[5, 5], **cfg
    )
    # anchor_k = 3 extra slots on top of the budget.
    assert single["cache_len_curve"] == [_E2E["budget"] + 3] * single["n_evict"]
    for candidate in batch["candidates"]:
        assert candidate["gen_ids"] == single["gen_ids"]
        assert candidate["cache_len_curve"] == single["cache_len_curve"]


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


@pytest.mark.parametrize("backend, anchor_mode", [("window", "anchor"), ("rkv", "none"), ("snapkv", "none")])
def test_teacher_forcing_own_generation_replays_the_same_evictions(tiny_llama, backend, anchor_mode):
    """score_trace_token shares the decode loop, so replaying a greedy generation
    must reproduce its eviction schedule and final cache exactly."""
    from kveviction.trace_scoring import score_trace_token

    model, prompt = tiny_llama
    kw = dict(backend=backend, anchor_mode=anchor_mode, budget=16, recent=4, sink=2, evict_every=4)
    gen = generate_token_evict(model, _Tokenizer(), prompt, max_new=40, do_sample=False, **kw)
    full = torch.cat([prompt, torch.tensor([gen["gen_ids"]])], dim=1)
    trace = score_trace_token(model, _Tokenizer(), full, prompt.shape[1], [3, 15], span_len=4,
                              debug=True, **kw)
    assert trace["final_cache_len"] == gen["final_cache_len"]
    # Trace debug events skip no-op compactions (nothing evicted).
    expected = [(e["step"], e["cache_len_before"], e["cache_len_after"])
                for e in gen["evict_events"] if e["evicted"] > 0]
    got = [(e["eviction_step"], e["cache_len_before"], e["cache_len_after"])
           for e in trace["eviction_events"]]
    assert got == expected and expected
    assert (trace["n_corr"], trace["n_noncorr"]) == (8, len(gen["gen_ids"]) - 8)
    assert trace["per_token_nll"].shape == (len(gen["gen_ids"]),)


def test_debug_dump_is_rejected_for_batches(tiny_llama, tmp_path):
    from kveviction.runner_token import _decode

    model, prompt = tiny_llama
    with pytest.raises(ValueError, match="debug dumps"):
        _decode(
            model, _Tokenizer(), prompt, batch_size=2, solo=False, budget=16, recent=4,
            sink=2, backend="window", anchor_mode="none", anchor_frac=0.0, evict_every=4,
            obs_window=8, obs_decay=0.9, max_new=4, do_sample=False, temperature=1.0,
            top_p=1.0, seeds=[0, 0], sig=None, policy_params=None,
            input_attention_mask=None, debug_path=str(tmp_path / "d.json"), debug_topk=0,
            execution_backend=None, cache_manager=None,
        )


def test_window_on_sdpa_fails_fast():
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    with pytest.raises(ValueError, match="eager"):
        require_eager_attention(model, get_policy("window"), use_attention_logits=False)
    require_eager_attention(model, get_policy("rkv"), use_attention_logits=True)
    require_eager_attention(model, get_policy("streamingllm"), use_attention_logits=False)
    require_eager_attention(model, get_policy("window"), use_attention_logits=False, track=False)


def test_debug_token_samples_can_be_disabled():
    class _Tokenizer:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("decode should not be called when limit is 0")

    samples = token_samples(
        _Tokenizer(),
        torch.tensor([11, 12, 13]),
        torch.tensor([0, 1, 2]),
        limit=0,
    )
    assert samples == []
