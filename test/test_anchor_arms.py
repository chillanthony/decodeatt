"""Anchor arms in the evaluation pipeline: parsing, validation and forwarding."""
from __future__ import annotations

import pytest
import torch

from kvbench.arms import parse_arms


class _Tokenizer:
    eos_token_id = None
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


@pytest.fixture(scope="module")
def tiny_llama():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    cfg._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(cfg).eval(), torch.randint(1, 64, (1, 12))


# --------------------------------------------------------------------------
# Anchor arms
# --------------------------------------------------------------------------

def test_anchor_arm_parsing_and_default_ids():
    arms = parse_arms([
        {"policy": "window", "budget": 512, "anchor_mode": "anchor", "anchor_frac": 0.05},
        {"policy": "window", "budget": 512, "anchor_mode": "lowent", "anchor_frac": 0.05},
        {"policy": "window", "budget": 512, "anchor_mode": "sig", "anchor_frac": 0.1,
         "sig": {"weights": [1, 0, 0, 0], "bias": 0}},
        {"policy": "window", "budget": 512},
    ])
    assert [arm.id for arm in arms] == [
        "window@512+anchor0.05", "window@512+lowent0.05", "window@512+sig0.1", "window@512",
    ]
    assert arms[2].sig == ((1.0, 0.0, 0.0, 0.0), 0.0)
    assert arms[3].anchor_frac == 0.0 and arms[3].sig is None


@pytest.mark.parametrize("item, message", [
    ({"policy": "window", "budget": 64, "anchor_mode": "anchor"}, "requires anchor_frac > 0"),
    ({"policy": "window", "budget": 64, "anchor_frac": 0.1}, "anchor_mode is 'none'"),
    ({"policy": "rkv", "budget": 64, "anchor_mode": "anchor", "anchor_frac": 0.1}, "slot-level"),
    ({"policy": "fullkv", "anchor_mode": "anchor", "anchor_frac": 0.1}, "slot-level"),
    ({"policy": "window", "budget": 64, "anchor_mode": "sig", "anchor_frac": 0.1}, "requires sig"),
    ({"policy": "random", "budget": 64, "anchor_mode": "sig", "anchor_frac": 0.1,
      "sig": {"weights": [1, 0, 0, 0], "bias": 0}}, "never observes"),
    ({"policy": "window", "budget": 64, "anchor_mode": "anchor", "anchor_frac": 0.1,
      "sig": {"weights": [1, 0, 0, 0], "bias": 0}}, "only used with anchor_mode 'sig'"),
])
def test_invalid_anchor_arms_are_rejected(item, message):
    with pytest.raises(ValueError, match=message):
        parse_arms([item])


def test_evaluator_forwards_anchor_settings(monkeypatch, tmp_path, tiny_llama):
    """Regression: the evaluator used to hard-code anchor_frac=0.0."""
    import kvbench.evaluator as evaluator
    from kveviction.runner_token import generate_token_evict

    model, prompt = tiny_llama
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return generate_token_evict(*args, **kwargs)

    monkeypatch.setattr(evaluator, "generate_token_evict", spy)
    monkeypatch.setattr(evaluator, "load_problems", lambda dataset, n: [
        {"id": "p0", "question": "q", "answer": "1"}
    ])
    monkeypatch.setattr(evaluator, "build_input_ids", lambda tokenizer, question, device: prompt)
    arm = parse_arms([{"policy": "window", "budget": 8, "anchor_mode": "sig", "anchor_frac": 0.25,
                       "sig": {"weights": [1, 0, 0, 0], "bias": 0}}])[0]
    payload = evaluator.run_generation_eval(
        model, _Tokenizer(), "sample", 1, [arm], tmp_path / "out.json",
        max_new=12, recent=2, sink=1, evict_every=4, obs_window=4, do_sample=False,
    )
    assert calls[0]["anchor_mode"] == "sig" and calls[0]["anchor_frac"] == 0.25
    assert calls[0]["sig"] == ((1.0, 0.0, 0.0, 0.0), 0.0)
    result = payload["records"][0]["arms"][arm.id]
    # anchor_k = 2 rescued slots on top of the budget at every compaction.
    assert result["n_evict"] > 0
    assert set(result["cache_len_curve"]) == {8 + 2}


@pytest.mark.parametrize("backend, anchor_mode", [("window", "anchor"), ("rkv", "none"), ("snapkv", "none")])
def test_teacher_forcing_own_generation_replays_the_same_evictions(tiny_llama, backend, anchor_mode):
    """score_trace_token shares the decode loop, so replaying a greedy generation
    must reproduce its eviction schedule and final cache exactly."""
    from kveviction.runner_token import generate_token_evict
    from kveviction.trace_scoring import score_trace_token

    model, prompt = tiny_llama
    kw = dict(backend=backend, anchor_mode=anchor_mode, budget=16, recent=4, sink=2, evict_every=4)
    gen = generate_token_evict(model, _Tokenizer(), prompt, max_new=40, do_sample=False, **kw)
    full = torch.cat([prompt, torch.tensor([gen["gen_ids"]])], dim=1)
    trace = score_trace_token(model, _Tokenizer(), full, prompt.shape[1], [], debug=True, **kw)
    assert trace["final_cache_len"] == gen["final_cache_len"]
    # Trace debug events skip no-op compactions (nothing evicted).
    expected = [(e["step"], e["cache_len_before"], e["cache_len_after"])
                for e in gen["evict_events"] if e["evicted"] > 0]
    got = [(e["eviction_step"], e["cache_len_before"], e["cache_len_after"])
           for e in trace["eviction_events"]]
    assert got == expected and expected
    assert trace["n_noncorr"] == len(gen["gen_ids"])


def test_only_entropy_anchors_request_entropy():
    from kveviction.strategies.anchors import with_anchors
    from kveviction.strategies.token import get_policy, policy_names

    assert not any(get_policy(name).needs_entropy for name in policy_names())
    window = get_policy("window")
    assert with_anchors(window, "anchor", 4).needs_entropy
    assert with_anchors(window, "lowent", 4).needs_entropy
    assert with_anchors(window, "sig", 4, sig=(torch.ones(4), 0.0)).needs_entropy
    assert not with_anchors(window, "random", 4).needs_entropy
