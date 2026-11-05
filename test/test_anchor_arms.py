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
