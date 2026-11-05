"""Canonical result schema shared by the HF and vLLM engines."""
from __future__ import annotations

import pytest
import torch

from kvbench.arms import parse_arms
from kvbench.result_schema import CANDIDATE_KEYS, missing_candidate_keys, no_eviction_candidate


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
# Result schema
# --------------------------------------------------------------------------

def test_hf_single_and_batched_results_use_the_canonical_keys(tiny_llama):
    from kveviction.runner_token import generate_token_evict, generate_token_evict_batch

    model, prompt = tiny_llama
    kw = dict(backend="window", budget=8, recent=2, sink=1, evict_every=4, max_new=12, do_sample=False)
    single = generate_token_evict(model, _Tokenizer(), prompt, **kw)
    batch = generate_token_evict_batch(model, _Tokenizer(), prompt, 2, **kw)["candidates"]
    for result in [single, *batch]:
        assert set(result) == set(CANDIDATE_KEYS)
    assert single["batch_size"] == 1 and batch[0]["batch_size"] == 2
    assert single["prompt_len"] == prompt.shape[1]


def test_no_eviction_candidate_matches_hf_fullkv(tiny_llama):
    """vLLM's FullKV builder must agree with the HF engine's FullKV fields."""
    from kveviction.runner_token import generate_token_evict

    model, prompt = tiny_llama
    hf = generate_token_evict(model, _Tokenizer(), prompt, backend="fullkv", max_new=6, do_sample=False)
    built = no_eviction_candidate(
        text=hf["text"], gen_ids=hf["gen_ids"], finished=hf["finished"], seed=0, prompt_len=prompt.shape[1],
        heads_per_layer=[2, 2], elapsed_sec=1.0, batch_elapsed_sec=1.0,
        batch_tokens_per_sec=6.0, batch_size=1,
    )
    assert set(built) == set(CANDIDATE_KEYS)
    timing_keys = {key for key in CANDIDATE_KEYS if "sec" in key} | {"peak_memory_bytes"}
    for key in sorted(set(CANDIDATE_KEYS) - timing_keys):
        if isinstance(hf[key], float):
            assert built[key] == pytest.approx(hf[key]), key
        else:
            assert built[key] == hf[key], key
    assert built["prefill_sec"] is None and built["eviction_sec_total"] == 0.0


def test_scoring_rejects_noncanonical_results():
    from kvbench.evaluator import _arm_payload

    with pytest.raises(KeyError, match="missing canonical fields"):
        _arm_payload([{"text": "", "gen_ids": []}], "1", 0, 0)
    assert "batch_size" in missing_candidate_keys({})


def test_vllm_and_hf_payloads_merge_into_one_summary(tiny_llama, tmp_path):
    from kvbench.evaluator import _arm_payload
    from kvbench.sharding import merge_payloads
    from kveviction.runner_token import generate_token_evict

    model, prompt = tiny_llama
    hf = generate_token_evict(model, _Tokenizer(), prompt, backend="fullkv", max_new=6, do_sample=False)
    vllm_like = no_eviction_candidate(
        text=hf["text"], gen_ids=hf["gen_ids"], finished=hf["finished"], seed=0, prompt_len=prompt.shape[1],
        heads_per_layer=[2, 2], elapsed_sec=0.5, batch_elapsed_sec=0.5,
        batch_tokens_per_sec=12.0, batch_size=4,
    )
    payloads = [
        {"records": [{"id": "p", "gold": "1", "arms": {name: _arm_payload([result], "1", 0, 0)}}]}
        for name, result in [("fullkv", hf), ("fullkv_vllm", vllm_like)]
    ]
    merged = merge_payloads(payloads, tmp_path / "merged.json")
    summary = merged["summary"]
    assert set(summary["fullkv"]) == set(summary["fullkv_vllm"])
    for key in ("mean_gen_len", "mean_final_cache_len", "mean_n_evict", "mean_compression_ratio",
                "mean_effective_cache_len", "mean_head_budget_entropy"):
        assert summary["fullkv"][key] == pytest.approx(summary["fullkv_vllm"][key]), key


def test_vllm_greedy_multi_candidate_uses_repeated_requests(monkeypatch, tmp_path):
    """vLLM rejects n > 1 with greedy decoding; candidates must still be canonical."""
    import sys
    import types

    import transformers

    import kvbench.vllm_evaluator as vllm_eval

    class SamplingParams:
        def __init__(self, n, max_tokens, temperature, top_p, seed):
            if temperature == 0.0 and n > 1:
                raise ValueError(f"n must be 1 when using greedy sampling, got {n}.")
            self.n = n

    class LLM:
        def __init__(self, model, **options):
            pass

        def generate(self, prompts, sampling_params, use_tqdm):
            return [
                types.SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[types.SimpleNamespace(text=r"\boxed{1}", token_ids=[7, 8], finish_reason="stop")
                             for _ in range(sampling_params.n)],
                )
                for _ in prompts
            ]

    class _Tok:
        chat_template = None

        def __call__(self, text, add_special_tokens=True):
            return types.SimpleNamespace(input_ids=[1, 2, 3])

        def decode(self, ids, skip_special_tokens=False):
            return "q"

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams, LLM=LLM))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: _Tok())
    monkeypatch.setattr(vllm_eval, "_heads_per_layer", lambda name: [2, 2])
    monkeypatch.setattr(vllm_eval, "load_problems", lambda dataset, n: [
        {"id": f"p{i}", "question": "q", "answer": "1"} for i in range(3)
    ])
    arm = parse_arms([{"id": "fullkv_vllm", "policy": "fullkv", "engine": "vllm"}])[0]
    for do_sample in (False, True):
        payload = vllm_eval.run_vllm_generation_eval(
            model_name="m", dataset="sample", n=3, arms=[arm], out_path=tmp_path / "v.json",
            do_sample=do_sample, num_return_sequences=2, seed=4, seed_offset=1,
        )
        for record in payload["records"]:
            candidates = record["arms"]["fullkv_vllm"]["candidates"]
            assert [c["seed"] for c in candidates] == [5, 6]
            assert all(c["ok"] and c["final_cache_len"] == 5 and c["batch_size"] == 6 for c in candidates)
