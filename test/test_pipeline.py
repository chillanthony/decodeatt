"""Evaluation pipeline (CPU): scoring, result schema, brief mode and resume."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import kvbench.evaluator as evaluator
from kvbench.arms import parse_arm, parse_arms
from kvbench.datasets import load_problems
from kvbench.evaluator import _arm_payload, _result_payload
from kvbench.result_schema import CANDIDATE_KEYS, missing_candidate_keys, no_eviction_candidate
from kvbench.resume import ResumeMismatch
from kvbench.sharding import merge_brief_summaries
from kveviction.runner_token import generate_token_evict



class _Tokenizer:
    eos_token_id = None
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------


def test_evaluator_forwards_anchor_settings(monkeypatch, tmp_path, tiny_llama):
    """Regression: the evaluator used to hard-code anchor_frac=0.0."""

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


def test_cross_problem_evaluator_restores_order_and_candidates(tmp_path):
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
        {"id": "short", "question": "a", "answer": "0"},
        {"id": "long", "question": "abcdefgh", "answer": "0"},
        {"id": "medium", "question": "abcd", "answer": "0"},
    ]
    with patch("kvbench.evaluator.load_problems", return_value=problems):
        payload = evaluator.run_generation_eval(
            _Model(),
            _Tokenizer(),
            dataset="sample",
            n=3,
            arms=[parse_arm("fullkv")],
            out_path=tmp_path / "result.json",
            problem_batch_size=2,
            prompt_bucket_size=3,
            num_return_sequences=3,
            batch_size=2,
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


# --------------------------------------------------------------------------
# Result schema
# --------------------------------------------------------------------------


def test_no_eviction_candidate_matches_hf_fullkv(tiny_llama):
    """vLLM's FullKV builder must agree with the HF engine's FullKV fields."""
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
    with pytest.raises(KeyError, match="missing canonical fields"):
        _arm_payload([{"text": "", "gen_ids": []}], "1", 0, 0)
    assert "batch_size" in missing_candidate_keys({})


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


# --------------------------------------------------------------------------
# Brief log mode
# --------------------------------------------------------------------------


def _candidate(*, ok: bool, seed: int, text_size: int = 0) -> dict:
    answer = "1" if ok else "2"
    return {
        "text": "x" * text_size + f"\\boxed{{{answer}}}",
        "gen_ids": list(range(20)),
        "finished": True,
        "seed": seed,
        "prompt_len": 16,
        "n_evict": 3,
        "final_cache_len": 128,
        "elapsed_sec": 2.0,
        "tokens_per_sec": 10.0,
        "batch_elapsed_sec": 2.0,
        "batch_tokens_per_sec": 10.0,
        "batch_size": 1,
        "peak_memory_bytes": None,
        "mean_compression_ratio": 0.5,
        "evict_events": [{"step": 4, "cache_len_after": 128}],
        "mean_effective_cache_len": 128.0,
        "max_effective_cache_len": 128.0,
        "min_effective_cache_len": 128.0,
        "total_effective_kv_tokens": 1024.0,
        "effective_kv_tokens_per_layer_head": [[128] * 8],
        "total_evicted_tokens": 64,
        "total_effective_evicted_tokens": 512.0,
        "mean_evicted_per_event": 64.0,
        "mean_effective_evicted_per_event": 512.0,
        "min_compression_ratio": 0.4,
        "max_compression_ratio": 0.6,
        "mean_effective_compression_ratio": 0.5,
        "min_effective_compression_ratio": 0.4,
        "max_effective_compression_ratio": 0.6,
        "cache_len_curve": [128],
        "effective_cache_len_curve": [128.0],
        "head_budget_mean": 128.0,
        "head_budget_std": 0.0,
        "head_budget_min": 128.0,
        "head_budget_max": 128.0,
        "head_budget_entropy": 1.0,
        "num_underfilled_heads": 0,
        "prefill_sec": 0.1,
        "decode_sec": 1.9,
        "decode_forward_sec": 1.5,
        "attention_observation_sec": 0.2,
        "eviction_sec_total": 0.1,
        "eviction_sec_mean": 0.03,
        "other_decode_sec": 0.1,
    }


def _record(problem_id: str, candidates: list[dict], *, log_mode: str) -> dict:
    return {
        "id": problem_id,
        "gold": "1",
        "arms": {
            "rkv@128": _arm_payload(
                candidates,
                "1",
                seed=0,
                seed_offset=0,
                log_mode=log_mode,
            )
        },
    }


def test_brief_payload_omits_large_generation_details():
    records = [
        _record(
            "p1",
            [_candidate(ok=True, seed=0, text_size=1_000_000)],
            log_mode="brief",
        )
    ]

    payload = _result_payload(records, "brief", final=True)
    encoded = json.dumps(payload)

    assert payload["log_mode"] == "brief"
    assert payload["num_problems"] == 1
    assert "records" not in payload
    assert "raw_output" not in encoded
    assert "evict_events" not in encoded
    assert len(encoded) < 10_000
    assert payload["summary"]["rkv@128"]["accuracy"] == 1.0


def test_brief_and_full_modes_produce_the_same_summary():
    candidates = [_candidate(ok=True, seed=0), _candidate(ok=False, seed=1)]
    full = _result_payload(
        [_record("p1", candidates, log_mode="full")], "full", final=True
    )
    brief = _result_payload(
        [_record("p1", candidates, log_mode="brief")], "brief", final=True
    )

    assert brief["summary"] == full["summary"]


def test_brief_shard_summaries_merge_with_candidate_weighting():
    shard_a = _result_payload(
        [_record("p1", [_candidate(ok=True, seed=0)], log_mode="brief")],
        "brief",
        final=True,
    )
    shard_b = _result_payload(
        [
            _record(
                "p2",
                [_candidate(ok=False, seed=1), _candidate(ok=True, seed=2)],
                log_mode="brief",
            )
        ],
        "brief",
        final=True,
    )

    summary = merge_brief_summaries([shard_a, shard_b])["rkv@128"]

    assert summary["num_problems"] == 2
    assert summary["correct"] == 2
    assert summary["total"] == 3
    assert math.isclose(summary["accuracy"], 2 / 3)
    assert summary["mean_gen_len"] == 20


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


PROBLEMS = [{"id": f"p{i}", "question": "ab" * (i + 1), "answer": "0"} for i in range(4)]


class _PromptTokenizer:
    chat_template = None
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, question, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.arange(3, 5 + len(question)).unsqueeze(0))

    def decode(self, ids, skip_special_tokens=True):
        return ""


class _EosModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(_name_or_path="fake", _attn_implementation="eager")

    def forward(self, input_ids, **_kwargs):
        logits = torch.full((*input_ids.shape, 8), -100.0)
        logits[:, -1, 2] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def _run(out, *, calls=None, **overrides):
    kwargs = dict(
        dataset="sample", n=4, arms=[parse_arm("fullkv")], out_path=out,
        max_new=2, do_sample=False, seed=0, policy_params={},
    )
    kwargs.update(overrides)
    real = evaluator.generate_token_evict

    def counting(*args, **kw):
        if calls is not None:
            calls.append(1)
        return real(*args, **kw)

    with patch("kvbench.evaluator.load_problems", return_value=PROBLEMS), \
            patch("kvbench.evaluator.generate_token_evict", side_effect=counting):
        return evaluator.run_generation_eval(_EosModel(), _PromptTokenizer(), **kwargs)


def _truncate(out: Path, keep: int) -> None:
    payload = json.loads(out.read_text())
    payload["records"] = payload["records"][:keep]
    payload.pop("summary")
    payload["complete"] = False
    out.write_text(json.dumps(payload))


def _scores(payload):
    return {
        arm: {key: summary[key] for key in ("num_problems", "total", "correct", "accuracy")}
        for arm, summary in payload["summary"].items()
    }


def test_complete_file_is_not_rerun(tmp_path):
    out = tmp_path / "r.json"
    first = _run(out)
    assert first["complete"] is True and len(first["records"]) == 4
    calls = []
    second = _run(out, calls=calls)
    assert calls == []
    assert second == first


@pytest.mark.parametrize("problem_batch_size", [1, 2])
def test_partial_file_resumes_remaining_problems_in_dataset_order(tmp_path, problem_batch_size):
    out = tmp_path / "r.json"
    full = _run(out, problem_batch_size=problem_batch_size)
    _truncate(out, keep=2 if problem_batch_size == 1 else 3)
    calls = []
    resumed = _run(out, calls=calls, problem_batch_size=problem_batch_size)
    if problem_batch_size == 1:
        assert len(calls) == 2
    assert [r["id"] for r in resumed["records"]] == ["p0", "p1", "p2", "p3"]
    assert _scores(resumed) == _scores(full)


def test_changed_config_refuses_to_resume_and_overwrite_starts_over(tmp_path):
    out = tmp_path / "r.json"
    _run(out)
    with pytest.raises(ResumeMismatch, match="seed: 0 -> 1"):
        _run(out, seed=1)
    with pytest.raises(ResumeMismatch, match="arms"):
        _run(out, arms=[parse_arm("window@8")])
    calls = []
    payload = _run(out, calls=calls, seed=1, resume=False)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"records": []}))
    with pytest.raises(ResumeMismatch, match="no run_config"):
        _run(legacy)
    assert len(calls) == 4 and payload["run_config"]["seed"] == 1


def test_resume_ignores_where_the_model_was_loaded_from(tmp_path):
    out = tmp_path / "r.json"
    _run(out, model_id="org/m")
    payload = json.loads(out.read_text())
    assert payload["run_config"]["model"] == "org/m" and payload["run_config"]["model_path"] == "fake"
    payload["run_config"]["model_path"] = "/elsewhere/m"   # e.g. written on another machine
    payload["complete"] = False
    payload["records"] = payload["records"][:1]
    out.write_text(json.dumps(payload))
    calls = []
    assert len(_run(out, calls=calls, model_id="org/m")["records"]) == 4 and len(calls) == 3
    with pytest.raises(ResumeMismatch, match="model"):
        _run(out, model_id="org/other")


def test_resume_from_skips_problems_done_in_other_shards(tmp_path):
    other = tmp_path / "other.json"
    _run(other, only_ids={"p0", "p1", "p2", "p3"})
    _truncate(other, keep=3)
    calls = []
    payload = _run(tmp_path / "mine.json", calls=calls, only_ids={"p0", "p1", "p2", "p3"},
                   resume_from=[other])
    assert len(calls) == 1
    assert [r["id"] for r in payload["records"]] == ["p3"]


def test_brief_partial_keeps_records_final_drops_them(tmp_path):
    out = tmp_path / "r.json"
    full = _run(out, log_mode="brief")
    assert "records" not in full and full["complete"] is True
    payload = evaluator._result_payload([], "brief", final=False, run_config={"x": 1})
    assert payload["records"] == [] and payload["complete"] is False


def test_repeated_sample_problems_get_unique_ids():
    ids = [problem["id"] for problem in load_problems("sample", 5)]
    assert len(set(ids)) == 5


def test_oom_splits_rows_and_keeps_results(monkeypatch, tiny_llama):
    """A CUDA OOM halves the batch; every row still generates what it would unsplit."""
    model, prompt = tiny_llama
    prompts = torch.cat([prompt, prompt.flip(-1), prompt.roll(3, -1)])
    arm = parse_arm("rkv@12")
    gen = dict(
        recent=2, sink=1, evict_every=4, obs_window=2, obs_decay=0.9, max_new=10,
        do_sample=True, temperature=0.8, top_p=0.95, seed=3, seed_offset=0,
        num_return_sequences=2, candidate_batch_size=2, policy_params={},
        execution_backend=None, cache_manager=None, sync_timing=False, cache_impl="growable",
    )
    expected = evaluator._generate_arm(model, _Tokenizer(), prompts, arm, dict(gen))

    real = evaluator.generate_token_evict_batch
    sizes = []

    def limited(*args, batch_size, **kwargs):
        sizes.append(batch_size)
        if batch_size > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(*args, batch_size=batch_size, **kwargs)

    monkeypatch.setattr(evaluator, "generate_token_evict_batch", limited)
    split_gen = dict(gen)
    actual = evaluator._generate_arm(model, _Tokenizer(), prompts, arm, split_gen)

    assert sizes == [6, 3, 2, 2, 2]
    assert split_gen["row_caps"] == {arm.name: 2}
    for want, got in zip(expected, actual):
        assert [c["gen_ids"] for c in got] == [c["gen_ids"] for c in want]
        assert [c["seed"] for c in got] == [c["seed"] for c in want]


def test_oom_on_a_single_row_is_raised(monkeypatch, tiny_llama):
    model, prompt = tiny_llama

    def always_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(evaluator, "generate_token_evict_batch", always_oom)
    gen = dict(
        recent=2, sink=1, evict_every=4, obs_window=2, obs_decay=0.9, max_new=4,
        do_sample=False, temperature=1.0, top_p=1.0, seed=0, seed_offset=0,
        num_return_sequences=2, candidate_batch_size=2, policy_params={},
        execution_backend=None, cache_manager=None, sync_timing=False, cache_impl="growable",
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        evaluator._generate_arm(model, _Tokenizer(), prompt, parse_arm("fullkv"), gen)
