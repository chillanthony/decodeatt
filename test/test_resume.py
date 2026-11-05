"""Resuming interrupted runs and attention-backend selection (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import kvbench.evaluator as evaluator
from kvbench.arms import parse_arm
from kvbench.config import arm_attn_backend
from kvbench.resume import ResumeMismatch
from kveviction.attention_logits import require_eager_attention
from kveviction.strategies.token import get_policy

PROBLEMS = [{"id": f"p{i}", "question": "ab" * (i + 1), "answer": "0"} for i in range(4)]


class _Tokenizer:
    chat_template = None
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, question, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.arange(3, 5 + len(question)).unsqueeze(0))

    def decode(self, ids, skip_special_tokens=True):
        return ""


class _Model(torch.nn.Module):
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
        return evaluator.run_generation_eval(_Model(), _Tokenizer(), **kwargs)


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


def test_partial_file_resumes_remaining_problems_in_dataset_order(tmp_path):
    out = tmp_path / "r.json"
    full = _run(out)
    _truncate(out, keep=2)
    calls = []
    resumed = _run(out, calls=calls)
    assert len(calls) == 2
    assert [r["id"] for r in resumed["records"]] == ["p0", "p1", "p2", "p3"]
    assert _scores(resumed) == _scores(full)


def test_cross_problem_batches_resume_too(tmp_path):
    out = tmp_path / "r.json"
    full = _run(out, problem_batch_size=2)
    _truncate(out, keep=3)
    resumed = _run(out, problem_batch_size=2)
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
    assert len(calls) == 4 and payload["run_config"]["seed"] == 1


def test_file_without_run_config_is_refused(tmp_path):
    out = tmp_path / "r.json"
    out.write_text(json.dumps({"records": []}))
    with pytest.raises(ResumeMismatch, match="no run_config"):
        _run(out)


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


def test_fast_observe_moves_only_cache_aware_policies_off_eager():
    for policy in ("rkv", "snapkv", "h2o"):
        assert arm_attn_backend(policy, "auto", "sdpa") == "eager"
        assert arm_attn_backend(policy, "auto", "sdpa", fast_observe=True) == "sdpa"
    assert arm_attn_backend("window", "auto", "sdpa", fast_observe=True) == "eager"
    assert arm_attn_backend("streamingllm", "auto", "sdpa") == "sdpa"
    assert arm_attn_backend("rkv", "eager", "sdpa", fast_observe=True) == "eager"


def test_window_on_sdpa_fails_fast():
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    with pytest.raises(ValueError, match="eager"):
        require_eager_attention(model, get_policy("window"), use_attention_logits=False)
    require_eager_attention(model, get_policy("rkv"), use_attention_logits=True)
    require_eager_attention(model, get_policy("streamingllm"), use_attention_logits=False)
    require_eager_attention(model, get_policy("window"), use_attention_logits=False, track=False)
