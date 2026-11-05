"""Offline regrading of saved generations."""
from __future__ import annotations

import pytest

from kvbench.regrade import regrade_payload


def _payload():
    def cand(text):
        return {"raw_output": text, "ok": True, "pred": "x", "gen_len": 3, "final_cache_len": 9,
                "n_evict": 0, "elapsed_sec": 1.0, "tokens_per_sec": 3.0, "mean_compression_ratio": 1.0,
                "peak_memory_bytes": None}
    return {
        "records": [
            {"id": "a", "gold": "9", "arms": {"rkv@128": {"candidates": [cand(r"so \boxed{9}"), cand("= 9")]}}},
            {"id": "b", "gold": "5", "arms": {"rkv@128": {"candidates": [cand("x = 5, then")]}}},
        ],
        "summary": {},
        "run_config": {"answer_extraction": "lenient"},
    }


def test_boxed_regrade_rejects_unboxed_answers():
    new = regrade_payload(_payload(), "boxed")
    a, b = (r["arms"]["rkv@128"] for r in new["records"])
    assert [c["ok"] for c in a["candidates"]] == [True, False]
    assert a["pass_at_1"] == 0.5 and a["ok"] is True
    assert b["ok"] is False and b["answer_source"] == "none"
    assert new["summary"]["rkv@128"]["correct"] == 1
    assert new["summary"]["rkv@128"]["no_answer_rate"] == pytest.approx(2 / 3)
    assert new["run_config"]["answer_extraction"] == "boxed"


def test_lenient_regrade_and_input_untouched():
    old = _payload()
    new = regrade_payload(old, "lenient")
    assert new["summary"]["rkv@128"]["correct"] == 3
    assert old["records"][0]["arms"]["rkv@128"]["candidates"][1]["pred"] == "x"


def test_brief_files_cannot_be_regraded():
    with pytest.raises(ValueError):
        regrade_payload({"log_mode": "brief", "summary": {}})
