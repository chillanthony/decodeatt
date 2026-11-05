"""Cross-file result summaries (script/summarize_results.py)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kvbench.datasets import load_problems
from kvbench.summary import COLUMNS, summarize

ROOT = Path(__file__).resolve().parents[1]


def _arm(ok: bool, cache: int) -> dict:
    candidate = {"ok": ok, "gen_len": 10, "final_cache_len": cache, "n_evict": 1,
                 "mean_compression_ratio": 0.5, "tokens_per_sec": 5.0,
                 "batch_tokens_per_sec": 5.0, "elapsed_sec": 2.0, "peak_memory_bytes": 100}
    return {**candidate, "candidates": [candidate]}


def _write(tmp_path, name, arm, oks, cache):
    records = [{"id": f"p{i}", "gold": "1", "arms": {arm: _arm(ok, cache)}} for i, ok in enumerate(oks)]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"records": records, "summary": {}}))
    return path


def test_arms_from_separate_files_are_compared_to_fullkv(tmp_path):
    full = _write(tmp_path, "full", "fullkv", [True, True, False, True], 400)
    rkv = _write(tmp_path, "rkv", "rkv@128", [True, False, False, True], 128)
    brief = tmp_path / "snap.json"
    brief.write_text(json.dumps({"log_mode": "brief", "summary": {"snapkv@128": {
        "num_problems": 4, "total": 4, "correct": 3, "mean_final_cache_len": 100.0,
        "mean_gen_len": 10.0, "max_peak_memory_bytes": 7,
    }}}))
    rows = {row["arm"]: row for row in summarize([full, rkv, brief])}
    assert list(rows) == ["fullkv", "rkv@128", "snapkv@128"]
    assert all(set(row) == set(COLUMNS) for row in rows.values())
    assert rows["fullkv"]["accuracy"] == 0.75
    assert rows["rkv@128"]["budget_ratio"] == pytest.approx(128 / 400)
    assert rows["rkv@128"]["fidelity_vs_full"] == pytest.approx(0.5 / 0.75)
    assert rows["rkv@128"]["agreement_with_full"] == 0.75   # p1 disagrees
    assert rows["snapkv@128"]["accuracy"] == 0.75
    assert rows["snapkv@128"]["peak_memory_bytes"] == 7
    assert rows["snapkv@128"]["agreement_with_full"] is None  # brief: no per-problem rows


def test_duplicate_problem_for_one_arm_is_rejected(tmp_path):
    a = _write(tmp_path, "a", "rkv@128", [True], 128)
    b = _write(tmp_path, "b", "rkv@128", [False], 128)
    with pytest.raises(ValueError, match="more than one file"):
        summarize([a, b])


def test_cli_writes_csv_and_json(tmp_path):
    full = _write(tmp_path, "full", "fullkv", [True, False], 400)
    rkv = _write(tmp_path, "rkv", "rkv@64", [True, True], 64)
    out, json_out = tmp_path / "s.csv", tmp_path / "s.json"
    subprocess.run(
        [sys.executable, str(ROOT / "script/summarize_results.py"), str(tmp_path / "*.json"),
         "--out", str(out), "--json-out", str(json_out)],
        check=True, capture_output=True,
    )
    assert out.read_text().splitlines()[0].split(",") == COLUMNS
    assert [row["arm"] for row in json.loads(json_out.read_text())] == ["fullkv", "rkv@64"]


def test_repeated_sample_problems_get_unique_ids():
    ids = [problem["id"] for problem in load_problems("sample", 5)]
    assert len(set(ids)) == 5
