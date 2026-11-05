"""Summarize eval result files into one per-arm table.

Split-config runs write one arm per file, so the FullKV reference usually
lives in a different file than the arm being compared.  Files are therefore
combined first (records aligned by problem id) and every arm is compared
against the FullKV arm across files.

Brief-mode files carry only per-arm summaries: their accuracy and throughput
columns are filled from the summary, while per-problem comparisons
(``agreement_with_full``) need full-mode records and are left empty.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

COLUMNS = [
    "arm", "budget", "files", "num_problems", "total", "correct", "accuracy",
    "mean_gen_len", "mean_final_cache_len", "mean_n_evict", "mean_compression_ratio",
    "mean_tokens_per_sec", "mean_batch_tokens_per_sec", "mean_elapsed_sec",
    "peak_memory_bytes", "budget_ratio", "fidelity_vs_full", "agreement_with_full",
]
# summary key in the result file -> column, used when only a summary exists.
_SUMMARY_FALLBACK = {
    "mean_gen_len": "mean_gen_len",
    "mean_final_cache_len": "mean_final_cache_len",
    "mean_n_evict": "mean_n_evict",
    "mean_compression_ratio": "mean_compression_ratio",
    "mean_tokens_per_sec": "mean_tokens_per_sec",
    "mean_batch_tokens_per_sec": "mean_batch_tokens_per_sec",
    "mean_elapsed_sec": "mean_elapsed_sec",
    "max_peak_memory_bytes": "peak_memory_bytes",
}
_BUDGET = re.compile(r"@(\d+)")


def _budget(arm: str) -> int | None:
    match = _BUDGET.search(arm)
    return int(match.group(1)) if match else None


def _mean(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(values) if values else None


def _ratio(num, den):
    if num is None or not den:
        return None
    return float(num) / float(den)


def load_arms(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    """Collect every arm across files: its per-problem rows, summary and source files."""
    arms: dict[str, dict[str, Any]] = {}
    for path in map(Path, paths):
        payload = json.loads(path.read_text())
        names = {arm for record in payload.get("records", []) for arm in record.get("arms", {})}
        names |= set(payload.get("summary", {}))
        for name in names:
            entry = arms.setdefault(name, {"records": {}, "summary": None, "files": []})
            entry["files"].append(str(path))
            for record in payload.get("records", []):
                if name not in record.get("arms", {}):
                    continue
                if record["id"] in entry["records"]:
                    raise ValueError(
                        f"arm {name!r} has problem {record['id']!r} in more than one file; "
                        "summarize each repeated run separately"
                    )
                entry["records"][record["id"]] = record["arms"][name]
            if name in payload.get("summary", {}):
                if entry["summary"] is not None and not entry["records"]:
                    raise ValueError(f"arm {name!r} appears in several brief-mode files")
                entry["summary"] = payload["summary"][name]
    return arms


def _reference_arm(arms: dict) -> str | None:
    if "fullkv" in arms:
        return "fullkv"
    return next((name for name in sorted(arms) if name.startswith("fullkv")), None)


def summarize(paths: list[str | Path]) -> list[dict[str, Any]]:
    arms = load_arms(paths)
    reference = _reference_arm(arms)
    rows_out = []
    stats = {}
    for name, entry in arms.items():
        problems = entry["records"]
        candidates = [c for arm in problems.values() for c in (arm.get("candidates") or [arm])]
        summary = entry["summary"] or {}
        if candidates:
            row = {
                "num_problems": len(problems),
                "total": len(candidates),
                "correct": sum(bool(c.get("ok")) for c in candidates),
                "mean_gen_len": _mean(candidates, "gen_len"),
                "mean_final_cache_len": _mean(candidates, "final_cache_len"),
                "mean_n_evict": _mean(candidates, "n_evict"),
                "mean_compression_ratio": _mean(candidates, "mean_compression_ratio"),
                "mean_tokens_per_sec": _mean(candidates, "tokens_per_sec"),
                "mean_batch_tokens_per_sec": _mean(candidates, "batch_tokens_per_sec"),
                "mean_elapsed_sec": _mean(candidates, "elapsed_sec"),
                "peak_memory_bytes": max(
                    (c["peak_memory_bytes"] for c in candidates if c.get("peak_memory_bytes")),
                    default=None,
                ),
            }
        else:
            row = {
                "num_problems": summary.get("num_problems"),
                "total": summary.get("total"),
                "correct": summary.get("correct"),
                **{column: summary.get(key) for key, column in _SUMMARY_FALLBACK.items()},
            }
        row["accuracy"] = _ratio(row["correct"], row["total"])
        stats[name] = row
        rows_out.append({"arm": name, "budget": _budget(name), "files": ";".join(entry["files"]), **row})

    ref = stats.get(reference) if reference else None
    ref_problems = arms[reference]["records"] if reference else {}
    for row in rows_out:
        if ref is None:
            row.update(budget_ratio=None, fidelity_vs_full=None, agreement_with_full=None)
            continue
        row["budget_ratio"] = _ratio(row["mean_final_cache_len"], ref["mean_final_cache_len"])
        row["fidelity_vs_full"] = _ratio(row["accuracy"], ref["accuracy"])
        problems = arms[row["arm"]]["records"]
        shared = sorted(set(problems) & set(ref_problems))
        row["agreement_with_full"] = (
            sum(bool(problems[pid]["ok"]) == bool(ref_problems[pid]["ok"]) for pid in shared) / len(shared)
            if shared and row["arm"] != reference else None
        )
    rows_out.sort(key=lambda row: (row["arm"] != reference, row["budget"] or 0, row["arm"]))
    return [{column: row.get(column) for column in COLUMNS} for row in rows_out]


__all__ = ["COLUMNS", "load_arms", "summarize"]
