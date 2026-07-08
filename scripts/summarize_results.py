"""Summarize KV benchmark JSON result files into tables."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


_ARM_BUDGET = re.compile(r"@(\d+)$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _arm_budget(arm: str) -> int | None:
    match = _ARM_BUDGET.search(arm)
    return int(match.group(1)) if match else None


def _safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def _rows_for_arm(records: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [record["arms"][arm] for record in records if arm in record.get("arms", {})]


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get("ok")) for row in rows) / len(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _max(rows: list[dict[str, Any]], key: str) -> float | int | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def summarize_file(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return []
    records = payload.get("records", [])
    arms = sorted({arm for record in records for arm in record.get("arms", {})})
    if not arms and payload.get("summary"):
        arms = sorted(payload["summary"])

    full_arm = "full" if "full" in arms else None
    full_rows = _rows_for_arm(records, full_arm) if full_arm else []
    full_accuracy = _accuracy(full_rows)
    full_cache = _mean(full_rows, "final_cache_len")
    out = []
    for arm in arms:
        rows = _rows_for_arm(records, arm)
        summary = payload.get("summary", {}).get(arm, {})
        accuracy = _accuracy(rows)
        if accuracy is None:
            accuracy = summary.get("accuracy")
        mean_cache = _mean(rows, "final_cache_len")
        if mean_cache is None:
            mean_cache = summary.get("mean_final_cache_len")

        agreement_with_full = None
        if full_arm and arm != full_arm and records:
            comparable = [
                record for record in records
                if full_arm in record.get("arms", {}) and arm in record.get("arms", {})
            ]
            if comparable:
                agreement_with_full = sum(
                    bool(record["arms"][arm].get("ok"))
                    == bool(record["arms"][full_arm].get("ok"))
                    for record in comparable
                ) / len(comparable)

        mean_gen_len = _mean(rows, "gen_len")
        peak_memory = _max(rows, "peak_memory_bytes")
        mean_tokens_per_sec = _mean(rows, "tokens_per_sec")
        mean_elapsed_sec = _mean(rows, "elapsed_sec")

        out.append(
            {
                "file": str(path),
                "arm": arm,
                "budget": _arm_budget(arm),
                "total": len(rows) or summary.get("total"),
                "correct": sum(bool(row.get("ok")) for row in rows)
                if rows else summary.get("correct"),
                "accuracy": accuracy,
                "pass_at_1": accuracy,
                "mean_gen_len": mean_gen_len
                if mean_gen_len is not None else summary.get("mean_gen_len"),
                "peak_memory_bytes": peak_memory
                if peak_memory is not None else summary.get("max_peak_memory_bytes"),
                "mean_tokens_per_sec": mean_tokens_per_sec
                if mean_tokens_per_sec is not None else summary.get("mean_tokens_per_sec"),
                "mean_elapsed_sec": mean_elapsed_sec
                if mean_elapsed_sec is not None else summary.get("mean_elapsed_sec"),
                "mean_final_cache_len": mean_cache,
                "budget_ratio": _safe_div(mean_cache, full_cache),
                "fidelity_vs_full": _safe_div(accuracy, full_accuracy),
                "agreement_with_full": agreement_with_full,
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "arm",
        "budget",
        "total",
        "correct",
        "accuracy",
        "pass_at_1",
        "mean_gen_len",
        "peak_memory_bytes",
        "mean_tokens_per_sec",
        "mean_elapsed_sec",
        "mean_final_cache_len",
        "budget_ratio",
        "fidelity_vs_full",
        "agreement_with_full",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["results/*.json"])
    parser.add_argument("--out", default="results/summary.csv")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    files: list[Path] = []
    for pattern in args.paths:
        matches = glob.glob(pattern)
        files.extend(Path(match) for match in matches)
    files = sorted(set(files))
    rows = [row for path in files for row in summarize_file(path)]

    out_path = Path(args.out)
    _write_csv(rows, out_path)
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
