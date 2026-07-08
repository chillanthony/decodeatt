"""Evaluation metrics and answer normalization."""
from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def normalize_answer(value: str) -> str:
    value = value.strip().replace(" ", "").replace("\\!", "").replace("\\,", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    value = value.rstrip(".")
    if value.startswith("\\text{") and value.endswith("}"):
        value = value[6:-1]
    return value


def extract_answer(text: str) -> str | None:
    boxed = list(_BOXED.finditer(text))
    if boxed:
        return normalize_answer(boxed[-1].group(1))
    fallback = re.findall(r"(?:answer|=)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text[-200:], re.I)
    return normalize_answer(fallback[-1]) if fallback else None


def is_correct(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    gold_norm = normalize_answer(gold)
    if prediction == gold_norm:
        return True
    try:
        return abs(float(prediction) - float(gold_norm)) < 1e-6
    except (TypeError, ValueError):
        return False


def summarize_accuracy(records: list[dict]) -> dict:
    arms = sorted({arm for row in records for arm in row["arms"]})
    summary = {}
    for arm in arms:
        rows = [row["arms"][arm] for row in records if arm in row["arms"]]
        correct = sum(bool(row["ok"]) for row in rows)
        summary[arm] = {
            "accuracy": correct / len(rows) if rows else 0.0,
            "correct": correct,
            "total": len(rows),
            "mean_gen_len": sum(row["gen_len"] for row in rows) / len(rows) if rows else 0.0,
            "mean_final_cache_len": (
                sum(row["final_cache_len"] for row in rows) / len(rows) if rows else 0.0
            ),
            "mean_n_evict": sum(row["n_evict"] for row in rows) / len(rows) if rows else 0.0,
            "mean_elapsed_sec": (
                sum(row["elapsed_sec"] for row in rows) / len(rows) if rows else 0.0
            ),
            "mean_tokens_per_sec": (
                sum(row["tokens_per_sec"] for row in rows) / len(rows) if rows else 0.0
            ),
            "mean_compression_ratio": (
                sum(row["mean_compression_ratio"] for row in rows) / len(rows) if rows else 1.0
            ),
            "max_peak_memory_bytes": max(
                (row["peak_memory_bytes"] or 0 for row in rows), default=0
            ),
        }
    return summary
