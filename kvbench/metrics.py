"""Evaluation metrics and answer normalization."""
from __future__ import annotations

import re


try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
    from math_verify import parse as _math_parse
    from math_verify import verify as _math_verify
except Exception:  # pragma: no cover - depends on optional runtime dependency
    ExprExtractionConfig = LatexExtractionConfig = StringExtractionConfig = None
    _math_parse = _math_verify = None


def normalize_answer(value: str) -> str:
    value = value.strip().replace(" ", "").replace("\\!", "").replace("\\,", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    value = value.rstrip(".")
    if value.startswith("\\text{") and value.endswith("}"):
        value = value[6:-1]
    return value


def _last_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        i = start + len(marker)
        depth = 1
        while i < len(text):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(marker) : i]
            i += 1
    return None


def _extraction_configs():
    configs = []
    for cls in (LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig):
        if cls is None:
            continue
        try:
            configs.append(cls())
        except TypeError:
            pass
    return configs


def _parse_math(value: str):
    if _math_parse is None:
        return []
    configs = _extraction_configs()
    try:
        return _math_parse(value, extraction_config=configs)
    except TypeError:
        return _math_parse(value)
    except Exception:
        return []


def extract_answer(text: str) -> str | None:
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return normalize_answer(boxed)

    parsed = _parse_math(text)
    if parsed:
        return normalize_answer(str(parsed[-1]))

    fallback = re.findall(
        r"(?:answer\s*(?:is)?\s*[:=]?|=)\s*(-?\d+(?:\.\d+)?)",
        text[-200:],
        re.I,
    )
    return normalize_answer(fallback[-1]) if fallback else None


def is_correct(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    gold_norm = normalize_answer(gold)
    if prediction == gold_norm:
        return True
    if _math_verify is not None:
        try:
            if _math_verify(_parse_math(gold), _parse_math(prediction)):
                return True
        except Exception:
            pass
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
