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


ANSWER_EXTRACTIONS = ("boxed", "lenient")


def extract_answer_with_source(text: str, mode: str = "boxed") -> tuple[str | None, str]:
    """Return ``(prediction, source)``; source is ``boxed``, ``parsed``, ``regex`` or ``none``.

    ``boxed`` (default) only accepts the last ``\\boxed{...}``: an output
    without one, typically a generation cut off by ``max_new``, has no answer.
    ``lenient`` falls back to the last math expression in the text, then to an
    ``answer is N`` / ``= N`` pattern near the end.  It recovers some
    unboxed answers but also grades intermediate numbers of truncated
    reasoning, so it is kept only for comparison with older results.
    """
    if mode not in ANSWER_EXTRACTIONS:
        raise ValueError(f"answer_extraction must be one of {ANSWER_EXTRACTIONS}, got {mode!r}")
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return normalize_answer(boxed), "boxed"
    if mode == "boxed":
        return None, "none"

    parsed = _parse_math(text)
    if parsed:
        return normalize_answer(str(parsed[-1])), "parsed"

    fallback = re.findall(
        r"(?:answer\s*(?:is)?\s*[:=]?|=)\s*(-?\d+(?:\.\d+)?)",
        text[-200:],
        re.I,
    )
    return (normalize_answer(fallback[-1]), "regex") if fallback else (None, "none")


def extract_answer(text: str, mode: str = "boxed") -> str | None:
    return extract_answer_with_source(text, mode)[0]


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
            # Bare LaTeX such as "\sqrt2", "\frac12" or "\text{(C)}" only parses
            # inside math delimiters; retry with both sides wrapped in $...$.
            if _math_verify(_parse_math(f"${gold}$"), _parse_math(f"${prediction}$")):
                return True
        except Exception:
            pass
    try:
        return abs(float(prediction) - float(gold_norm)) < 1e-6
    except (TypeError, ValueError):
        return False


def _rate(rows: list[dict], key: str, predicate) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(bool(predicate(value)) for value in values) / len(values) if values else None


def summarize_accuracy(records: list[dict]) -> dict:
    arms = sorted({arm for row in records for arm in row["arms"]})
    summary = {}
    for arm in arms:
        arm_rows = [row["arms"][arm] for row in records if arm in row["arms"]]
        rows = [
            candidate
            for row in arm_rows
            for candidate in (row.get("candidates") or [row])
        ]
        correct = sum(bool(row["ok"]) for row in rows)

        def mean_field(key: str, default: float = 0.0) -> float:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            return sum(values) / len(values) if values else default

        def min_field(key: str, default: float = 0.0) -> float:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            return min(values) if values else default

        def max_field(key: str, default: float = 0.0) -> float:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            return max(values) if values else default

        summary[arm] = {
            "accuracy": correct / len(rows) if rows else 0.0,
            "pass_at_1": correct / len(rows) if rows else 0.0,
            "num_problems": len(arm_rows),
            "correct": correct,
            "total": len(rows),
            # Share of candidates that hit max_new before EOS / produced no answer.
            # Candidates from older result files without these fields are skipped.
            "truncation_rate": _rate(rows, "finished", lambda value: not value),
            "no_answer_rate": _rate(rows, "answer_source", lambda value: value == "none"),
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
            "mean_batch_tokens_per_sec": mean_field("batch_tokens_per_sec"),
            "max_batch_size": max_field("batch_size"),
            "mean_compression_ratio": (
                sum(row["mean_compression_ratio"] for row in rows) / len(rows) if rows else 1.0
            ),
            "max_peak_memory_bytes": max(
                (row["peak_memory_bytes"] or 0 for row in rows), default=0
            ),
            "mean_effective_cache_len": mean_field("mean_effective_cache_len"),
            "max_effective_cache_len": max_field("max_effective_cache_len"),
            "mean_total_effective_kv_tokens": mean_field("total_effective_kv_tokens"),
            "mean_total_evicted_tokens": mean_field("total_evicted_tokens"),
            "mean_total_effective_evicted_tokens": mean_field("total_effective_evicted_tokens"),
            "mean_evicted_per_event": mean_field("mean_evicted_per_event"),
            "mean_effective_evicted_per_event": mean_field("mean_effective_evicted_per_event"),
            "min_compression_ratio": min_field("min_compression_ratio", 1.0),
            "max_compression_ratio": max_field("max_compression_ratio", 1.0),
            "mean_effective_compression_ratio": mean_field("mean_effective_compression_ratio", 1.0),
            "min_effective_compression_ratio": min_field("min_effective_compression_ratio", 1.0),
            "max_effective_compression_ratio": max_field("max_effective_compression_ratio", 1.0),
            "mean_head_budget_mean": mean_field("head_budget_mean"),
            "mean_head_budget_std": mean_field("head_budget_std"),
            "min_head_budget_min": min_field("head_budget_min"),
            "max_head_budget_max": max_field("head_budget_max"),
            "mean_head_budget_entropy": mean_field("head_budget_entropy"),
            "mean_num_underfilled_heads": mean_field("num_underfilled_heads"),
            "mean_prefill_sec": mean_field("prefill_sec"),
            "mean_decode_sec": mean_field("decode_sec"),
            "mean_decode_forward_sec": mean_field("decode_forward_sec"),
            "mean_attention_observation_sec": mean_field("attention_observation_sec"),
            "mean_eviction_sec_total": mean_field("eviction_sec_total"),
            "mean_eviction_sec_mean": mean_field("eviction_sec_mean"),
            "mean_other_decode_sec": mean_field("other_decode_sec"),
        }
    return summary
