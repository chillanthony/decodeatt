"""Detect and evaluate TideProbe transition events from fixed trace signals."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from kvbench.diagnostics.gen_traces import _load_config, _pick


DETECTORS = ("logit_entropy", "attention_entropy", "kv_delta", "fusion")
_BACKTRACK_PATTERN = re.compile(
    r"\b(?:but\s+wait|wait|i\s+made\s+(?:a|an)\s+(?:mistake|error)|"
    r"i\s+was\s+wrong|let\s+me\s+(?:reconsider|recheck|correct)|"
    r"on\s+second\s+thought)\b",
    flags=re.IGNORECASE,
)
_CONCLUSION_PATTERN = re.compile(
    r"\b(?:answer|result)\s*(?:is|=|:)\s*\$?([+-]?\d{1,8}(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling standardization with clipped boundary windows."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rolling_zscore expects a one-dimensional signal")
    if window < 3:
        raise ValueError("rolling window must be at least 3")
    if values.size == 0:
        return values.copy()
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    half = window // 2
    indices = np.arange(finite.size)
    starts = np.maximum(indices - half, 0)
    stops = np.minimum(indices + half + 1, finite.size)
    prefix = np.concatenate(([0.0], np.cumsum(finite)))
    prefix_squared = np.concatenate(([0.0], np.cumsum(finite * finite)))
    counts = stops - starts
    means = (prefix[stops] - prefix[starts]) / counts
    variances = (prefix_squared[stops] - prefix_squared[starts]) / counts - means * means
    standard_deviations = np.sqrt(np.maximum(variances, 1e-12))
    return (finite - means) / standard_deviations


def local_peaks(
    scores: np.ndarray,
    *,
    threshold: float,
    radius: int,
    min_distance: int,
) -> list[int]:
    """Thresholded local maxima followed by score-ordered non-maximum suppression."""
    scores = np.asarray(scores, dtype=np.float64)
    candidates = []
    for index, score in enumerate(scores):
        if score < threshold:
            continue
        start = max(0, index - radius)
        stop = min(scores.size, index + radius + 1)
        if score >= scores[start:stop].max():
            candidates.append(index)
    selected = []
    for index in sorted(candidates, key=lambda item: (-scores[item], item)):
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    return sorted(selected)


def build_detector_scores(trace: dict, signals: dict, window: int) -> dict[str, np.ndarray]:
    logit_entropy = trace["step_entropy"].detach().float().cpu().numpy()
    attention_entropy = signals["attention_entropy"].detach().float().cpu().numpy()
    kv_delta = signals["kv_delta"].detach().float().cpu().numpy()
    length = min(logit_entropy.size, attention_entropy.size, kv_delta.size)
    if length == 0:
        raise ValueError(f"trace {trace.get('id')} has no aligned signal samples")
    logit_change = np.abs(np.diff(logit_entropy[:length], prepend=logit_entropy[0]))
    attention_change = np.abs(
        np.diff(attention_entropy[:length], prepend=attention_entropy[0])
    )
    raw = {
        "logit_entropy": logit_change,
        "attention_entropy": attention_change,
        "kv_delta": kv_delta[:length],
    }
    standardized = {name: rolling_zscore(values, window) for name, values in raw.items()}
    positive = [np.maximum(standardized[name], 0.0) for name in DETECTORS[:-1]]
    standardized["fusion"] = np.mean(positive, axis=0)
    return standardized


def _char_to_token_index(token_text: list[str], character_index: int) -> int:
    consumed = 0
    for token_index, text in enumerate(token_text):
        consumed += len(text)
        if character_index < consumed:
            return token_index
    return max(0, len(token_text) - 1)


def _boxed_expressions(text: str):
    """Yield boxed expressions while respecting nested LaTeX braces."""
    search_start = 0
    marker = r"\boxed"
    while True:
        marker_start = text.find(marker, search_start)
        if marker_start < 0:
            return
        opening = marker_start + len(marker)
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening >= len(text) or text[opening] != "{":
            search_start = opening
            continue
        depth = 1
        cursor = opening + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            yield marker_start, text[marker_start:cursor], text[opening + 1:cursor - 1]
            search_start = cursor
        else:
            return


def automatic_weak_labels(trace: dict) -> list[dict]:
    """Find backtracking phrases and changed boxed answers in generated text."""
    token_text = trace.get("token_text") or []
    text = "".join(token_text) if token_text else trace.get("gen_text", "")
    labels = []
    for match in _BACKTRACK_PATTERN.finditer(text):
        labels.append(
            {
                "token_index": _char_to_token_index(token_text, match.start()),
                "type": "backtrack_phrase",
                "text": match.group(0),
            }
        )

    previous_answer = None
    for character_index, expression, raw_answer in _boxed_expressions(text):
        answer = re.sub(r"\s+", "", raw_answer)
        if previous_answer is not None and answer != previous_answer:
            labels.append(
                {
                    "token_index": _char_to_token_index(token_text, character_index),
                    "type": "boxed_answer_flip",
                    "text": expression,
                    "previous_answer": previous_answer,
                    "new_answer": answer,
                }
            )
        previous_answer = answer

    previous_conclusion = None
    for match in _CONCLUSION_PATTERN.finditer(text):
        conclusion = match.group(1)
        if previous_conclusion is not None and conclusion != previous_conclusion:
            labels.append(
                {
                    "token_index": _char_to_token_index(token_text, match.start()),
                    "type": "intermediate_answer_flip",
                    "text": match.group(0),
                    "previous_answer": previous_conclusion,
                    "new_answer": conclusion,
                }
            )
        previous_conclusion = conclusion

    deduplicated = {}
    for label in labels:
        key = (int(label["token_index"]), label["type"])
        deduplicated[key] = label
    return sorted(deduplicated.values(), key=lambda item: item["token_index"])


def load_manual_labels(path: str | Path | None) -> dict[str, list[dict]]:
    if not path:
        return {}
    labels: dict[str, list[dict]] = {}
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "token_index" not in row:
                raise ValueError(f"{path}:{line_number} requires id and token_index")
            row.setdefault("type", "manual")
            labels.setdefault(str(row["id"]), []).append(row)
    return labels


def detector_metrics(
    scores: np.ndarray,
    peaks: list[int],
    labels: list[dict],
    tolerance: int,
) -> dict:
    label_positions = [int(label["token_index"]) for label in labels]
    matched_labels = sum(
        any(abs(peak - position) <= tolerance for peak in peaks)
        for position in label_positions
    )
    unmatched_peaks = sum(
        not any(abs(peak - position) <= tolerance for position in label_positions)
        for peak in peaks
    )
    target = np.zeros(len(scores), dtype=np.int8)
    for position in label_positions:
        start = max(0, position - tolerance)
        stop = min(len(scores), position + tolerance + 1)
        target[start:stop] = 1
    auprc = float(average_precision_score(target, scores)) if target.any() else None
    return {
        "num_candidates": len(peaks),
        "matched_events": matched_labels,
        "num_weak_labels": len(label_positions),
        "event_recall_at_tolerance": (
            matched_labels / len(label_positions) if label_positions else None
        ),
        "auprc_tolerance_window": auprc,
        "unmatched_candidates_per_1k_tokens": unmatched_peaks * 1000.0 / max(1, len(scores)),
    }


def grouped_detector_metrics(
    score_parts: list[np.ndarray],
    peak_parts: list[list[int]],
    label_parts: list[list[dict]],
    tolerance: int,
) -> dict:
    """Aggregate event metrics without allowing matches across trace boundaries."""
    matched_labels = 0
    unmatched_peaks = 0
    num_labels = 0
    targets = []
    for scores, peaks, labels in zip(score_parts, peak_parts, label_parts):
        positions = [int(label["token_index"]) for label in labels]
        num_labels += len(positions)
        matched_labels += sum(
            any(abs(peak - position) <= tolerance for peak in peaks)
            for position in positions
        )
        unmatched_peaks += sum(
            not any(abs(peak - position) <= tolerance for position in positions)
            for peak in peaks
        )
        target = np.zeros(len(scores), dtype=np.int8)
        for position in positions:
            start = max(0, position - tolerance)
            stop = min(len(scores), position + tolerance + 1)
            target[start:stop] = 1
        targets.append(target)
    concatenated_scores = np.concatenate(score_parts)
    concatenated_target = np.concatenate(targets)
    auprc = (
        float(average_precision_score(concatenated_target, concatenated_scores))
        if concatenated_target.any()
        else None
    )
    return {
        "num_candidates": sum(len(peaks) for peaks in peak_parts),
        "matched_events": matched_labels,
        "num_weak_labels": num_labels,
        "event_recall_at_tolerance": matched_labels / num_labels if num_labels else None,
        "auprc_tolerance_window": auprc,
        "unmatched_candidates_per_1k_tokens": (
            unmatched_peaks * 1000.0 / max(1, len(concatenated_scores))
        ),
    }


def _event_context(trace: dict, token_index: int, radius: int) -> str:
    tokens = trace.get("token_text") or []
    if not tokens:
        return ""
    start = max(0, token_index - radius)
    stop = min(len(tokens), token_index + radius + 1)
    return "".join(tokens[start:stop])


def _plot_trace(
    trace_id: str,
    scores: dict[str, np.ndarray],
    peaks: list[int],
    labels: list[dict],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for axis, detector in zip(axes, DETECTORS):
        axis.plot(scores[detector], linewidth=0.8, label=detector)
        axis.axhline(0.0, color="black", linewidth=0.4)
        if detector == "fusion":
            for peak in peaks:
                axis.axvline(peak, color="tab:red", alpha=0.35, linewidth=0.7)
        for label in labels:
            axis.axvline(
                int(label["token_index"]), color="tab:green", alpha=0.45, linewidth=0.8
            )
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("generated token index")
    figure.suptitle(f"TideProbe transition signals: {trace_id}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_detection(
    *,
    trace_dir: Path,
    signals_dir: Path,
    events_path: Path,
    metrics_path: Path,
    figures_dir: Path,
    manual_labels_path: str | Path | None,
    window: int,
    threshold: float,
    peak_radius: int,
    min_distance: int,
    tolerance: int,
    max_events: int,
    context_radius: int,
) -> dict:
    trace_paths = sorted(trace_dir.glob("*.pt"))
    if not trace_paths:
        raise ValueError(f"no trace files found in {trace_dir}")
    manual_labels = load_manual_labels(manual_labels_path)
    per_detector = {name: {"scores": [], "peaks": [], "labels": []} for name in DETECTORS}
    all_events = []
    trace_summaries = []
    figures_dir.mkdir(parents=True, exist_ok=True)

    for trace_path in trace_paths:
        signal_path = signals_dir / trace_path.name
        if not signal_path.exists():
            raise ValueError(f"missing signal file for {trace_path.name}: {signal_path}")
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        signals = torch.load(signal_path, map_location="cpu", weights_only=False)
        if trace.get("id") != signals.get("id"):
            raise ValueError(f"trace/signal id mismatch: {trace.get('id')} != {signals.get('id')}")
        scores = build_detector_scores(trace, signals, window)
        labels = automatic_weak_labels(trace) + manual_labels.get(str(trace["id"]), [])
        detector_peaks = {
            name: local_peaks(
                scores[name],
                threshold=threshold,
                radius=peak_radius,
                min_distance=min_distance,
            )
            for name in DETECTORS
        }
        for name in DETECTORS:
            per_detector[name]["scores"].append(scores[name])
            per_detector[name]["peaks"].append(detector_peaks[name])
            per_detector[name]["labels"].append(labels)

        fusion_peaks = detector_peaks["fusion"]
        for token_index in fusion_peaks:
            nearby_labels = [
                label
                for label in labels
                if abs(int(label["token_index"]) - token_index) <= tolerance
            ]
            all_events.append(
                {
                    "id": trace["id"],
                    "token_index": token_index,
                    "score": float(scores["fusion"][token_index]),
                    "signals": {
                        name: float(scores[name][token_index]) for name in DETECTORS[:-1]
                    },
                    "weak_labels": nearby_labels,
                    "context": _event_context(trace, token_index, context_radius),
                }
            )
        _plot_trace(
            str(trace["id"]),
            scores,
            fusion_peaks,
            labels,
            figures_dir / f"{trace['id']}.png",
        )
        trace_summaries.append(
            {
                "id": trace["id"],
                "num_tokens": len(scores["fusion"]),
                "num_weak_labels": len(labels),
                "num_fusion_candidates": len(fusion_peaks),
            }
        )

    metrics = {}
    for name, rows in per_detector.items():
        metrics[name] = grouped_detector_metrics(
            rows["scores"], rows["peaks"], rows["labels"], tolerance
        )

    selected_events = sorted(all_events, key=lambda event: event["score"], reverse=True)
    if max_events > 0:
        selected_events = selected_events[:max_events]
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w") as handle:
        for event in selected_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    payload = {
        "config": {
            "rolling_window": window,
            "peak_threshold": threshold,
            "peak_radius": peak_radius,
            "min_distance": min_distance,
            "event_tolerance": tolerance,
            "max_review_events": max_events,
        },
        "num_traces": len(trace_summaries),
        "num_tokens": sum(row["num_tokens"] for row in trace_summaries),
        "num_output_events": len(selected_events),
        "detectors": metrics,
        "traces": trace_summaries,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--signals-dir", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--manual-labels", default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--peak-radius", type=int, default=None)
    parser.add_argument("--min-distance", type=int, default=None)
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--context-radius", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    diagnostic_config = config.get("diagnostics", {})
    transition_config = diagnostic_config.get("transition", {})
    output_root = Path("runs/tideprobe_step1")
    payload = run_detection(
        trace_dir=Path(_pick(args.trace_dir, config, ("diagnostics", "trace_dir"), "/home/ma-user/work/bucket-wulan-green/chenyanbo/trace")),
        signals_dir=Path(_pick(args.signals_dir, config, ("diagnostics", "signals_dir"), output_root / "signals")),
        events_path=Path(_pick(args.events, config, ("diagnostics", "events_path"), output_root / "transition_events.jsonl")),
        metrics_path=Path(_pick(args.metrics, config, ("diagnostics", "metrics_path"), output_root / "transition_metrics.json")),
        figures_dir=Path(_pick(args.figures_dir, config, ("diagnostics", "figures_dir"), output_root / "figures")),
        manual_labels_path=args.manual_labels or diagnostic_config.get("manual_labels"),
        window=int(args.window if args.window is not None else transition_config.get("rolling_window", 64)),
        threshold=float(args.threshold if args.threshold is not None else transition_config.get("peak_threshold", 2.0)),
        peak_radius=int(args.peak_radius if args.peak_radius is not None else transition_config.get("peak_radius", 8)),
        min_distance=int(args.min_distance if args.min_distance is not None else transition_config.get("min_distance", 16)),
        tolerance=int(args.tolerance if args.tolerance is not None else transition_config.get("event_tolerance", 16)),
        max_events=int(args.max_events if args.max_events is not None else transition_config.get("max_review_events", 50)),
        context_radius=int(args.context_radius if args.context_radius is not None else transition_config.get("context_radius", 32)),
    )
    print(json.dumps(payload["detectors"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
