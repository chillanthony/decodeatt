"""Align exact eviction positions with transition windows and downstream NLL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from kvbench.diagnostics.eviction_scoring import load_event_positions
from kvbench.diagnostics.gen_traces import _load_config, _pick


def weighted_position_stat(position_counts: list[list[int]], function) -> float:
    total = sum(int(count) for _, count in position_counts)
    if total == 0:
        return 0.0
    return (
        sum(function(int(position)) * int(count) for position, count in position_counts)
        / total
    )


def transition_hit_fraction(
    position_counts: list[list[int]],
    *,
    prompt_length: int,
    transition_positions: list[int],
    window: int,
) -> float:
    """Fraction of layer/head eviction slots landing in any transition window."""
    return weighted_position_stat(
        position_counts,
        lambda absolute_position: float(
            absolute_position >= prompt_length
            and any(
                abs((absolute_position - prompt_length) - transition) <= window
                for transition in transition_positions
            )
        ),
    )


def _event_rows(
    raw: dict,
    trace: dict,
    transition_positions: list[int],
    *,
    transition_window: int,
    downstream_horizon: int,
) -> list[dict]:
    sparse_nll = raw["per_token_nll"].float()
    fullkv_nll = trace["fullkv_nll"].float()
    prompt_length = int(raw["prompt_length"])
    rows = []
    for event_index, event in enumerate(raw["eviction_events"]):
        evicted_counts = event["evicted_position_counts"]
        kept_counts = event["kept_position_counts"]
        effective_evicted = sum(int(count) for _, count in evicted_counts)
        if effective_evicted == 0:
            continue
        eviction_step = int(event["eviction_step"])
        downstream_stop = min(
            eviction_step + downstream_horizon,
            int(sparse_nll.numel()),
            int(fullkv_nll.numel()),
        )
        if downstream_stop <= eviction_step:
            continue
        current_absolute_position = prompt_length + eviction_step - 1
        mean_age = weighted_position_stat(
            evicted_counts,
            lambda position, current=current_absolute_position: max(
                0, current - position
            ),
        )
        downstream_sparse = float(sparse_nll[eviction_step:downstream_stop].mean())
        downstream_fullkv = float(fullkv_nll[eviction_step:downstream_stop].mean())
        rows.append(
            {
                "event_id": (
                    f"{raw['trace_id']}:{raw['strategy']}:b{raw['budget']}:"
                    f"step{eviction_step}:event{event_index}"
                ),
                "trace_id": raw["trace_id"],
                "strategy": raw["strategy"],
                "budget": int(raw["budget"]),
                "eviction_step": eviction_step,
                "cache_len_before": int(event["cache_len_before"]),
                "cache_len_after": int(event["cache_len_after"]),
                "evicted_tokens": int(event["evicted"]),
                "effective_evicted_slots": effective_evicted,
                "mean_evicted_age": mean_age,
                "transition_hit": transition_hit_fraction(
                    evicted_counts,
                    prompt_length=prompt_length,
                    transition_positions=transition_positions,
                    window=transition_window,
                ),
                "retained_transition_hit": transition_hit_fraction(
                    kept_counts,
                    prompt_length=prompt_length,
                    transition_positions=transition_positions,
                    window=transition_window,
                ),
                "downstream_start": eviction_step,
                "downstream_stop": downstream_stop,
                "downstream_tokens": downstream_stop - eviction_step,
                "nll_fullkv_after": downstream_fullkv,
                "nll_sparse_after": downstream_sparse,
                "nll_delta_after": downstream_sparse - downstream_fullkv,
                "pre_eviction_nll_max_abs_error": float(
                    raw.get("pre_eviction_nll_max_abs_error", float("nan"))
                ),
                "evicted_positions": json.dumps(event["evicted_positions"]),
                "kept_positions": json.dumps(event["kept_positions"]),
                "evicted_position_counts": json.dumps(evicted_counts),
                "kept_position_counts": json.dumps(kept_counts),
            }
        )
    return rows


def add_matched_controls(
    frame: pd.DataFrame,
    *,
    high_hit_threshold: float,
    low_hit_threshold: float,
    evict_every: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["hit_group"] = "mid"
    frame.loc[frame["transition_hit"] >= high_hit_threshold, "hit_group"] = "high"
    frame.loc[frame["transition_hit"] <= low_hit_threshold, "hit_group"] = "low"
    frame["is_matched_control"] = False
    frame["matched_control_for"] = None
    frame["matched_control_id"] = None
    used_controls: set[int] = set()
    high_rows = frame[frame["hit_group"] == "high"].sort_values(
        "transition_hit", ascending=False
    )
    for high_index, high in high_rows.iterrows():
        candidates = frame[
            (frame["hit_group"] == "low")
            & (frame["trace_id"] == high["trace_id"])
            & (frame["strategy"] == high["strategy"])
            & (frame["budget"] == high["budget"])
            & (~frame.index.isin(used_controls))
        ]
        if candidates.empty:
            continue
        time_distance = (
            candidates["eviction_step"] - high["eviction_step"]
        ).abs() / max(evict_every, 1)
        amount_scale = max(float(high["effective_evicted_slots"]), 1.0)
        amount_distance = (
            candidates["effective_evicted_slots"] - high["effective_evicted_slots"]
        ).abs() / amount_scale
        age_scale = max(float(high["mean_evicted_age"]), 1.0)
        age_distance = (
            candidates["mean_evicted_age"] - high["mean_evicted_age"]
        ).abs() / age_scale
        control_index = (time_distance + amount_distance + age_distance).idxmin()
        used_controls.add(int(control_index))
        frame.at[control_index, "is_matched_control"] = True
        frame.at[control_index, "matched_control_for"] = high["event_id"]
        frame.at[high_index, "matched_control_id"] = frame.at[control_index, "event_id"]
    return frame


def _safe_float(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def alignment_metrics(frame: pd.DataFrame) -> dict:
    grouped = []
    for (strategy, budget, hit_group), rows in frame.groupby(
        ["strategy", "budget", "hit_group"]
    ):
        grouped.append(
            {
                "strategy": strategy,
                "budget": int(budget),
                "hit_group": hit_group,
                "count": len(rows),
                "transition_hit_mean": float(rows["transition_hit"].mean()),
                "nll_delta_mean": float(rows["nll_delta_after"].mean()),
                "nll_delta_std": _safe_float(rows["nll_delta_after"].std()),
            }
        )
    correlations = []
    for (strategy, budget), rows in frame.groupby(["strategy", "budget"]):
        correlation = rows["transition_hit"].rank().corr(rows["nll_delta_after"].rank())
        correlations.append(
            {
                "strategy": strategy,
                "budget": int(budget),
                "count": len(rows),
                "spearman_transition_hit_nll": _safe_float(correlation),
            }
        )
    paired = []
    controls = frame[frame["is_matched_control"]]
    by_id = frame.set_index("event_id")
    for _, control in controls.iterrows():
        high_id = control["matched_control_for"]
        if high_id not in by_id.index:
            continue
        high = by_id.loc[high_id]
        paired.append(
            {
                "high_event_id": high_id,
                "control_event_id": control["event_id"],
                "strategy": high["strategy"],
                "budget": int(high["budget"]),
                "nll_delta_difference": float(
                    high["nll_delta_after"] - control["nll_delta_after"]
                ),
            }
        )
    return {
        "num_events": len(frame),
        "num_high_hit": int((frame["hit_group"] == "high").sum()),
        "num_low_hit": int((frame["hit_group"] == "low").sum()),
        "num_matched_controls": int(frame["is_matched_control"].sum()),
        "pre_eviction_nll_max_abs_error": _safe_float(
            frame["pre_eviction_nll_max_abs_error"].max()
        ),
        "group_summary": grouped,
        "correlations": correlations,
        "matched_pairs": paired,
        "matched_pair_mean_nll_difference": (
            float(np.mean([row["nll_delta_difference"] for row in paired]))
            if paired
            else None
        ),
    }


def _plot_alignment(frame: pd.DataFrame, path: Path) -> None:
    strategies = sorted(frame["strategy"].unique())
    columns = 2
    rows = math.ceil(len(strategies) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4.5 * rows), squeeze=False)
    for axis, strategy in zip(axes.flat, strategies):
        strategy_rows = frame[frame["strategy"] == strategy]
        for budget, budget_rows in strategy_rows.groupby("budget"):
            axis.scatter(
                budget_rows["transition_hit"],
                budget_rows["nll_delta_after"],
                s=16,
                alpha=0.55,
                label=f"budget={int(budget)}",
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(strategy)
        axis.set_xlabel("transition-hit fraction")
        axis.set_ylabel("next-token-window NLL delta")
        axis.legend()
    for axis in axes.flat[len(strategies) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze_eviction_alignment(
    *,
    raw_dir: Path,
    trace_dir: Path,
    events_path: Path,
    output_csv: Path,
    metrics_path: Path,
    figure_path: Path,
    transition_window: int,
    downstream_horizon: int,
    high_hit_threshold: float,
    low_hit_threshold: float,
    evict_every: int,
) -> dict:
    raw_paths = sorted(raw_dir.glob("*.pt"))
    if not raw_paths:
        raise ValueError(f"no eviction score files found in {raw_dir}")
    event_positions = load_event_positions(events_path)
    traces = {}
    for path in trace_dir.glob("*.pt"):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        traces[str(trace["id"])] = trace
    rows = []
    for raw_path in raw_paths:
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        trace_id = str(raw["trace_id"])
        trace = traces.get(trace_id)
        if trace is None:
            raise ValueError(f"missing source trace for {trace_id}")
        rows.extend(
            _event_rows(
                raw,
                trace,
                event_positions.get(trace_id, []),
                transition_window=transition_window,
                downstream_horizon=downstream_horizon,
            )
        )
    if not rows:
        raise ValueError("eviction score files contain no analyzable eviction events")
    frame = add_matched_controls(
        pd.DataFrame(rows),
        high_hit_threshold=high_hit_threshold,
        low_hit_threshold=low_hit_threshold,
        evict_every=evict_every,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    metrics = alignment_metrics(frame)
    metrics.update(
        {
            "transition_window": transition_window,
            "downstream_horizon": downstream_horizon,
            "high_hit_threshold": high_hit_threshold,
            "low_hit_threshold": low_hit_threshold,
        }
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_alignment(frame, figure_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--figure", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    diagnostic_config = config.get("diagnostics", {})
    alignment_config = diagnostic_config.get("eviction_alignment", {})
    root = Path("/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs/tideprobe_step1")
    metrics = analyze_eviction_alignment(
        raw_dir=Path(
            _pick(
                args.raw_dir,
                config,
                ("diagnostics", "eviction_alignment_dir"),
                root / "eviction_alignment_raw",
            )
        ),
        trace_dir=Path(
            _pick(args.trace_dir, config, ("diagnostics", "trace_dir"), Path("/home/ma-user/work/bucket-wulan-green/chenyanbo/trace"))
        ),
        events_path=Path(
            _pick(
                args.events,
                config,
                ("diagnostics", "events_path"),
                root / "transition_events.jsonl",
            )
        ),
        output_csv=Path(
            _pick(
                args.out,
                config,
                ("diagnostics", "eviction_alignment_path"),
                root / "eviction_alignment.csv",
            )
        ),
        metrics_path=Path(
            _pick(
                args.metrics,
                config,
                ("diagnostics", "eviction_alignment_metrics_path"),
                root / "eviction_alignment_metrics.json",
            )
        ),
        figure_path=Path(
            _pick(
                args.figure,
                config,
                ("diagnostics", "eviction_alignment_figure"),
                root / "figures" / "transition_hit_nll.png",
            )
        ),
        transition_window=int(alignment_config.get("transition_window", 16)),
        downstream_horizon=int(alignment_config.get("downstream_horizon", 32)),
        high_hit_threshold=float(alignment_config.get("high_hit_threshold", 0.01)),
        low_hit_threshold=float(alignment_config.get("low_hit_threshold", 0.0)),
        evict_every=int(alignment_config.get("evict_every", 128)),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
