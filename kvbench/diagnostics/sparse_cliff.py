"""Correction Fidelity and sparse-cliff analysis from Experiment 3 NLL traces."""

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
from kvbench.diagnostics.layer_sensitivity import parse_int_list


def transition_token_mask(
    length: int, transition_positions: list[int], window: int
) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    for position in transition_positions:
        start = max(0, int(position) - window)
        stop = min(length, int(position) + window + 1)
        mask[start:stop] = True
    return mask


def correction_fidelity(fullkv_nll: float, sparse_nll: float, epsilon: float = 1e-12):
    if fullkv_nll <= epsilon:
        return None
    return 1.0 - (sparse_nll - fullkv_nll) / fullkv_nll


def _region_stats(
    fullkv_nll: torch.Tensor, sparse_nll: torch.Tensor, mask: torch.Tensor, prefix: str
) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {
            f"n_{prefix}_tokens": 0,
            f"fullkv_{prefix}_nll_sum": 0.0,
            f"sparse_{prefix}_nll_sum": 0.0,
            f"fullkv_{prefix}_nll": None,
            f"sparse_{prefix}_nll": None,
            f"{prefix}_nll_delta": None,
            f"{prefix}_fidelity": None,
        }
    full_sum = float(fullkv_nll[mask].sum())
    sparse_sum = float(sparse_nll[mask].sum())
    full_mean = full_sum / count
    sparse_mean = sparse_sum / count
    return {
        f"n_{prefix}_tokens": count,
        f"fullkv_{prefix}_nll_sum": full_sum,
        f"sparse_{prefix}_nll_sum": sparse_sum,
        f"fullkv_{prefix}_nll": full_mean,
        f"sparse_{prefix}_nll": sparse_mean,
        f"{prefix}_nll_delta": sparse_mean - full_mean,
        f"{prefix}_fidelity": correction_fidelity(full_mean, sparse_mean),
    }


def trace_cliff_row(
    raw: dict, trace: dict, transitions: list[int], window: int
) -> dict | None:
    sparse_nll = raw["per_token_nll"].detach().float().cpu()
    fullkv_nll = trace["fullkv_nll"].detach().float().cpu()
    length = min(int(sparse_nll.numel()), int(fullkv_nll.numel()))
    sparse_nll = sparse_nll[:length]
    fullkv_nll = fullkv_nll[:length]
    transition_mask = transition_token_mask(length, transitions, window)
    if not transition_mask.any():
        return None
    normal_mask = ~transition_mask
    return {
        "trace_id": raw["trace_id"],
        "strategy": raw["strategy"],
        "budget": int(raw["budget"]),
        "transition_window": window,
        "pre_eviction_nll_max_abs_error": float(
            raw.get("pre_eviction_nll_max_abs_error", float("nan"))
        ),
        **_region_stats(fullkv_nll, sparse_nll, transition_mask, "transition"),
        **_region_stats(fullkv_nll, sparse_nll, normal_mask, "normal"),
    }


def summarize_cliff_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, budget), group in frame.groupby(["strategy", "budget"]):
        transition_count = int(group["n_transition_tokens"].sum())
        normal_count = int(group["n_normal_tokens"].sum())
        full_transition = float(group["fullkv_transition_nll_sum"].sum()) / max(
            transition_count, 1
        )
        sparse_transition = float(group["sparse_transition_nll_sum"].sum()) / max(
            transition_count, 1
        )
        full_normal = float(group["fullkv_normal_nll_sum"].sum()) / max(normal_count, 1)
        sparse_normal = float(group["sparse_normal_nll_sum"].sum()) / max(
            normal_count, 1
        )
        trace_cf = group["transition_fidelity"].dropna()
        rows.append(
            {
                "strategy": strategy,
                "budget": int(budget),
                "num_traces": int(group["trace_id"].nunique()),
                "n_transition_tokens": transition_count,
                "n_normal_tokens": normal_count,
                "fullkv_transition_nll": full_transition,
                "sparse_transition_nll": sparse_transition,
                "transition_nll_delta": sparse_transition - full_transition,
                "correction_fidelity": correction_fidelity(
                    full_transition, sparse_transition
                ),
                "trace_cf_mean": float(trace_cf.mean()),
                "trace_cf_std": float(trace_cf.std()) if len(trace_cf) > 1 else 0.0,
                "fullkv_normal_nll": full_normal,
                "sparse_normal_nll": sparse_normal,
                "normal_nll_delta": sparse_normal - full_normal,
                "normal_fidelity": correction_fidelity(full_normal, sparse_normal),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "budget"]).reset_index(drop=True)


def cliff_diagnostics(summary: pd.DataFrame, safety_threshold: float) -> dict:
    diagnostics = {}
    for strategy, rows in summary.groupby("strategy"):
        rows = rows.sort_values("budget")
        budgets = rows["budget"].to_numpy(dtype=int)
        fidelity = rows["correction_fidelity"].to_numpy(dtype=float)
        gains = np.diff(fidelity)
        budget_steps = np.diff(budgets)
        gains_per_512 = gains * 512.0 / budget_steps if len(gains) else np.array([])
        intervals = [
            {
                "from_budget": int(budgets[index]),
                "to_budget": int(budgets[index + 1]),
                "cf_gain": float(gains[index]),
                "cf_gain_per_512": float(gains_per_512[index]),
            }
            for index in range(len(gains))
        ]
        largest_interval = (
            intervals[int(np.argmax(gains_per_512))] if len(gains_per_512) else None
        )
        slope_changes = np.diff(gains_per_512)
        strongest_bend = None
        if len(slope_changes):
            bend_index = int(np.argmax(np.abs(slope_changes)))
            strongest_bend = {
                "budget": int(budgets[bend_index + 1]),
                "slope_change_per_512": float(slope_changes[bend_index]),
            }
        safe = budgets[fidelity >= safety_threshold]
        diagnostics[str(strategy)] = {
            "budget_points": budgets.tolist(),
            "correction_fidelity": fidelity.tolist(),
            "interval_gains": intervals,
            "largest_gain_interval": largest_interval,
            "strongest_bend": strongest_bend,
            "monotonicity_violations": int((gains < 0).sum()),
            "first_budget_at_or_above_safety_threshold": (
                int(safe[0]) if safe.size else None
            ),
        }
    return diagnostics


def _plot_budget_cf(summary: pd.DataFrame, path: Path, safety_threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for strategy, rows in summary.groupby("strategy"):
        rows = rows.sort_values("budget")
        axis.errorbar(
            rows["budget"],
            rows["correction_fidelity"],
            yerr=rows["trace_cf_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=strategy,
        )
    axis.axhline(1.0, color="black", linewidth=0.8, label="FullKV")
    axis.axhline(
        safety_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=0.8,
        label=f"safety={safety_threshold:g}",
    )
    axis.set_xlabel("KV budget")
    axis.set_ylabel("Correction Fidelity")
    axis.set_title("TideProbe sparse-cliff pilot")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze_sparse_cliff(
    *,
    raw_dir: Path,
    trace_dir: Path,
    events_path: Path,
    output_csv: Path,
    summary_csv: Path,
    metrics_path: Path,
    figure_path: Path,
    strategies: list[str],
    budgets: list[int],
    transition_window: int,
    safety_threshold: float,
) -> dict:
    raw_paths = sorted(raw_dir.glob("*.pt"))
    if not raw_paths:
        raise ValueError(f"no eviction score files found in {raw_dir}")
    transition_positions = load_event_positions(events_path)
    traces = {}
    for path in trace_dir.glob("*.pt"):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        traces[str(trace["id"])] = trace

    observed_tasks = set()
    rows = []
    for raw_path in raw_paths:
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        strategy = str(raw["strategy"])
        budget = int(raw["budget"])
        if strategy not in strategies or budget not in budgets:
            continue
        trace_id = str(raw["trace_id"])
        observed_tasks.add((trace_id, strategy, budget))
        if trace_id not in traces:
            raise ValueError(f"missing source trace for {trace_id}")
        row = trace_cliff_row(
            raw,
            traces[trace_id],
            transition_positions.get(trace_id, []),
            transition_window,
        )
        if row is not None:
            rows.append(row)
    expected_tasks = {
        (trace_id, strategy, budget)
        for trace_id in traces
        for strategy in strategies
        for budget in budgets
    }
    missing = sorted(expected_tasks - observed_tasks)
    if missing:
        raise ValueError(f"missing Experiment 3 strategy/budget results: {missing}")
    if not rows:
        raise ValueError("no traces contain transition-window tokens")

    frame = pd.DataFrame(rows).sort_values(["strategy", "budget", "trace_id"])
    summary = summarize_cliff_rows(frame)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_budget_cf(summary, figure_path, safety_threshold)
    payload = {
        "num_trace_rows": len(frame),
        "num_traces": int(frame["trace_id"].nunique()),
        "transition_window": transition_window,
        "safety_threshold": safety_threshold,
        "strategies": strategies,
        "budgets": budgets,
        "pre_eviction_nll_max_abs_error": _finite_or_none(
            frame["pre_eviction_nll_max_abs_error"].max()
        ),
        "cliff_diagnostics": cliff_diagnostics(summary, safety_threshold),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--budgets", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    diagnostic_config = config.get("diagnostics", {})
    cliff_config = diagnostic_config.get("sparse_cliff", {})
    root = Path("runs/tideprobe_step1")
    strategies = [
        item.strip()
        for item in (
            args.strategies
            or cliff_config.get("strategies", "rkv,snapkv,window,random")
        ).split(",")
        if item.strip()
    ]
    budgets = parse_int_list(
        args.budgets or cliff_config.get("budgets", "512,1024,1536")
    )
    payload = analyze_sparse_cliff(
        raw_dir=Path(
            _pick(
                args.raw_dir,
                config,
                ("diagnostics", "eviction_alignment_dir"),
                root / "eviction_alignment_raw",
            )
        ),
        trace_dir=Path(
            _pick(args.trace_dir, config, ("diagnostics", "trace_dir"), Path("/home/ma-user/work/trace"))
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
                ("diagnostics", "sparse_cliff_path"),
                root / "sparse_cliff.csv",
            )
        ),
        summary_csv=Path(
            _pick(
                args.summary,
                config,
                ("diagnostics", "sparse_cliff_summary_path"),
                root / "sparse_cliff_summary.csv",
            )
        ),
        metrics_path=Path(
            _pick(
                args.metrics,
                config,
                ("diagnostics", "sparse_cliff_metrics_path"),
                root / "sparse_cliff_metrics.json",
            )
        ),
        figure_path=Path(
            _pick(
                args.figure,
                config,
                ("diagnostics", "sparse_cliff_figure"),
                root / "figures" / "budget_cf.png",
            )
        ),
        strategies=strategies,
        budgets=budgets,
        transition_window=int(cliff_config.get("transition_window", 16)),
        safety_threshold=float(cliff_config.get("safety_threshold", 0.9)),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
