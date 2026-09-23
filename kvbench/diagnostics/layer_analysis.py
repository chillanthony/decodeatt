"""Merge layer-probe shards and render TideProbe sensitivity summaries."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kvbench.diagnostics.gen_traces import _load_config, _pick


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    shared = left.index.intersection(right.index)
    if len(shared) < 2:
        return None
    left_rank = left.loc[shared].rank().to_numpy(dtype=float)
    right_rank = right.loc[shared].rank().to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _mean_pairwise_ranking(frame: pd.DataFrame, column: str) -> dict:
    rankings = {}
    for value, rows in frame.groupby(column, dropna=False):
        rankings[str(value)] = rows.groupby("layers")["nll_delta"].mean()
    correlations = [
        correlation
        for left, right in combinations(rankings.values(), 2)
        if (correlation := _rank_correlation(left, right)) is not None
    ]
    return {
        "num_rankings": len(rankings),
        "num_pairs": len(correlations),
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "min_spearman": float(np.min(correlations)) if correlations else None,
    }


def stability_metrics(frame: pd.DataFrame) -> dict:
    payload = {}
    for (scan_mode, budget, policy), rows in frame.groupby(
        ["scan_mode", "budget", "mask_policy"]
    ):
        key = f"{scan_mode}:{policy}:budget={int(budget)}"
        entry = {"cross_trace": _mean_pairwise_ranking(rows, "trace_id")}
        seeded = rows[rows["mask_seed"].notna()]
        entry["cross_seed"] = (
            _mean_pairwise_ranking(seeded, "mask_seed") if len(seeded) else None
        )
        payload[key] = entry
    return payload


def _plot_sensitivity(frame: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    preferred_mode = "layer" if (frame["scan_mode"] == "layer").any() else "group"
    plot_rows = frame[
        (frame["scan_mode"] == preferred_mode) & (frame["mask_policy"] == "random")
    ]
    if plot_rows.empty:
        plot_rows = frame[frame["scan_mode"] == preferred_mode]
    aggregate = (
        plot_rows.groupby(["layer_start", "layers", "budget"], as_index=False)[
            "nll_delta"
        ]
        .mean()
        .sort_values(["layer_start", "budget"])
    )
    pivot = aggregate.pivot(index="layers", columns="budget", values="nll_delta")
    ordered_labels = (
        aggregate[["layer_start", "layers"]]
        .drop_duplicates()
        .sort_values("layer_start")["layers"]
    )
    pivot = pivot.reindex(ordered_labels)

    figure, axis = plt.subplots(figsize=(7, max(4, 0.35 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    axis.set_xticks(
        range(len(pivot.columns)), [str(int(value)) for value in pivot.columns]
    )
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_xlabel("KV budget")
    axis.set_ylabel("layer" if preferred_mode == "layer" else "layer group")
    axis.set_title("Mean FullKV NLL increase")
    figure.colorbar(image, ax=axis, label="NLL delta")
    figure.tight_layout()
    figure.savefig(figures_dir / "layer_sensitivity_heatmap.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    for budget in pivot.columns:
        axis.plot(
            x, pivot[budget].to_numpy(), marker="o", label=f"budget={int(budget)}"
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(x, pivot.index, rotation=60, ha="right")
    axis.set_ylabel("mean NLL delta")
    axis.set_xlabel("layer" if preferred_mode == "layer" else "layer group")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "layer_sensitivity_curve.png", dpi=160)
    plt.close(figure)


def summarize_layer_sensitivity(
    *,
    shard_dir: Path,
    output_csv: Path,
    summary_csv: Path,
    metrics_path: Path,
    figures_dir: Path,
) -> dict:
    shard_paths = sorted(shard_dir.glob("layer_sensitivity.*.rank*.csv"))
    if not shard_paths:
        raise ValueError(f"no layer-sensitivity shards found in {shard_dir}")
    frames = [pd.read_csv(path) for path in shard_paths]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    if frame.empty:
        raise ValueError("layer-sensitivity shards contain no rows")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    summary = (
        frame.groupby(
            [
                "scan_mode",
                "layer_start",
                "layer_end",
                "layers",
                "budget",
                "mask_policy",
            ],
            as_index=False,
        )["nll_delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "nll_delta_mean", "std": "nll_delta_std"})
    )
    summary.to_csv(summary_csv, index=False)
    _plot_sensitivity(frame, figures_dir)
    payload = {
        "num_rows": len(frame),
        "num_traces": int(frame["trace_id"].nunique()),
        "scan_modes": sorted(frame["scan_mode"].unique().tolist()),
        "budgets": sorted(int(value) for value in frame["budget"].unique()),
        "mask_policies": sorted(frame["mask_policy"].unique().tolist()),
        "baseline_max_abs_error": float(frame["baseline_max_abs_error"].max()),
        "ranking_stability": stability_metrics(frame),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--figures-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    root = Path("/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs/tideprobe_step1")
    payload = summarize_layer_sensitivity(
        shard_dir=Path(
            _pick(
                args.shard_dir,
                config,
                ("diagnostics", "layer_sensitivity_dir"),
                root / "layer_sensitivity",
            )
        ),
        output_csv=Path(
            _pick(
                args.out,
                config,
                ("diagnostics", "layer_sensitivity_path"),
                root / "layer_sensitivity.csv",
            )
        ),
        summary_csv=Path(
            _pick(
                args.summary,
                config,
                ("diagnostics", "layer_sensitivity_summary_path"),
                root / "layer_sensitivity_summary.csv",
            )
        ),
        metrics_path=Path(
            _pick(
                args.metrics,
                config,
                ("diagnostics", "layer_sensitivity_metrics_path"),
                root / "layer_sensitivity_metrics.json",
            )
        ),
        figures_dir=Path(
            _pick(
                args.figures_dir,
                config,
                ("diagnostics", "figures_dir"),
                root / "figures",
            )
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
