#!/usr/bin/env python
"""Aggregate the fast-branch R-KV benchmark JSONs into a markdown table.

Reads every ``*.json`` under a results dir (default
``runs/rkvrepro/fast``), prints a markdown table of the key throughput rows
to stdout, and writes the same table to ``efficiency/benchmark/RESULTS.md``.

Usage:
    python efficiency/scripts/summarize.py [results_dir]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_MODEL_LABEL = {
    "aime": "AIME_2024 (30 problems)",
    "gsm8k": "GSM8K few-shot (200 questions)",
}

_COLS = [
    ("label", "label"),
    ("wall_s", "wall_s"),
    ("avg_gen_len", "avg_len"),
    ("decode_tok_s", "decode_tok/s"),
    ("per_gpu_decode_tok_s", "per-GPU"),
    ("dp", "dp"),
    ("compactions", "compactions"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="runs/rkvrepro/fast")
    ap.add_argument("--out", default="efficiency/benchmark/RESULTS.md")
    args = ap.parse_args()

    jsons = sorted(
        p for p in os.listdir(args.results_dir) if p.endswith(".json")
    )
    rows = []
    for name in jsons:
        with open(os.path.join(args.results_dir, name)) as f:
            rows.append(json.load(f))

    if not rows:
        print("no result JSONs found in", args.results_dir, file=sys.stderr)
        return 1

    header = "| " + " | ".join(c[1] for c in _COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [f"# Fast-branch R-KV throughput on 8×A100/A800", ""]
    # A single header line for the shared dataset/model across all rows.
    model = rows[0].get("model", "R1-Distill-Llama-8B")
    lines.append(f"- **model**: {model}")
    lines.append(f"- **dataset**: {_MODEL_LABEL.get(rows[0].get('data', 'aime'), 'aime')}")
    lines.append(f"- **max_tokens**: {rows[0].get('max_tokens')}")
    lines.append("")

    for key in ("single-GPU", "8-GPU data-parallel"):
        group = (
            [r for r in rows if r.get("dp", 1) == 1]
            if key == "single-GPU"
            else [r for r in rows if r.get("dp", 1) > 1]
        )
        if not group:
            continue
        lines.append(f"## {key}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for r in group:
            cells = []
            for field, _label in _COLS:
                v = r.get(field)
                if isinstance(v, float):
                    v = f"{v:.1f}"
                cells.append(str(v if v is not None else "-"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    out = "\n".join(lines) + "\n"
    print(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
