"""Summarize eval result JSON files into a per-arm CSV (and optional JSON) table.

    python script/summarize_results.py runs/*/*.json --out summary.csv --json-out summary.json

Files are combined before comparison, so arms from separate split-config runs
are compared against the FullKV arm (budget_ratio, fidelity_vs_full,
agreement_with_full).  See ``kvbench.summary``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.summary import COLUMNS, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="Result JSON files or glob patterns.")
    parser.add_argument("--out", required=True, help="CSV output path.")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"result file(s) not found: {', '.join(missing)}")

    rows = summarize(paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    for row in rows:
        acc = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "-"
        fid = f"{row['fidelity_vs_full']:.3f}" if row["fidelity_vs_full"] is not None else "-"
        print(f"{row['arm']:28s} acc={acc} fidelity={fid} total={row['total']}")


if __name__ == "__main__":
    main()
