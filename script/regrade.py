"""Re-score result files with the current grader (no generation needed).

    python script/regrade.py runs/*/*.json                       # writes <name>.regraded.json
    python script/regrade.py runs/*/*.json --answer-extraction lenient --suffix .lenient

Prints old vs new accuracy per arm.  See ``kvbench.regrade``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.metrics import ANSWER_EXTRACTIONS  # noqa: E402
from kvbench.regrade import regrade_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="Full-mode result JSON files or glob patterns.")
    parser.add_argument("--answer-extraction", choices=ANSWER_EXTRACTIONS, default="boxed")
    parser.add_argument("--suffix", default=".regraded", help="Inserted before .json in the output name.")
    args = parser.parse_args()

    paths = [match for pattern in args.paths for match in (sorted(glob.glob(pattern)) or [pattern])]
    for path in map(Path, paths):
        if path.stem.endswith(args.suffix):
            continue
        old = json.loads(path.read_text())
        try:
            new = regrade_payload(old, args.answer_extraction)
        except ValueError as exc:
            print(f"skip {path}: {exc}")
            continue
        out = path.with_name(f"{path.stem}{args.suffix}{path.suffix}")
        out.write_text(json.dumps(new, ensure_ascii=False, indent=1))
        for arm, summary in new.get("summary", {}).items():
            before = old.get("summary", {}).get(arm, {}).get("accuracy")
            before = f"{before:.3f}" if before is not None else "-"
            print(f"{out}  {arm:28s} acc {before} -> {summary['accuracy']:.3f}  "
                  f"no_answer={summary['no_answer_rate']}")


if __name__ == "__main__":
    main()
