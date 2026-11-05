"""Re-score saved generations under the current grader, without regenerating.

Full-mode result files keep every candidate's ``raw_output``; regrading
recomputes ``pred``, ``answer_source`` and ``ok`` per candidate, the arm-level
``ok``/``pred``/``pass_at_1`` (from candidate 0, as the evaluator does) and
the summary.  Brief-mode files have no outputs and cannot be regraded.
"""
from __future__ import annotations

import copy

from kvbench.metrics import extract_answer_with_source, is_correct, summarize_accuracy


def regrade_payload(payload: dict, answer_extraction: str = "boxed") -> dict:
    if "records" not in payload or payload.get("log_mode") == "brief":
        raise ValueError("only full-mode result files keep raw outputs and can be regraded")
    payload = copy.deepcopy(payload)
    for record in payload["records"]:
        for arm in record["arms"].values():
            candidates = arm.get("candidates") or []
            for candidate in candidates:
                if "raw_output" not in candidate:
                    raise ValueError(f"problem {record['id']!r}: candidate without raw_output")
                pred, source = extract_answer_with_source(candidate["raw_output"], answer_extraction)
                candidate.update(pred=pred, answer_source=source, ok=is_correct(pred, record["gold"]))
            if candidates:
                arm.update(
                    ok=candidates[0]["ok"],
                    pred=candidates[0]["pred"],
                    answer_source=candidates[0]["answer_source"],
                    pass_at_1=sum(c["ok"] for c in candidates) / len(candidates),
                )
    if "summary" in payload:
        payload["summary"] = summarize_accuracy(payload["records"])
    if isinstance(payload.get("run_config"), dict):
        payload["run_config"]["answer_extraction"] = answer_extraction
    payload["regraded"] = {"answer_extraction": answer_extraction}
    return payload


__all__ = ["regrade_payload"]
