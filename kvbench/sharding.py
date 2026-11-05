"""Merge per-group / per-rank result payloads and schedule distributed shards."""
from __future__ import annotations

import json
from pathlib import Path

from kvbench.metrics import summarize_accuracy


_SUMMARY_MIN_FIELDS = {
    "min_compression_ratio",
    "min_effective_compression_ratio",
    "min_head_budget_min",
}
_SUMMARY_MAX_FIELDS = {
    "max_peak_memory_bytes",
    "max_effective_cache_len",
    "max_compression_ratio",
    "max_effective_compression_ratio",
    "max_head_budget_max",
}


def merge_brief_summaries(payloads: list[dict]) -> dict:
    chunks_by_arm: dict[str, list[dict]] = {}
    for payload in payloads:
        for arm, summary in payload.get("summary", {}).items():
            chunks_by_arm.setdefault(arm, []).append(summary)

    merged = {}
    for arm, chunks in chunks_by_arm.items():
        total = sum(int(chunk.get("total", 0)) for chunk in chunks)
        correct = sum(int(chunk.get("correct", 0)) for chunk in chunks)
        num_problems = sum(int(chunk.get("num_problems", 0)) for chunk in chunks)
        keys = {key for chunk in chunks for key in chunk}
        arm_summary = {
            "accuracy": correct / total if total else 0.0,
            "pass_at_1": correct / total if total else 0.0,
            "num_problems": num_problems,
            "correct": correct,
            "total": total,
        }
        for key in keys - set(arm_summary):
            values = [
                (chunk[key], int(chunk.get("total", 0)))
                for chunk in chunks
                if isinstance(chunk.get(key), (int, float))
            ]
            if not values:
                continue
            if key in _SUMMARY_MIN_FIELDS:
                arm_summary[key] = min(value for value, _ in values)
            elif key in _SUMMARY_MAX_FIELDS:
                arm_summary[key] = max(value for value, _ in values)
            else:
                weight = sum(count for _, count in values)
                arm_summary[key] = (
                    sum(value * count for value, count in values) / weight if weight else 0.0
                )
        merged[arm] = arm_summary
    return merged


def merge_payloads(
    payloads: list[dict], out_path: str | Path, log_mode: str = "full"
) -> dict:
    if log_mode == "brief":
        summary = merge_brief_summaries(payloads)
        merged = {
            "log_mode": "brief",
            "num_problems": max(
                (arm.get("num_problems", 0) for arm in summary.values()), default=0
            ),
            "summary": summary,
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
        return merged

    rows_by_id: dict[str, dict] = {}
    order: list[str] = []
    for payload in payloads:
        for row in payload["records"]:
            problem_id = row["id"]
            if problem_id not in rows_by_id:
                rows_by_id[problem_id] = {
                    "id": problem_id,
                    "gold": row["gold"],
                    "arms": {},
                }
                order.append(problem_id)
            rows_by_id[problem_id]["arms"].update(row["arms"])
    records = [rows_by_id[problem_id] for problem_id in order]
    merged = {"records": records, "summary": summarize_accuracy(records)}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    return merged


def allocate_group_ranks(groups: dict[str, list], world_size: int) -> list[tuple[str, int, int]]:
    """Assign ranks to attention groups in proportion to their arm counts."""
    items = list(groups.items())
    if world_size < len(items):
        raise ValueError(
            f"distributed evaluation needs at least {len(items)} ranks for "
            f"{len(items)} attention groups, got {world_size}"
        )

    counts = [1] * len(items)
    remaining = world_size - len(items)
    if remaining:
        total_weight = sum(len(group_arms) for _, group_arms in items)
        quotas = [remaining * len(group_arms) / total_weight for _, group_arms in items]
        floors = [int(quota) for quota in quotas]
        counts = [count + floor for count, floor in zip(counts, floors)]
        leftover = remaining - sum(floors)
        fractional_order = sorted(
            range(len(items)),
            key=lambda index: (quotas[index] - floors[index], len(items[index][1])),
            reverse=True,
        )
        for index in fractional_order[:leftover]:
            counts[index] += 1

    assignments: list[tuple[str, int, int]] = []
    for (backend, _), group_world_size in zip(items, counts):
        assignments.extend(
            (backend, group_rank, group_world_size)
            for group_rank in range(group_world_size)
        )
    return assignments


def distributed_shard_path(out_path: Path, rank: int, backend: str) -> Path:
    shard_dir = out_path.parent / f"{out_path.stem}.shards"
    return shard_dir / f"{out_path.stem}.rank{rank}.{backend}{out_path.suffix}"


def claim_next_problem(store, queue_key: str) -> int:
    """Atomically claim the next zero-based problem index from a torch Store."""
    return int(store.add(queue_key, 1)) - 1


__all__ = [
    "merge_brief_summaries", "merge_payloads", "allocate_group_ranks",
    "distributed_shard_path", "claim_next_problem",
]
