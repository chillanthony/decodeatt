import json
import math

from kvbench.evaluator import _arm_payload, _result_payload
from scripts.eval import _merge_brief_summaries


def _candidate(*, ok: bool, seed: int, text_size: int = 0) -> dict:
    answer = "1" if ok else "2"
    return {
        "text": "x" * text_size + f"\\boxed{{{answer}}}",
        "gen_ids": list(range(20)),
        "seed": seed,
        "n_evict": 3,
        "final_cache_len": 128,
        "elapsed_sec": 2.0,
        "tokens_per_sec": 10.0,
        "peak_memory_bytes": None,
        "mean_compression_ratio": 0.5,
        "evict_events": [{"step": 4, "cache_len_after": 128}],
        "mean_effective_cache_len": 128.0,
        "max_effective_cache_len": 128.0,
        "total_effective_kv_tokens": 1024.0,
        "total_evicted_tokens": 64,
        "total_effective_evicted_tokens": 512.0,
        "mean_evicted_per_event": 64.0,
        "mean_effective_evicted_per_event": 512.0,
        "min_compression_ratio": 0.4,
        "max_compression_ratio": 0.6,
        "mean_effective_compression_ratio": 0.5,
        "min_effective_compression_ratio": 0.4,
        "max_effective_compression_ratio": 0.6,
        "head_budget_mean": 128.0,
        "head_budget_std": 0.0,
        "head_budget_min": 128.0,
        "head_budget_max": 128.0,
        "head_budget_entropy": 1.0,
        "num_underfilled_heads": 0,
        "prefill_sec": 0.1,
        "decode_sec": 1.9,
        "decode_forward_sec": 1.5,
        "attention_observation_sec": 0.2,
        "eviction_sec_total": 0.1,
        "eviction_sec_mean": 0.03,
        "other_decode_sec": 0.1,
    }


def _record(problem_id: str, candidates: list[dict], *, log_mode: str) -> dict:
    return {
        "id": problem_id,
        "gold": "1",
        "arms": {
            "rkv@128": _arm_payload(
                candidates,
                "1",
                seed=0,
                seed_offset=0,
                log_mode=log_mode,
            )
        },
    }


def test_brief_payload_omits_large_generation_details():
    records = [
        _record(
            "p1",
            [_candidate(ok=True, seed=0, text_size=1_000_000)],
            log_mode="brief",
        )
    ]

    payload = _result_payload(records, "brief", final=True)
    encoded = json.dumps(payload)

    assert payload["log_mode"] == "brief"
    assert payload["num_problems"] == 1
    assert "records" not in payload
    assert "raw_output" not in encoded
    assert "evict_events" not in encoded
    assert len(encoded) < 10_000
    assert payload["summary"]["rkv@128"]["accuracy"] == 1.0


def test_full_payload_keeps_backward_compatible_records():
    records = [_record("p1", [_candidate(ok=True, seed=0)], log_mode="full")]
    payload = _result_payload(records, "full", final=True)

    assert payload["records"] is records
    assert "raw_output" in payload["records"][0]["arms"]["rkv@128"]
    assert "evict_events" in payload["records"][0]["arms"]["rkv@128"]


def test_brief_and_full_modes_produce_the_same_summary():
    candidates = [_candidate(ok=True, seed=0), _candidate(ok=False, seed=1)]
    full = _result_payload(
        [_record("p1", candidates, log_mode="full")], "full", final=True
    )
    brief = _result_payload(
        [_record("p1", candidates, log_mode="brief")], "brief", final=True
    )

    assert brief["summary"] == full["summary"]


def test_brief_shard_summaries_merge_with_candidate_weighting():
    shard_a = _result_payload(
        [_record("p1", [_candidate(ok=True, seed=0)], log_mode="brief")],
        "brief",
        final=True,
    )
    shard_b = _result_payload(
        [
            _record(
                "p2",
                [_candidate(ok=False, seed=1), _candidate(ok=True, seed=2)],
                log_mode="brief",
            )
        ],
        "brief",
        final=True,
    )

    summary = _merge_brief_summaries([shard_a, shard_b])["rkv@128"]

    assert summary["num_problems"] == 2
    assert summary["correct"] == 2
    assert summary["total"] == 3
    assert math.isclose(summary["accuracy"], 2 / 3)
    assert summary["mean_gen_len"] == 20


if __name__ == "__main__":
    test_brief_payload_omits_large_generation_details()
    test_full_payload_keeps_backward_compatible_records()
    test_brief_and_full_modes_produce_the_same_summary()
    test_brief_shard_summaries_merge_with_candidate_weighting()
    print("Brief log mode tests passed")
