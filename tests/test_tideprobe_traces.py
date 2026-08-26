from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from kv_eviction.runner_token import score_trace_token
from kvbench.diagnostics.eviction_alignment import (
    analyze_eviction_alignment,
    transition_hit_fraction,
)
from kvbench.diagnostics.eviction_scoring import assigned_strategy
from kvbench.diagnostics.gen_traces import token_statistics
from kvbench.diagnostics.layer_analysis import summarize_layer_sensitivity
from kvbench.diagnostics.layer_sensitivity import (
    CSV_FIELDS,
    layer_sets,
    probe_trace_point,
    select_probe_points,
)
from kvbench.diagnostics.sparse_cliff import (
    analyze_sparse_cliff,
    correction_fidelity,
    transition_token_mask,
)
from kvbench.diagnostics.transition_detection import (
    automatic_weak_labels,
    local_peaks,
    rolling_zscore,
    run_detection,
)
from kvbench.diagnostics.transition_signals import (
    attention_entropy_by_layer,
    kv_delta_by_layer,
)


def test_token_statistics_matches_categorical_distribution():
    logits = (
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, torch.log(torch.tensor(3.0))]]),
    )
    token_ids = torch.tensor([0, 1])

    entropy, nll = token_statistics(logits, token_ids)

    expected_entropy = torch.tensor(
        [
            torch.log(torch.tensor(2.0)),
            -(
                0.25 * torch.log(torch.tensor(0.25))
                + 0.75 * torch.log(torch.tensor(0.75))
            ),
        ]
    )
    expected_nll = torch.tensor(
        [
            torch.log(torch.tensor(2.0)),
            -torch.log(torch.tensor(0.75)),
        ]
    )
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(nll, expected_nll)


def test_token_statistics_rejects_length_mismatch():
    try:
        token_statistics((torch.zeros(1, 2),), torch.tensor([0, 1]))
    except ValueError as error:
        assert "length mismatch" in str(error)
    else:
        raise AssertionError("expected a length mismatch error")


def test_attention_entropy_and_kv_delta_primitives():
    attentions = (torch.full((1, 2, 1, 2), 0.5),)
    entropy = attention_entropy_by_layer(attentions)
    torch.testing.assert_close(entropy, torch.tensor([torch.log(torch.tensor(2.0))]))

    previous = [(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]]))]
    unchanged = [(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]]))]
    reversed_kv = [(torch.tensor([[[-1.0, 0.0]]]), torch.tensor([[[0.0, -1.0]]]))]
    torch.testing.assert_close(
        kv_delta_by_layer(previous, unchanged), torch.tensor([0.0])
    )
    torch.testing.assert_close(
        kv_delta_by_layer(previous, reversed_kv), torch.tensor([2.0])
    )


def test_rolling_standardization_and_peak_suppression():
    values = np.zeros(101)
    values[50] = 10.0
    scores = rolling_zscore(values, window=21)
    assert local_peaks(scores, threshold=2.0, radius=4, min_distance=8) == [50]


def test_automatic_weak_labels_finds_backtrack_and_boxed_flip():
    text = (
        r"The answer is 1 and first \boxed{\frac{2}{2}}. "
        r"But wait, the answer is 2; correction: \boxed{2}."
    )
    trace = {"token_text": list(text), "gen_text": text}

    labels = automatic_weak_labels(trace)

    assert {label["type"] for label in labels} == {
        "backtrack_phrase",
        "boxed_answer_flip",
        "intermediate_answer_flip",
    }


def test_transition_detection_writes_events_metrics_and_figure(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    signals_dir = tmp_path / "signals"
    figures_dir = tmp_path / "figures"
    trace_dir.mkdir()
    signals_dir.mkdir()
    length = 100
    entropy = torch.zeros(length)
    entropy[50] = 8.0
    attention = torch.zeros(length)
    attention[50] = 8.0
    kv_delta = torch.zeros(length)
    kv_delta[50] = 8.0
    text = "x" * 44 + " but wait " + "x" * 46
    trace = {
        "id": "synthetic-0",
        "step_entropy": entropy,
        "token_text": list(text),
        "gen_text": text,
    }
    signals = {
        "id": "synthetic-0",
        "attention_entropy": attention,
        "kv_delta": kv_delta,
    }
    torch.save(trace, trace_dir / "synthetic-0.pt")
    torch.save(signals, signals_dir / "synthetic-0.pt")
    events_path = tmp_path / "transition_events.jsonl"
    metrics_path = tmp_path / "transition_metrics.json"

    payload = run_detection(
        trace_dir=trace_dir,
        signals_dir=signals_dir,
        events_path=events_path,
        metrics_path=metrics_path,
        figures_dir=figures_dir,
        manual_labels_path=None,
        window=21,
        threshold=2.0,
        peak_radius=4,
        min_distance=8,
        tolerance=16,
        max_events=50,
        context_radius=8,
    )

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert payload["num_traces"] == 1
    assert payload["detectors"]["fusion"]["event_recall_at_tolerance"] == 1.0
    assert events[0]["token_index"] == 50
    assert (figures_dir / "synthetic-0.png").is_file()


def test_layer_groups_and_probe_point_selection():
    assert layer_sets(10, scan_mode="group", group_size=4) == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9),
    ]
    assert layer_sets(10, scan_mode="layer", group_size=4, selected_layers=[2, 7]) == [
        (2,),
        (7,),
    ]
    points = select_probe_points(
        500,
        [{"token_index": 250, "score": 3.0}],
        horizon=32,
        min_probe_index=32,
        uniform_count=1,
        max_transitions=1,
        ordinary_offset=128,
        transition_exclusion=64,
        probe_types={"uniform", "transition", "ordinary"},
    )
    assert {point["probe_type"] for point in points} == {
        "uniform",
        "transition",
        "ordinary",
    }


def test_layer_mask_probe_with_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(3)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    prompt = torch.tensor([1, 4, 7, 2])
    generated = torch.tensor([8, 3, 9, 5, 6, 11, 12, 2])
    full_ids = torch.cat([prompt, generated]).reshape(1, -1)
    with torch.no_grad():
        logits = model(full_ids).logits[0, len(prompt) - 1 : -1].float()
        fullkv_nll = (
            -torch.log_softmax(logits, -1).gather(1, generated.unsqueeze(1)).squeeze(1)
        )
    trace = {
        "id": "tiny",
        "prompt_ids": prompt,
        "gen_ids": generated,
        "fullkv_nll": fullkv_nll,
    }

    rows = probe_trace_point(
        model,
        trace,
        {"probe_type": "uniform", "probe_index": 3, "score": None},
        layer_groups=[(0,), (1,)],
        scan_mode="layer",
        budgets=[2],
        mask_policies=["random", "window", "rkv"],
        random_seeds=[0],
        horizon=3,
        rkv_params={"alpha": 1, "redundancy_chunk": 2},
    )

    assert len(rows) == 6
    assert max(row["baseline_max_abs_error"] for row in rows) < 1e-4
    assert all(np.isfinite(row["nll_sparse"]) for row in rows)
    assert any(abs(row["nll_delta"]) > 1e-7 for row in rows)


def test_layer_sensitivity_summary_outputs(tmp_path: Path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = []
    for trace_index in range(2):
        for layer_index in range(2):
            for seed in range(2):
                rows.append(
                    {
                        "trace_id": f"trace-{trace_index}",
                        "probe_type": "uniform",
                        "probe_index": 100,
                        "transition_score": None,
                        "history_length": 120,
                        "horizon": 32,
                        "scan_mode": "layer",
                        "layer_start": layer_index,
                        "layer_end": layer_index,
                        "layers": str(layer_index),
                        "budget": 64,
                        "mask_policy": "random",
                        "mask_seed": seed,
                        "nll_fullkv": 1.0,
                        "nll_sparse": 1.0 + 0.1 * (layer_index + trace_index + seed),
                        "nll_delta": 0.1 * (layer_index + trace_index + seed),
                        "baseline_max_abs_error": 1e-6,
                    }
                )
    with open(
        shard_dir / "layer_sensitivity.layer.random.rank0.csv", "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = summarize_layer_sensitivity(
        shard_dir=shard_dir,
        output_csv=tmp_path / "layer_sensitivity.csv",
        summary_csv=tmp_path / "layer_sensitivity_summary.csv",
        metrics_path=tmp_path / "layer_sensitivity_metrics.json",
        figures_dir=tmp_path / "figures",
    )

    assert payload["num_rows"] == 8
    assert payload["num_traces"] == 2
    assert (tmp_path / "figures" / "layer_sensitivity_heatmap.png").is_file()
    assert (tmp_path / "figures" / "layer_sensitivity_curve.png").is_file()


def test_strategy_rank_assignment_is_balanced():
    strategies = ["rkv", "snapkv", "window", "random"]
    assignments = [assigned_strategy(strategies, rank, 8) for rank in range(8)]
    assert [assignment[0] for assignment in assignments] == strategies * 2
    assert [assignment[1] for assignment in assignments] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(assignment[2] == 2 for assignment in assignments)


def test_score_trace_debug_returns_exact_position_counts():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    full_ids = torch.arange(1, 17) % config.vocab_size

    for backend, policy_params in (
        ("random", {}),
        ("rkv", {"alpha": 1, "redundancy_chunk": 2}),
    ):
        result = score_trace_token(
            model,
            None,
            full_ids,
            prompt_len=4,
            refl_steps=[3],
            anchor_mode="none",
            budget=6,
            recent=1,
            sink=0,
            backend=backend,
            evict_every=2,
            obs_window=1,
            span_len=2,
            policy_params=policy_params,
            debug=True,
            seed=7,
        )
        assert result["per_token_nll"].shape == (12,)
        assert result["eviction_events"]
        for event in result["eviction_events"]:
            kept_count = sum(count for _, count in event["kept_position_counts"])
            evicted_count = sum(count for _, count in event["evicted_position_counts"])
            assert kept_count == 2 * 2 * event["cache_len_after"]
            assert evicted_count == 2 * 2 * (
                event["cache_len_before"] - event["cache_len_after"]
            )


def test_eviction_alignment_analysis_and_matching(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    raw_dir = tmp_path / "raw"
    trace_dir.mkdir()
    raw_dir.mkdir()
    prompt_length = 5
    num_tokens = 100
    trace = {
        "id": "trace-0",
        "fullkv_nll": torch.ones(num_tokens),
    }
    torch.save(trace, trace_dir / "trace-0.pt")
    events_path = tmp_path / "transition_events.jsonl"
    events_path.write_text(
        json.dumps({"id": "trace-0", "token_index": 10, "score": 3.0}) + "\n"
    )
    sparse_nll = torch.ones(num_tokens)
    sparse_nll[20:30] += 0.5
    sparse_nll[60:70] += 0.1
    raw = {
        "trace_id": "trace-0",
        "strategy": "random",
        "budget": 64,
        "prompt_length": prompt_length,
        "per_token_nll": sparse_nll,
        "pre_eviction_nll_max_abs_error": 1e-6,
        "eviction_events": [
            {
                "eviction_step": 20,
                "cache_len_before": 80,
                "cache_len_after": 64,
                "evicted": 16,
                "evicted_positions": [0, 15],
                "kept_positions": [1, 2, 3],
                "evicted_position_counts": [[0, 4], [15, 4]],
                "kept_position_counts": [[1, 4], [2, 4], [3, 4]],
            },
            {
                "eviction_step": 60,
                "cache_len_before": 80,
                "cache_len_after": 64,
                "evicted": 16,
                "evicted_positions": [30],
                "kept_positions": [40, 41],
                "evicted_position_counts": [[30, 8]],
                "kept_position_counts": [[40, 4], [41, 4]],
            },
        ],
    }
    torch.save(raw, raw_dir / "trace-0.random.b64.pt")

    metrics = analyze_eviction_alignment(
        raw_dir=raw_dir,
        trace_dir=trace_dir,
        events_path=events_path,
        output_csv=tmp_path / "eviction_alignment.csv",
        metrics_path=tmp_path / "eviction_alignment_metrics.json",
        figure_path=tmp_path / "figures" / "transition_hit_nll.png",
        transition_window=2,
        downstream_horizon=10,
        high_hit_threshold=0.1,
        low_hit_threshold=0.0,
        evict_every=20,
    )

    assert (
        transition_hit_fraction(
            [[0, 4], [15, 4]],
            prompt_length=prompt_length,
            transition_positions=[10],
            window=2,
        )
        == 0.5
    )
    assert metrics["num_events"] == 2
    assert metrics["num_high_hit"] == 1
    assert metrics["num_matched_controls"] == 1
    assert abs(metrics["matched_pair_mean_nll_difference"] - 0.4) < 1e-6
    assert (tmp_path / "eviction_alignment.csv").is_file()
    assert (tmp_path / "figures" / "transition_hit_nll.png").is_file()


def test_sparse_cliff_analysis_from_eviction_nll(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    raw_dir = tmp_path / "raw"
    trace_dir.mkdir()
    raw_dir.mkdir()
    num_tokens = 24
    trace = {
        "id": "trace-0",
        "fullkv_nll": torch.full((num_tokens,), 2.0),
    }
    torch.save(trace, trace_dir / "trace-0.pt")
    events_path = tmp_path / "transition_events.jsonl"
    events_path.write_text(
        json.dumps({"id": "trace-0", "token_index": 10, "score": 3.0}) + "\n"
    )
    transition_mask = transition_token_mask(num_tokens, [10], window=1)
    assert transition_mask.nonzero(as_tuple=True)[0].tolist() == [9, 10, 11]
    for budget, transition_nll in ((512, 3.0), (1024, 2.5), (1536, 2.0)):
        sparse_nll = torch.full((num_tokens,), 2.0)
        sparse_nll[transition_mask] = transition_nll
        raw = {
            "trace_id": "trace-0",
            "strategy": "random",
            "budget": budget,
            "per_token_nll": sparse_nll,
            "pre_eviction_nll_max_abs_error": 1e-6,
        }
        torch.save(raw, raw_dir / f"trace-0.random.b{budget}.pt")

    metrics = analyze_sparse_cliff(
        raw_dir=raw_dir,
        trace_dir=trace_dir,
        events_path=events_path,
        output_csv=tmp_path / "sparse_cliff.csv",
        summary_csv=tmp_path / "sparse_cliff_summary.csv",
        metrics_path=tmp_path / "sparse_cliff_metrics.json",
        figure_path=tmp_path / "figures" / "budget_cf.png",
        strategies=["random"],
        budgets=[512, 1024, 1536],
        transition_window=1,
        safety_threshold=0.9,
    )

    assert correction_fidelity(2.0, 3.0) == 0.5
    diagnostic = metrics["cliff_diagnostics"]["random"]
    assert diagnostic["correction_fidelity"] == [0.5, 0.75, 1.0]
    assert diagnostic["first_budget_at_or_above_safety_threshold"] == 1536
    assert diagnostic["monotonicity_violations"] == 0
    assert (tmp_path / "sparse_cliff.csv").is_file()
    assert (tmp_path / "sparse_cliff_summary.csv").is_file()
    assert (tmp_path / "figures" / "budget_cf.png").is_file()
