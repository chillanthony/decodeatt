"""A2 长 trace 可分离性诊断（16k-32k，对齐 R-KV 的实验设置）。

与 step0_separability.py 同逻辑，但用行级探针 probe_rows（O(n_rows×L) 显存）：
- 探针行 = 全部 reflection 步位置 ∪ 等距采样行（stride）；
- cum_attn/concentration/G 集/错杀率 用等距采样行统计（无偏代表全序列）；
- receiver 反查用 reflection 行。

用法：PYTHONPATH=. python scripts/step0_separability_long.py \
        --trace-dir data/traces_long --model <本地权重> [--stride 32]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import yaml

from rkv.runner import load_model
from rkv.attn_probe_rows import probe_rows, receiver_lookup_row
from rkv.rescue.recurrence import topk_pages_per_step, compute_mri
from rkv.utils.features import FEATURE_NAMES, gather_labeled, page_features
from rkv.utils.reflection import mark_reflection_steps
from rkv.utils import scoring


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_trace_long(trace, tokenizer, model, cfg, stride: int):
    page_size = cfg.get("page_size", 16)
    prompt_ids = trace["prompt_ids"]
    gen_ids = trace["gen_ids"]
    prompt_len = prompt_ids.shape[0]
    full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)
    L = full.shape[1]

    refl_gen = mark_reflection_steps(
        gen_ids, tokenizer,
        step_entropy=trace.get("step_entropy"),
        entropy_quantile=cfg.get("entropy_quantile"),
    )
    if not refl_gen:
        return None
    refl_pos = [prompt_len + t for t in refl_gen if prompt_len + t < L]

    sample_pos = list(range(0, L, stride))
    rows = sorted(set(refl_pos) | set(sample_pos))
    page_attn = probe_rows(model, full, rows, page_size=page_size,
                           layers=cfg.get("probe_layers"))
    row_index = {p: i for i, p in enumerate(rows)}

    # R_t：reflection 行的 receiver 反查
    r_pages: set[int] = set()
    for p in refl_pos:
        r_pages |= receiver_lookup_row(
            page_attn[row_index[p]], p, page_size=page_size,
            receiver_topk=cfg.get("receiver_topk", 8),
            exclude_recent_pages=cfg.get("exclude_recent_pages", 4),
        )
    if not r_pages:
        return None

    # 统计行（等距采样）：MRI 基线 / G 集 / 错杀率 / cum_attn 都用它，保证无偏
    stat_idx = torch.tensor([row_index[p] for p in sample_pos])
    stat_attn = page_attn[stat_idx]                       # [n_stat, num_pages]
    mri = compute_mri(topk_pages_per_step(stat_attn, cfg.get("mri_topk", 32)))
    g_pages = set(mri.keys()) - r_pages

    feats = page_features(
        stat_attn, trace.get("step_entropy"), prompt_len=prompt_len,
        page_size=page_size, row_positions=torch.tensor(sample_pos), seq_len=L,
    )
    X, y = gather_labeled(feats, r_pages, g_pages)
    pages = [p for p in sorted(r_pages | g_pages) if p < feats.shape[0]]
    mri_col = [float(mri.get(p, 0)) for p in pages]

    evicted = scoring.evicted_by_budget(stat_attn.sum(0), cfg.get("keep_frac", 0.2))
    kr = scoring.kill_rate(r_pages, evicted)
    return X, y, mri_col, kr, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--trace-dir", default="data/traces_long")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--max-traces", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    cfg["attn_implementation"] = "sdpa"      # 行级探针不需要 eager，sdpa 省显存提速
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{args.trace_dir}/*.pt"))
    if args.max_traces:
        files = files[: args.max_traces]
    print(f"[separability-long] {len(files)} traces from {args.trace_dir}")

    model, tokenizer = load_model(cfg)

    all_X, all_y, all_mri, kills, lens = [], [], [], [], []
    used = 0
    for f in files:
        trace = torch.load(f, weights_only=False)
        res = extract_trace_long(trace, tokenizer, model, cfg, args.stride)
        if res is None:
            print(f"  skip {Path(f).name}")
            continue
        X, y, mri_col, kr, L = res
        all_X += X
        all_y += y
        all_mri += mri_col
        kills.append(kr)
        lens.append(L)
        used += 1
        print(f"  {Path(f).name}: L={L} n={len(y)} (R_t={sum(y)}) kill={kr:.2f}")

    if used == 0 or sum(all_y) == 0:
        print("无可用样本。")
        return

    clf = scoring.classifier_auc(all_X, all_y)
    feat_auc = scoring.per_feature_auc(all_X, all_y, FEATURE_NAMES)
    mri_auc = scoring.per_feature_auc([[m] for m in all_mri], all_y, ["mri"])["mri"]
    no_pos_cols = [FEATURE_NAMES.index(c) for c in ("entropy", "cum_attn", "concentration")]
    clf3 = scoring.classifier_auc([[x[j] for j in no_pos_cols] for x in all_X], all_y)
    mean_kill = sum(kills) / len(kills)
    scoring.plot_feature_dists(all_X, all_y, FEATURE_NAMES,
                               str(results_dir / "feature_cdf_long.png"))
    torch.save({"X": all_X, "y": all_y, "mri": all_mri, "feature_names": FEATURE_NAMES},
               results_dir / "separability_long_features.pt")

    summary = {
        "n_traces": used, "n_samples": len(all_y), "n_r": int(sum(all_y)),
        "mean_len": sum(lens) / len(lens), "max_len": max(lens), "stride": args.stride,
        "auc_4d": clf, "auc_3d_no_position": clf3,
        "per_feature_auc": feat_auc, "mri_baseline_auc": mri_auc,
        "mean_kill_rate": mean_kill,
    }
    (results_dir / "separability_long_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== 长 trace 可分离性结果 ===")
    print(f"trace: {used} 条, 平均长度 {summary['mean_len']:.0f}, 最长 {summary['max_len']}")
    print(f"样本: {len(all_y)} (R_t={sum(all_y)})")
    print(f"四维 AUC: {clf['auc_mean']:.3f} ± {clf['auc_std']:.3f}")
    print(f"去 position 三维 AUC: {clf3['auc_mean']:.3f} ± {clf3['auc_std']:.3f}")
    print("单特征 AUC:", {k: round(v, 3) if v is not None else None for k, v in feat_auc.items()})
    print(f"MRI 基线: {mri_auc:.3f}" if mri_auc is not None else "MRI 基线: N/A")
    print(f"平均错杀率: {mean_kill:.3f}")
    go = clf["auc_mean"] > 0.65 and clf3["auc_mean"] > 0.65
    print(f"\n判定: {'GO ✅ (长序列下可分性成立且不依赖 position)' if go else 'NO-GO ❌'}")


if __name__ == "__main__":
    main()
