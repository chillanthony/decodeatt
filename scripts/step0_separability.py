"""可分离性诊断（对接 03-SCRL升级版 的可分离性命题）。

Step 0 续：MRI 被证伪（04-rescuekv-假设验证实验.md NO-GO）后的信号重定位——
把 MRI 换成四维签名 φ = (熵, 累计注意力, 位置, receiver 集中度)，
测 R_t（纠错回指 page）与 G（普通被访问 page）是否可分。

对每条 trace：标 reflection 步 -> 探针 -> 反查 R_t -> page_features 抽 φ；
跨 trace 汇总 (X, y) -> 逻辑回归 5 折交叉验证 AUC + 每维单特征 AUC（含 MRI 作基线对照）。

GO：四维 AUC > 0.65（明显强于随机 0.5 且强于 MRI 基线）。

用法：uv run python scripts/step0_separability.py --config configs/default.yaml [--model 本地路径]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import yaml

from rkv.runner import load_model
from rkv.attn_probe import probe_sequence, receiver_lookup
from rkv.rescue.recurrence import topk_pages_per_step, compute_mri
from rkv.utils.features import FEATURE_NAMES, gather_labeled, page_features
from rkv.utils.reflection import mark_reflection_steps
from rkv.utils import scoring


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_trace(trace, tokenizer, model, cfg):
    """单条 trace -> (X, y, mri_col)。无 reflection 步 / R_t 为空则返回 None。

    X: [n_samples, 4] 特征；y: 1=R_t / 0=G；mri_col: 各样本的 MRI（作单特征基线对照）。
    """
    page_size = cfg.get("page_size", 16)
    prompt_ids = trace["prompt_ids"]
    gen_ids = trace["gen_ids"]
    prompt_len = prompt_ids.shape[0]
    full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)

    refl_gen = mark_reflection_steps(
        gen_ids, tokenizer,
        step_entropy=trace.get("step_entropy"),
        entropy_quantile=cfg.get("entropy_quantile"),
    )
    if not refl_gen:
        return None

    page_attn = probe_sequence(
        model, full, page_size=page_size,
        layers=cfg.get("probe_layers"), max_len=cfg.get("probe_max_len"),
    )
    L = page_attn.shape[0]

    # R_t：所有 reflection 步反查到的早期 page（绝对位置 = prompt_len + gen_step）
    r_pages: set[int] = set()
    for t in refl_gen:
        pos = prompt_len + t
        if pos >= L:
            continue
        r_pages |= receiver_lookup(
            page_attn, pos, page_size=page_size,
            receiver_topk=cfg.get("receiver_topk", 8),
            exclude_recent_pages=cfg.get("exclude_recent_pages", 4),
        )
    if not r_pages:
        return None

    mri = compute_mri(topk_pages_per_step(page_attn, cfg.get("mri_topk", 32)))
    g_pages = set(mri.keys()) - r_pages

    feats = page_features(page_attn.cpu(), trace.get("step_entropy"),
                          prompt_len=prompt_len, page_size=page_size)
    X, y = gather_labeled(feats, r_pages, g_pages)
    pages = [p for p in sorted(r_pages | g_pages) if p < feats.shape[0]]
    mri_col = [float(mri.get(p, 0)) for p in pages]
    return X, y, mri_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="覆盖 config 的模型路径（用本地权重）")
    ap.add_argument("--max-traces", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{cfg.get('trace_dir', 'data/traces')}/*.pt"))
    if args.max_traces:
        files = files[: args.max_traces]
    print(f"[separability] {len(files)} traces")

    model, tokenizer = load_model(cfg)

    all_X, all_y, all_mri = [], [], []
    used = 0
    for f in files:
        trace = torch.load(f, weights_only=False)
        res = extract_trace(trace, tokenizer, model, cfg)
        if res is None:
            print(f"  skip {Path(f).name}")
            continue
        X, y, mri_col = res
        all_X += X
        all_y += y
        all_mri += mri_col
        used += 1
        print(f"  {Path(f).name}: n={len(y)} (R_t={sum(y)})")

    if used == 0 or sum(all_y) == 0:
        print("无可用样本。")
        return

    clf = scoring.classifier_auc(all_X, all_y)
    feat_auc = scoring.per_feature_auc(all_X, all_y, FEATURE_NAMES)
    # MRI 单特征基线（被证伪的旧信号，作对照）
    mri_auc = scoring.per_feature_auc(
        [[m] for m in all_mri], all_y, ["mri"])["mri"]
    scoring.plot_feature_dists(all_X, all_y, FEATURE_NAMES,
                               str(results_dir / "feature_cdf.png"))

    summary = {
        "n_traces": used,
        "n_samples": len(all_y),
        "n_r": int(sum(all_y)),
        "auc_4d": clf,
        "per_feature_auc": feat_auc,
        "mri_baseline_auc": mri_auc,
        "feature_names": FEATURE_NAMES,
    }
    (results_dir / "separability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== 可分离性诊断结果 ===")
    print(f"样本: {len(all_y)} (R_t={sum(all_y)}, G={len(all_y) - sum(all_y)}), {used} traces")
    print(f"四维分类器 AUC: {clf['auc_mean']:.3f} ± {clf['auc_std']:.3f}")
    print(f"标准化系数 {dict(zip(FEATURE_NAMES, [round(c, 3) for c in clf['coef']]))}")
    print("单特征 AUC:", {k: round(v, 3) if v is not None else None
                          for k, v in feat_auc.items()})
    print(f"MRI 基线 AUC: {mri_auc:.3f}" if mri_auc is not None else "MRI 基线 AUC: N/A")

    go = clf["auc_mean"] > 0.65
    print(f"\n判定: {'GO ✅ (R_t 可分，签名有效)' if go else 'NO-GO ❌ (四维签名不可分)'}")
    print(f"产物已存到 {results_dir}/")


if __name__ == "__main__":
    main()
