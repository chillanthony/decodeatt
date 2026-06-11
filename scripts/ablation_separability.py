"""A1 消融：特征子集的可分离性 AUC（堵 position 构造偏置的口）。

输入：step0_separability.py --save-features 存的 results/separability_features.pt。
离线跑（CPU 即可，不需模型/GPU）。

子集设计：
- 4d 全量（基准）
- 去单维 leave-one-out（重点：去 position —— receiver 反查屏蔽近窗导致 R_t 机制性偏早，
  position 维有构造偏置；去掉后 AUC 仍 >0.65 才能声称可分性不是偏置撑的）
- entropy+concentration 二维（纯"内容+访问模式"，完全无位置信息）
- entropy 单维 / +MRI 五维（MRI 基线对照）

用法：uv run python scripts/ablation_separability.py [--features results/separability_features.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rkv.utils.scoring import classifier_auc


def subset_auc(X, y, names: list[str], cols: list[str]) -> dict:
    idx = [names.index(c) for c in cols]
    res = classifier_auc(X[:, idx], y)
    return {"features": cols, "auc": round(res["auc_mean"], 4),
            "std": round(res["auc_std"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/separability_features.pt")
    ap.add_argument("--out", default="results/ablation_summary.json")
    args = ap.parse_args()

    blob = torch.load(args.features, weights_only=False)
    X = np.asarray(blob["X"], dtype=float)
    y = np.asarray(blob["y"], dtype=int)
    names = list(blob["feature_names"])
    # MRI 作为可选第五维拼上（基线对照）
    X5 = np.concatenate([X, np.asarray(blob["mri"], dtype=float)[:, None]], axis=1)
    names5 = names + ["mri"]

    runs = [
        ["entropy", "cum_attn", "position", "concentration"],          # 4d 基准
        ["entropy", "cum_attn", "concentration"],                      # 去 position（关键）
        ["cum_attn", "position", "concentration"],                     # 去 entropy
        ["entropy", "position", "concentration"],                      # 去 cum_attn
        ["entropy", "cum_attn", "position"],                           # 去 concentration
        ["entropy", "concentration"],                                  # 无位置二维
        ["entropy"],                                                   # 最强单维
        ["entropy", "cum_attn", "position", "concentration", "mri"],   # +MRI 五维
    ]
    results = [subset_auc(X5, y, names5, cols) for cols in runs]

    print(f"n={len(y)} (R_t={int(y.sum())})")
    for r in results:
        print(f"  AUC {r['auc']:.3f} ± {r['std']:.3f}  <- {'+'.join(r['features'])}")

    no_pos = next(r for r in results if r["features"] == ["entropy", "cum_attn", "concentration"])
    verdict = no_pos["auc"] > 0.65
    print(f"\n去 position 三维 AUC = {no_pos['auc']:.3f} -> "
          f"{'可分性不依赖构造偏置 ✅' if verdict else '可分性主要由 position 偏置贡献 ❌'}")

    Path(args.out).write_text(json.dumps(
        {"n": len(y), "n_r": int(y.sum()), "runs": results,
         "no_position_ok": bool(verdict)}, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
