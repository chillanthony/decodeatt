"""把 Step 0 的标准化逻辑回归导出成自包含的**原始空间线性打分器**。

离线 pipeline 是 impute(mean) -> StandardScaler -> LogisticRegression，
打分 = sum_j coef_j * (x_j - mean_j)/std_j + intercept。
合并成 score(x) = w_raw · x + b_raw，便于在线（解码时）逐页打分：
    w_raw_j = coef_j / std_j
    b_raw   = intercept - sum_j coef_j * mean_j / std_j
NaN 用训练集列均值插补（在线生成场景每页都有 entropy，不会触发）。

用法：PYTHONPATH=. python scripts/fit_scorer.py \
        --features results/separability_long_features.pt \
        --out results/signature_scorer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/separability_long_features.pt")
    ap.add_argument("--out", default="results/signature_scorer.json")
    args = ap.parse_args()

    d = torch.load(args.features, weights_only=False)
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["y"], dtype=int)
    names = d["feature_names"]

    imp = SimpleImputer(strategy="mean").fit(X)
    Xi = imp.transform(X)
    sc = StandardScaler().fit(Xi)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        sc.transform(Xi), y)

    coef = clf.coef_[0]
    mean, std = sc.mean_, sc.scale_
    w_raw = coef / std
    b_raw = float(clf.intercept_[0] - (coef * mean / std).sum())

    out = {
        "feature_names": names,
        "w_raw": w_raw.tolist(),
        "b_raw": b_raw,
        "impute_mean": imp.statistics_.tolist(),
        "note": "score(x)=w_raw·x+b_raw（原始特征空间）；分越高越像 R_t。",
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # 自检：原始空间打分与 sklearn pipeline 决策函数逐样本一致
    raw_score = Xi @ w_raw + b_raw
    ref = clf.decision_function(sc.transform(Xi))
    diff = np.abs(raw_score - ref).max()
    print(f"导出 {args.out}")
    print(f"w_raw={dict(zip(names, w_raw.round(4)))}  b_raw={b_raw:.4f}")
    print(f"与 pipeline 决策函数 max|diff|={diff:.2e}")
    assert diff < 1e-6, "原始空间线性化与 pipeline 不一致"
    print("OK")


if __name__ == "__main__":
    main()
