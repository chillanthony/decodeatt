"""Step 0 诊断指标：MRI 分布对比（KS 检验/CDF）、淘汰错杀率、四维特征可分离性（AUC）。"""
from __future__ import annotations

import statistics


def mri_distribution_test(r_mri: list[int], g_mri: list[int]) -> dict:
    """R_t vs G 的 MRI 分布检验：KS 统计量 + p 值 + 中位数。"""
    from scipy.stats import ks_2samp

    if not r_mri or not g_mri:
        return {"n_r": len(r_mri), "n_g": len(g_mri), "ks": None, "p": None,
                "median_r": None, "median_g": None}
    ks, p = ks_2samp(r_mri, g_mri, alternative="less")  # H1: R 的 CDF 偏右(MRI 更大)
    return {
        "n_r": len(r_mri),
        "n_g": len(g_mri),
        "ks": float(ks),
        "p": float(p),
        "median_r": float(statistics.median(r_mri)),
        "median_g": float(statistics.median(g_mri)),
    }


def kill_rate(target_pages: set[int], evicted_pages: set[int]) -> float:
    """target_pages 中被淘汰器丢弃的比例（错杀率）。"""
    if not target_pages:
        return float("nan")
    return len(target_pages & evicted_pages) / len(target_pages)


def evicted_by_budget(total_attn, keep_frac: float = 0.2) -> set[int]:
    """通用淘汰器代理：按累计注意力排序，保留 top keep_frac，其余视为被淘汰。"""
    import torch

    num_pages = total_attn.shape[0]
    keep = max(1, int(num_pages * keep_frac))
    kept = set(torch.topk(total_attn, keep).indices.tolist())
    return set(range(num_pages)) - kept


def classifier_auc(X, y, n_splits: int = 5, seed: int = 0) -> dict:
    """四维特征 -> R_t/G 的可分离性：均值插补 + 标准化 + 逻辑回归，分层 K 折交叉验证 AUC。

    返回 {auc_mean, auc_std, fold_aucs, coef}；coef 是全量数据拟合后的标准化系数
    （符号/大小可读成各特征的判别方向与贡献）。
    """
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    pipe = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X, y)
    coef = pipe.named_steps["logisticregression"].coef_[0].tolist()
    return {
        "auc_mean": float(scores.mean()),
        "auc_std": float(scores.std()),
        "fold_aucs": [float(s) for s in scores],
        "coef": coef,
    }


def per_feature_auc(X, y, feature_names: list[str]) -> dict:
    """每个特征单独的判别力 AUC（NaN 样本丢弃）。AUC<0.5 表示方向相反（值越小越像 R_t）。"""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    out = {}
    for j, name in enumerate(feature_names):
        col = X[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 10 or len(set(y[ok])) < 2:
            out[name] = None
            continue
        out[name] = float(roc_auc_score(y[ok], col[ok]))
    return out


def plot_feature_dists(X, y, feature_names: list[str], path: str):
    """每个特征画 R_t vs G 的经验 CDF（2x2 子图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(feature_names)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(9, 7))
    for j, (name, ax) in enumerate(zip(feature_names, axes.flat)):
        for mask, label in [(y == 1, "R_t (correction)"), (y == 0, "G (ordinary)")]:
            col = X[mask, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            xs = np.sort(col)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", label=f"{label} (n={len(xs)})")
        if name == "concentration":
            ax.set_xscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("CDF")
        ax.legend(fontsize=7)
    fig.suptitle("Per-feature CDF: R_t vs G")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mri_cdf(r_mri: list[int], g_mri: list[int], path: str):
    """画 R_t vs G 的 MRI 经验 CDF。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 4))
    for data, label in [(r_mri, "R_t (correction)"), (g_mri, "G (ordinary)")]:
        if not data:
            continue
        xs = np.sort(np.array(data, dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", label=f"{label} (n={len(xs)})")
    ax.set_xlabel("MRI")
    ax.set_ylabel("CDF")
    ax.set_title("MRI distribution: R_t vs G")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
