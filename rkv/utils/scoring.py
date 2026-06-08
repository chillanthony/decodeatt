"""Step 0 诊断指标：MRI 分布对比（KS 检验/CDF）、淘汰错杀率。"""
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
