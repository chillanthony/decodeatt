"""可分离性命题（03-SCRL升级版）：每 page 的四维特征签名 φ(p)。

替代 MRI 单信号——对每个 page 算 φ = (熵, 累计注意力, 位置, receiver 集中度)，
供分类器测试 R_t（纠错回指 page）与 G（普通被访问 page）是否线性可分。

特征定义（均在 page 粒度）：
- entropy        : 该 page 内**生成区** token 的平均预测熵（来自 trace 的 step_entropy）。
                   纯 prompt page 没有熵 -> NaN（下游用列均值插补）。
- cum_attn       : 该 page 被访问的平均注意力 = page_attn[start:, p].mean()，
                   只在因果有效步（q >= page 首 token 位置）上取均值，去掉位置带来的
                   "晚出现的 page 可被访问的步天然更少"的偏置。
- position       : page 下标 / num_pages（归一化位置）。
- concentration  : 峰均比 = max / (mean + eps)，刻画"注意力是尖峰复访型还是均匀型"。

坐标约定：page_attn 来自 attn_probe.probe_sequence，[L_query, num_pages]，
全序列（prompt+gen）坐标；step_entropy 是 gen 坐标，第 t 步 = 全序列位置 prompt_len + t。
"""
from __future__ import annotations

import torch

FEATURE_NAMES = ["entropy", "cum_attn", "position", "concentration"]
_EPS = 1e-12


@torch.no_grad()
def page_features(
    page_attn: torch.Tensor,
    step_entropy: torch.Tensor | None,
    prompt_len: int,
    page_size: int = 16,
) -> torch.Tensor:
    """返回 [num_pages, 4] 的特征矩阵（列序 = FEATURE_NAMES）。

    纯 prompt page（无生成 token 落在该 page）的 entropy 列为 NaN。
    """
    L, num_pages = page_attn.shape
    feats = torch.full((num_pages, len(FEATURE_NAMES)), float("nan"))

    ent = None
    if step_entropy is not None:
        ent = step_entropy.float() if isinstance(step_entropy, torch.Tensor) \
            else torch.tensor(step_entropy, dtype=torch.float32)

    for p in range(num_pages):
        start = p * page_size          # 该 page 首 token 的全序列位置
        if start >= L:
            break

        # entropy：page 内生成区 token 的平均熵（gen 步 t 的全序列位置 = prompt_len + t）
        if ent is not None and ent.numel() > 0:
            lo = max(start, prompt_len) - prompt_len           # gen 坐标
            hi = min(start + page_size, L) - prompt_len
            lo, hi = max(lo, 0), min(hi, ent.numel())
            if hi > lo:
                feats[p, 0] = ent[lo:hi].mean()

        # cum_attn / concentration：只看因果有效的查询步
        col = page_attn[start:, p].float()
        if col.numel() > 0:
            mean = col.mean()
            feats[p, 1] = mean
            feats[p, 3] = col.max() / (mean + _EPS)

        feats[p, 2] = p / max(num_pages, 1)

    return feats


def gather_labeled(
    feats: torch.Tensor, r_pages: set[int], g_pages: set[int]
) -> tuple[list[list[float]], list[int]]:
    """从单条 trace 的特征矩阵里取 R_t(=1) / G(=0) 两类样本，返回 (X, y) 列表。"""
    X, y = [], []
    for p in sorted(r_pages | g_pages):
        if p >= feats.shape[0]:
            continue
        X.append(feats[p].tolist())
        y.append(1 if p in r_pages else 0)
    return X, y


def _selftest():
    """合成数据手验（不需 GPU/模型）：4 page、page_size=2、prompt_len=3、L=8。"""
    page_attn = torch.tensor([
        # p0    p1    p2    p3
        [1.0, 0.0, 0.0, 0.0],   # q0
        [0.5, 0.5, 0.0, 0.0],   # q1
        [0.2, 0.8, 0.0, 0.0],   # q2
        [0.1, 0.1, 0.8, 0.0],   # q3
        [0.7, 0.1, 0.2, 0.0],   # q4
        [0.1, 0.1, 0.2, 0.6],   # q5
        [0.4, 0.2, 0.2, 0.2],   # q6
        [0.1, 0.1, 0.1, 0.7],   # q7
    ])
    ent = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # gen 步 0..4 (全序列位置 3..7)
    f = page_features(page_attn, ent, prompt_len=3, page_size=2)

    # p0(token0-1): 纯 prompt -> entropy NaN；cum=col0[0:].mean()
    assert torch.isnan(f[0, 0]), f[0]
    # p1(token2-3): token3 是 gen 步 0 -> entropy=1.0
    assert abs(f[1, 0] - 1.0) < 1e-6, f[1]
    # p2(token4-5): gen 步 1,2 -> mean(2,3)=2.5
    assert abs(f[2, 0] - 2.5) < 1e-6, f[2]
    # p3(token6-7): gen 步 3,4 -> mean(4,5)=4.5；cum=mean(0.2,0.7)=0.45
    assert abs(f[3, 0] - 4.5) < 1e-6 and abs(f[3, 1] - 0.45) < 1e-6, f[3]
    # position 单调
    assert f[0, 2] < f[1, 2] < f[2, 2] < f[3, 2]
    # p0 的峰均比 = 1.0 / mean(col0)
    expect = 1.0 / page_attn[:, 0].mean()
    assert abs(f[0, 3] - expect) < 1e-4, (f[0, 3], expect)

    X, y = gather_labeled(f, r_pages={1}, g_pages={0, 2})
    assert y == [0, 1, 0] and len(X) == 3
    print("OK: page_features / gather_labeled 与手算一致")
    print(f)


if __name__ == "__main__":
    _selftest()
