"""信号 1：复发周期 MRI（Maximum Recurrence Interval）。

离线版（Step 0 诊断用）：
- topk_pages_per_step：从 probe 的 [L_query, num_pages] 取每步 Top-mri_topk 的 page。
- access_steps：每个 page 被"算重要"的步序列。
- compute_mri：MRI(p) = 该步序列中**最大相邻间隔**（冷落最久后又复发 -> MRI 大）。

直觉：
  page 在步 [5,6,7] 后再没出现 -> MRI=1（局部重要后即丢，非复发）
  page 在步 [5,6, 500,501] -> gaps=[1,494,1] -> MRI=494（沉睡后复发，正是要保护的）
  page 只出现 1 次 -> MRI=0

在线版（Step 1 运行时增量更新）留待后续。
"""
from __future__ import annotations


def topk_pages_per_step(page_attn, mri_topk: int = 32) -> list[list[int]]:
    """对每个查询步取 Top-mri_topk 的 page（只在有正注意力的因果有效 page 里取）。"""
    import torch

    result: list[list[int]] = []
    for q in range(page_attn.shape[0]):
        row = page_attn[q]
        valid = int((row > 0).sum().item())
        k = min(mri_topk, valid)
        if k <= 0:
            result.append([])
            continue
        idx = torch.topk(row, k).indices.tolist()
        result.append(idx)
    return result


def access_steps(topk_per_step: list[list[int]]) -> dict[int, list[int]]:
    """page -> 它被判为重要的步序列（升序）。"""
    acc: dict[int, list[int]] = {}
    for step, pages in enumerate(topk_per_step):
        for p in pages:
            acc.setdefault(p, []).append(step)
    return acc


def compute_mri(topk_per_step: list[list[int]]) -> dict[int, int]:
    """返回 page -> MRI（最大相邻访问间隔；<2 次访问记 0）。"""
    acc = access_steps(topk_per_step)
    mri: dict[int, int] = {}
    for p, steps in acc.items():
        if len(steps) < 2:
            mri[p] = 0
        else:
            mri[p] = max(steps[i + 1] - steps[i] for i in range(len(steps) - 1))
    return mri


def page_stats(topk_per_step: list[list[int]]) -> dict[int, dict]:
    """每 page 的诊断信息：访问次数 / 首末步 / MRI（供 Step 5 对比 R_t vs G）。"""
    acc = access_steps(topk_per_step)
    mri = compute_mri(topk_per_step)
    return {
        p: {
            "count": len(steps),
            "first": steps[0],
            "last": steps[-1],
            "mri": mri[p],
        }
        for p, steps in acc.items()
    }


def _selftest():
    """合成序列手验 MRI（不需 GPU/模型）。"""
    # 步:    0      1      2        3      4
    topk = [[0, 1], [0, 2], [1],     [],    [0, 3]]
    # page0 出现于步 [0,1,4] -> gaps [1,3] -> MRI 3
    # page1 出现于步 [0,2]   -> gaps [2]   -> MRI 2
    # page2 出现于步 [1]     -> 1 次       -> MRI 0
    # page3 出现于步 [4]     -> 1 次       -> MRI 0
    acc = access_steps(topk)
    mri = compute_mri(topk)
    print("access_steps:", dict(sorted(acc.items())))
    print("MRI:", dict(sorted(mri.items())))
    expect = {0: 3, 1: 2, 2: 0, 3: 0}
    assert mri == expect, f"FAIL: {mri} != {expect}"
    print("OK: MRI 与手算一致", expect)


if __name__ == "__main__":
    _selftest()
