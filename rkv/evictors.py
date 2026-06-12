"""Step 1 盲区量化:页级淘汰器仿真(渐进淘汰、无复活)。

回答的问题:在"页被需要的那一刻"(R = 纠错回指 / G = 普通回看),主流淘汰/选择
策略是否已经把它丢了?——即 03-SCRL升级版 的"系统性错杀"主结果。

仿真模型(忠实于真实系统的关键约束):
- 流式:沿等距采样行(stat 行)推进,每步先记录"此刻的存活集",再吸收该行的注意力;
- 渐进淘汰、**无复活**:被淘汰的页永久消失(真实 H2O/SnapKV 都不回收);
- 预算随序列增长:budget(t) = max(1, keep_frac × 当前因果有效页数);
- 所有淘汰策略始终豁免最近 protect_recent 页(真实系统都保 recent window)。

选择类(quest/oracle)单独处理:不淘汰,但每步只有 top-B 页参与计算——
"被需要却没被选中"同样是盲区。
"""
from __future__ import annotations

import bisect
import random

import torch

EVICT_POLICIES = ["h2o", "window", "lra", "recent", "random"]
SELECT_POLICIES = ["quest", "oracle"]


@torch.no_grad()
def simulate_eviction(
    stat_attn: torch.Tensor,
    stat_pos: list[int],
    policy: str,
    keep_frac: float,
    page_size: int = 16,
    protect_recent: int = 4,
    window: int = 8,
    access_topk: int = 32,
    seed: int = 0,
) -> list[set[int]]:
    """返回 alive_before:第 i 项 = **处理 stat 行 i 之前**的存活页集合。

    在事件评估时用 alive_before[i]:第 i 步的访问是"需要",需要时页还在才算活。
    """
    assert policy in EVICT_POLICIES, policy
    n_stat, num_pages = stat_attn.shape
    rng = random.Random(seed)

    cum = torch.zeros(num_pages)
    recent_rows: list[torch.Tensor] = []          # window 策略的观察窗
    last_access = {}                              # lra:页 -> 最近被 top-k 访问的步
    rand_score = {}                               # random:页 -> 固定随机分
    alive: set[int] = set()
    seen_pages = 0
    alive_before: list[set[int]] = []

    for i in range(n_stat):
        pos = stat_pos[i]
        cur_page = pos // page_size
        # 新页入场(因果上已存在的页)
        while seen_pages <= cur_page:
            alive.add(seen_pages)
            last_access.setdefault(seen_pages, -1)
            rand_score[seen_pages] = rng.random()
            seen_pages += 1

        alive_before.append(set(alive))

        # 吸收该行注意力(更新各策略的状态)
        row = stat_attn[i].float()
        cum += row
        recent_rows.append(row)
        if len(recent_rows) > window:
            recent_rows.pop(0)
        k = min(access_topk, int((row > 0).sum().item()))
        if k > 0:
            for p in torch.topk(row, k).indices.tolist():
                last_access[p] = i

        # 预算检查 + 渐进淘汰(永久)
        budget = max(1, int((cur_page + 1) * keep_frac))
        protected = {p for p in alive if p >= cur_page - protect_recent}
        evictable = alive - protected
        n_evict = len(alive) - budget
        if n_evict > 0 and evictable:
            if policy == "h2o":
                score = {p: cum[p].item() for p in evictable}
            elif policy == "window":
                win = torch.stack(recent_rows).sum(0)
                score = {p: win[p].item() for p in evictable}
            elif policy == "lra":
                score = {p: last_access[p] for p in evictable}
            elif policy == "recent":
                score = {p: p for p in evictable}
            else:  # random
                score = {p: rand_score[p] for p in evictable}
            for p in sorted(evictable, key=lambda p: score[p])[:n_evict]:
                alive.discard(p)

    return alive_before


@torch.no_grad()
def selection_set(
    score_row: torch.Tensor, pos: int, keep_frac: float,
    page_size: int = 16,
) -> set[int]:
    """选择类策略(quest/oracle):按该行分数取 top-B(B = keep_frac × 因果有效页数)。"""
    cur_page = pos // page_size
    n_valid = cur_page + 1
    budget = max(1, int(n_valid * keep_frac))
    row = score_row[:n_valid].float()
    k = min(budget, n_valid)
    return set(torch.topk(row, k).indices.tolist())


def collect_events(
    stat_attn: torch.Tensor,
    stat_pos: list[int],
    r_events: list[tuple[int, set[int]]],
    r_pages_all: set[int],
    prompt_len: int,
    page_size: int = 16,
    exclude_recent_pages: int = 4,
    access_topk: int = 32,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """抽取 (stat 步下标, 页) 事件,R/G 用同一早期页排除规则保证可比。

    - R 事件:每个 reflection (位置 refl_pos, receiver 集),映射到最后一个
      stat_pos <= refl_pos 的步(回指那一刻的缓存状态);
    - G 事件:生成区 stat 行的真实注意力 top-k 中,**早期**(排除最近
      exclude_recent_pages 页)且不属于任何 R_t 的页。
    """
    r_out: list[tuple[int, int]] = []
    for refl_pos, pages in r_events:
        i = bisect.bisect_right(stat_pos, refl_pos) - 1
        if i < 0:
            continue
        for p in pages:
            r_out.append((i, p))

    g_out: list[tuple[int, int]] = []
    for i, pos in enumerate(stat_pos):
        if pos < prompt_len:
            continue
        cutoff = pos // page_size - exclude_recent_pages
        if cutoff <= 0:
            continue
        row = stat_attn[i]
        k = min(access_topk, int((row > 0).sum().item()))
        if k <= 0:
            continue
        for p in torch.topk(row, k).indices.tolist():
            if p < cutoff and p not in r_pages_all:
                g_out.append((i, p))
    return r_out, g_out


def event_kill_rate(events: list[tuple[int, int]], alive_before: list[set[int]]) -> tuple[int, int]:
    """返回 (killed, total):事件发生时页已不存活的计数。"""
    killed = sum(1 for i, p in events if p not in alive_before[i])
    return killed, len(events)


def _selftest():
    """合成数据手验(CPU):3 页预算下 h2o 淘汰低累计页、recent 淘汰最老页、无复活。"""
    # 8 个 stat 行,page_size=4 -> 行 i 位置 = 4*i+3,页 i 在行 i 入场;共 8 页
    num_pages = 8
    stat_pos = [4 * i + 3 for i in range(8)]
    attn = torch.zeros(8, num_pages)
    for i in range(8):
        attn[i, : i + 1] = 0.01
        attn[i, i] = 0.5            # 当前页自注意力高
        attn[i, 0] = 0.4            # 页 0 一直被重看(高累计)
    # keep_frac=0.5, protect_recent=1 -> 行 i:页数 i+1,预算 max(1,(i+1)//2)
    ab_h2o = simulate_eviction(attn, stat_pos, "h2o", 0.5, page_size=4, protect_recent=1)
    ab_rec = simulate_eviction(attn, stat_pos, "recent", 0.5, page_size=4, protect_recent=1)

    # 行 7 之前:h2o 应保住高累计的页 0;recent 应已把页 0 淘汰(只保新)
    assert 0 in ab_h2o[7], f"h2o 不应淘汰高累计页: {ab_h2o[7]}"
    assert 0 not in ab_rec[7], f"recent 应淘汰最老页: {ab_rec[7]}"
    # 无复活:一旦消失不再回来
    for ab in (ab_h2o, ab_rec):
        gone: set[int] = set()
        for i in range(1, 8):
            gone |= ab[i - 1] - ab[i]
            assert not (gone & ab[i]), f"复活了: step{i} {gone & ab[i]}"
    # 预算约束:存活数 <= max(budget, protected数+budget) 粗检
    assert len(ab_h2o[7]) <= 4, ab_h2o[7]

    # 事件抽取:reflection 在位置 19(行 4 时刻),receiver={0,1};页 0 被 G 大量访问
    r_ev, g_ev = collect_events(
        attn, stat_pos, r_events=[(19, {0, 1})], r_pages_all={0, 1},
        prompt_len=8, page_size=4, exclude_recent_pages=1, access_topk=2,
    )
    assert (4, 0) in r_ev and (4, 1) in r_ev, r_ev
    assert all(p not in (0, 1) for _, p in g_ev), g_ev  # G 排除 R 页

    k, n = event_kill_rate([(7, 0), (7, 5)], ab_rec)
    assert n == 2 and k >= 1, (k, n)  # 页 0 在 recent 策略下死了

    # selection:oracle 用真实行,页 0 注意力高应被选中
    sel = selection_set(attn[7], stat_pos[7], keep_frac=0.3, page_size=4)
    assert 0 in sel and 7 in sel, sel
    print("OK: evictors 仿真/事件/选择 与手算一致")


if __name__ == "__main__":
    _selftest()
