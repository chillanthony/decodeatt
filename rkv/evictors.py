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
    """返回 alive_before,共 n_stat+1 项:第 i 项 = **处理 stat 行 i 之前**的
    存活页集合;最后一项 = 处理完全部行后的终态。

    事件评估:G 事件(行 i 的访问)用 alive_before[i];R 事件(reflection 位置 p)
    用 alive_before[bisect_left(stat_pos, p)]——恰为 p 时刻已生效的淘汰状态。
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

    alive_before.append(set(alive))          # 终态(R 事件可能落在最后一行之后)
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

    - R 事件:每个 reflection (位置 refl_pos, receiver 集),映射到
      bisect_left(stat_pos, refl_pos)——该下标的 alive_before 恰为回指那一刻
      已生效的淘汰状态(之前所有 stat 行的淘汰都已发生);
    - G 事件:生成区 stat 行的真实注意力 top-k 中,**早期**(排除最近
      exclude_recent_pages 页)且不属于任何 R_t 的页。
    """
    r_out: list[tuple[int, int]] = []
    for refl_pos, pages in r_events:
        i = bisect.bisect_left(stat_pos, refl_pos)
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
    """合成数据手验(CPU)。场景:page_size=4,两个 stat 行在位置 15/19,
    即行 0 时页 0-3 已全部存在、行 1 时页 4 入场;keep_frac=0.75 -> 每行恰好
    淘汰 1 页且可淘汰集 >1,策略的选择差异可被断言(此前的场景预算太紧,
    保护页挤占后只剩唯一候选,策略无选择空间,断言无意义)。"""
    stat_pos = [15, 19]
    attn = torch.tensor([
        # p0    p1    p2   p3   p4
        [0.50, 0.01, 0.20, 0.30, 0.00],   # 行0(页4尚未因果存在)
        [0.40, 0.00, 0.10, 0.20, 0.30],   # 行1
    ])
    kw = dict(page_size=4, protect_recent=0, access_topk=2)
    ab_h2o = simulate_eviction(attn, stat_pos, "h2o", 0.75, **kw)
    ab_rec = simulate_eviction(attn, stat_pos, "recent", 0.75, **kw)
    ab_lra = simulate_eviction(attn, stat_pos, "lra", 0.75, **kw)

    # 行0前:4 页全活(淘汰发生在吸收行0之后);返回 n_stat+1=3 项(含终态)
    assert len(ab_h2o) == 3 and ab_h2o[0] == {0, 1, 2, 3}, ab_h2o
    # 行0后淘汰 1 页(预算 int(4*0.75)=3,p3 是当前页受保护):
    #   h2o 按累计 [0.5,0.01,0.2] 淘汰 p1;recent 淘汰最老 p0;
    #   lra 的 top-2 访问 = {p0,p3},p1/p2 都未被访问,先淘汰 p1(稳定排序)。
    assert ab_h2o[1] == {0, 2, 3, 4}, ab_h2o[1]
    assert ab_rec[1] == {1, 2, 3, 4}, ab_rec[1]
    assert ab_lra[1] == {0, 2, 3, 4}, ab_lra[1]
    # 行1后(终态)再淘汰 1 页:h2o 淘汰累计最低的 p2;recent 淘汰 p1;lra 淘汰未访问的 p2
    assert ab_h2o[2] == {0, 3, 4}, ab_h2o[2]
    assert ab_rec[2] == {2, 3, 4}, ab_rec[2]
    assert ab_lra[2] == {0, 3, 4}, ab_lra[2]
    # 无复活
    for ab in (ab_h2o, ab_rec, ab_lra):
        gone: set[int] = set()
        for i in range(1, len(ab)):
            gone |= ab[i - 1] - ab[i]
            assert not (gone & ab[i]), (i, gone & ab[i])
    # random 可复现
    assert (simulate_eviction(attn, stat_pos, "random", 0.75, seed=1, **kw)
            == simulate_eviction(attn, stat_pos, "random", 0.75, seed=1, **kw))

    # 事件抽取:reflection 在位置 16 -> bisect_left([15,19],16)=1,
    # 即评估"行0 的淘汰已生效"的状态,receiver={0};
    # G 事件取每行 top-3 中的早期页(excl=0 -> cutoff=当前页),排除 R 页
    r_ev, g_ev = collect_events(
        attn, stat_pos, r_events=[(16, {0})], r_pages_all={0},
        prompt_len=8, page_size=4, exclude_recent_pages=0, access_topk=3,
    )
    assert r_ev == [(1, 0)], r_ev
    assert g_ev == [(0, 2), (1, 3)], g_ev

    k, n = event_kill_rate([(1, 0), (1, 1)], ab_rec)
    assert (k, n) == (1, 2), (k, n)        # recent 已杀 p0,p1 还活着

    # selection:行1 的 top-2 = {p0, p4}
    sel = selection_set(attn[1], stat_pos[1], keep_frac=0.4, page_size=4)
    assert sel == {0, 4}, sel
    print("OK: evictors 仿真/事件/选择 与手算一致")


if __name__ == "__main__":
    _selftest()
