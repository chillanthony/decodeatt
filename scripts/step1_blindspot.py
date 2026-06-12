"""Step 1 Phase 1：盲区量化——主流淘汰/选择策略对 R_t vs G 的错杀率。

对每条长 trace：
1. 行级探针（reflection 行 ∪ 等距采样行），顺带算 Quest 精确界分数；
   探针输出缓存到 --cache-dir，二次分析免重算；
2. 抽取 R 事件（reflection 时刻 × receiver 页）与 G 事件（采样时刻 × 普通早期被访问页），
   两者用同一早期页排除规则；
3. 淘汰类策略（h2o/window/lra/recent/random）流式仿真（渐进淘汰、无复活）；
   选择类（quest/oracle）按事件行的分数取 top-B；
4. 汇总 策略 × 预算 → kill(R) vs kill(G)。

用法：PYTHONPATH=. python scripts/step1_blindspot.py \
        --trace-dir data/traces_long --model <本地权重> [--stride 32]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import yaml

from rkv import evictors
from rkv.attn_probe_rows import probe_rows, receiver_lookup_row
from rkv.utils.reflection import mark_reflection_steps

BUDGETS = [0.1, 0.2, 0.3]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def probe_trace(trace, tokenizer, model, cfg, stride: int, cache_path: Path):
    """探针（带缓存）：返回 dict(rows/sample_pos/refl_pos/page_attn/quest/prompt_len/L)。"""
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    page_size = cfg.get("page_size", 16)
    prompt_len = trace["prompt_ids"].shape[0]
    full = torch.cat([trace["prompt_ids"], trace["gen_ids"]]).unsqueeze(0)
    L = full.shape[1]

    refl_gen = mark_reflection_steps(
        trace["gen_ids"], tokenizer,
        step_entropy=trace.get("step_entropy"),
        entropy_quantile=cfg.get("entropy_quantile"),
    )
    if not refl_gen:
        return None
    refl_pos = [prompt_len + t for t in refl_gen if prompt_len + t < L]
    sample_pos = list(range(0, L, stride))
    rows = sorted(set(refl_pos) | set(sample_pos))

    page_attn, quest = probe_rows(
        model, full, rows, page_size=page_size,
        layers=cfg.get("probe_layers"), return_quest=True,
    )
    out = {"rows": rows, "sample_pos": sample_pos, "refl_pos": refl_pos,
           "page_attn": page_attn, "quest": quest,
           "prompt_len": prompt_len, "L": L}
    torch.save(out, cache_path)
    return out


def analyze_trace(pb, cfg):
    """单条 trace -> 各策略 × 预算的 (killed, total) 事件计数。"""
    page_size = cfg.get("page_size", 16)
    excl = cfg.get("exclude_recent_pages", 4)
    row_index = {p: i for i, p in enumerate(pb["rows"])}
    stat_idx = torch.tensor([row_index[p] for p in pb["sample_pos"]])
    stat_attn = pb["page_attn"][stat_idx]
    stat_quest = pb["quest"][stat_idx]
    stat_pos = pb["sample_pos"]

    # R：reflection 行的 receiver 反查
    r_events: list[tuple[int, set[int]]] = []
    r_pages_all: set[int] = set()
    for p in pb["refl_pos"]:
        pages = receiver_lookup_row(
            pb["page_attn"][row_index[p]], p, page_size=page_size,
            receiver_topk=cfg.get("receiver_topk", 8), exclude_recent_pages=excl,
        )
        if pages:
            r_events.append((p, pages))
            r_pages_all |= pages
    if not r_events:
        return None

    r_ev, g_ev = evictors.collect_events(
        stat_attn, stat_pos, r_events, r_pages_all, pb["prompt_len"],
        page_size=page_size, exclude_recent_pages=excl,
        access_topk=cfg.get("mri_topk", 32),
    )
    counts: dict[str, dict[float, dict]] = {}

    # 淘汰类：流式仿真
    for policy in evictors.EVICT_POLICIES:
        counts[policy] = {}
        for kf in BUDGETS:
            ab = evictors.simulate_eviction(
                stat_attn, stat_pos, policy, kf, page_size=page_size,
                protect_recent=excl,
            )
            kr, nr = evictors.event_kill_rate(r_ev, ab)
            kg, ng = evictors.event_kill_rate(g_ev, ab)
            counts[policy][kf] = {"kr": kr, "nr": nr, "kg": kg, "ng": ng}

    # 选择类：R 事件在真实 reflection 行上评估（回指那一刻的 query），G 在采样行
    for policy, mat_stat in (("quest", stat_quest), ("oracle", stat_attn)):
        counts[policy] = {}
        full_mat = pb["quest"] if policy == "quest" else pb["page_attn"]
        for kf in BUDGETS:
            kr = nr = 0
            for refl_pos, pages in r_events:
                sel = evictors.selection_set(
                    full_mat[row_index[refl_pos]], refl_pos, kf, page_size)
                nr += len(pages)
                kr += len(pages - sel)
            kg = ng = 0
            sel_cache: dict[int, set[int]] = {}
            for i, p in g_ev:
                if i not in sel_cache:
                    sel_cache[i] = evictors.selection_set(
                        mat_stat[i], stat_pos[i], kf, page_size)
                ng += 1
                kg += p not in sel_cache[i]
            counts[policy][kf] = {"kr": kr, "nr": nr, "kg": kg, "ng": ng}
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--trace-dir", default="data/traces_long")
    ap.add_argument("--cache-dir", default="data/probe_cache_long")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--max-traces", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    cfg["attn_implementation"] = "sdpa"
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{args.trace_dir}/*.pt"))
    if args.max_traces:
        files = files[: args.max_traces]
    print(f"[blindspot] {len(files)} traces from {args.trace_dir}")

    model = tokenizer = None
    pooled: dict[str, dict[float, dict]] = {}
    used, lens = 0, []
    for f in files:
        cache_path = cache_dir / (Path(f).stem + ".probe.pt")
        if not cache_path.exists() and model is None:
            from rkv.runner import load_model
            model, tokenizer = load_model(cfg)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
        trace = torch.load(f, weights_only=False)
        pb = probe_trace(trace, tokenizer, model, cfg, args.stride, cache_path)
        if pb is None:
            print(f"  skip {Path(f).name}（无 reflection）")
            continue
        counts = analyze_trace(pb, cfg)
        if counts is None:
            print(f"  skip {Path(f).name}（无 receiver 页）")
            continue
        for policy, by_kf in counts.items():
            for kf, c in by_kf.items():
                agg = pooled.setdefault(policy, {}).setdefault(
                    kf, {"kr": 0, "nr": 0, "kg": 0, "ng": 0})
                for k in agg:
                    agg[k] += c[k]
        used += 1
        lens.append(pb["L"])
        h2o = counts["h2o"][0.2]
        print(f"  {Path(f).name}: L={pb['L']} R事件={h2o['nr']} G事件={h2o['ng']} "
              f"h2o@0.2 kill(R)={h2o['kr']/max(h2o['nr'],1):.2f}")

    if not used:
        print("无可用 trace。")
        return

    summary = {"n_traces": used, "mean_len": sum(lens) / len(lens),
               "max_len": max(lens), "budgets": BUDGETS, "policies": {}}
    print(f"\n=== 盲区量化（{used} 条 trace，事件池化）===")
    print("| 策略 | 预算 | kill(R_t) | kill(G) | gap |")
    print("|------|------|-----------|---------|-----|")
    for policy in evictors.EVICT_POLICIES + evictors.SELECT_POLICIES:
        summary["policies"][policy] = {}
        for kf in BUDGETS:
            c = pooled[policy][kf]
            kr = c["kr"] / max(c["nr"], 1)
            kg = c["kg"] / max(c["ng"], 1)
            summary["policies"][policy][str(kf)] = {
                "kill_r": kr, "kill_g": kg, "gap": kr - kg,
                "n_r_events": c["nr"], "n_g_events": c["ng"],
            }
            print(f"| {policy} | {kf} | {kr:.3f} | {kg:.3f} | {kr - kg:+.3f} |")

    (results_dir / "blindspot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n已写 {results_dir / 'blindspot_summary.json'}")


if __name__ == "__main__":
    main()
