"""Step 2 T3:因果 patching——锚点页是否**因果上**决定纠错可预测性。

对每条长 trace:
1. 标 reflection 步,用全量 KV 探针在每个 reflection 位置 receiver 反查得该步的 R_t 锚点页;
2. 全量 prefill 得每层参考 KV(供 patch);
3. 跑四 mode 的 teacher-forced replay,测纠错 span 的 NLL:
   full / evict / patch_anchor(reflection 时把 R_t patch 回) / patch_random(patch 等量随机已丢页)。

判据:patch_anchor 的 ΔNLL(相对 full)显著 < evict,且 < patch_random
→ 正是这些锚点页因果上恢复了纠错预测(单点干预 + null 对照)。

用法:PYTHONPATH=. python scripts/step2_patch.py --model <权重> --max-traces 8
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import yaml

from rkv.attn_probe_rows import probe_rows, receiver_lookup_row
from rkv.online_features import OnlineSignature
from rkv.runner_evict import score_trace_patch
from rkv.utils.reflection import mark_reflection_steps

MODES = ["full", "evict", "patch_anchor", "patch_random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scorer", default="results/signature_scorer.json")
    ap.add_argument("--trace-dir", default="data/traces_long")
    ap.add_argument("--keep-frac", type=float, default=0.2)
    ap.add_argument("--protect-recent", type=int, default=24)
    ap.add_argument("--span-len", type=int, default=32)
    ap.add_argument("--max-traces", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=5000)
    ap.add_argument("--out", default="results/step2_patch.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"] = args.model
    cfg["attn_implementation"] = "eager"
    ps = cfg.get("page_size", 16)

    from rkv.runner import load_model
    model, tokenizer = load_model(cfg)
    device = next(model.parameters()).device
    files = sorted(glob.glob(f"{args.trace_dir}/*.pt"))[: args.max_traces]
    print(f"[patch] {len(files)} traces × {len(MODES)} modes, keep={args.keep_frac}")

    records = []
    for fi, f in enumerate(files):
        tr = torch.load(f, weights_only=False)
        P = tr["prompt_ids"].shape[0]
        gen = tr["gen_ids"][: args.max_len]
        refl = mark_reflection_steps(gen, tokenizer, step_entropy=tr.get("step_entropy"),
                                     entropy_quantile=cfg.get("entropy_quantile"))
        if not refl:
            print(f"  skip {Path(f).name}（无 reflection）"); continue
        full_ids = torch.cat([tr["prompt_ids"], gen]).unsqueeze(0).to(device)
        L = full_ids.shape[1]

        # 1) 每个 reflection 的 R_t 锚点页：全量 KV 探针 + receiver 反查
        refl_rows = [P + s for s in refl if P + s < L]
        attn = probe_rows(model, full_ids, refl_rows, page_size=ps,
                          layers=cfg.get("probe_layers"))
        refl_pages = {}
        for i, s in enumerate([s for s in refl if P + s < L]):
            pages = receiver_lookup_row(attn[i], P + s, page_size=ps,
                                        receiver_topk=cfg.get("receiver_topk", 8),
                                        exclude_recent_pages=cfg.get("exclude_recent_pages", 4))
            if pages:
                refl_pages[s] = pages
        if not refl_pages:
            print(f"  skip {Path(f).name}（无 receiver 页）"); continue

        rec = {"id": tr.get("id", Path(f).stem), "n_refl_used": len(refl_pages),
               "gen_len": int(gen.numel()), "modes": {}}
        # 2) full mode 先跑,逐 token 末态缓存 = 全序列参考 KV(无 OOM,顺手拿)
        full_res, ref_kv = score_trace_patch(
            model, tokenizer, full_ids, P, list(refl_pages), refl_pages, None, "full",
            span_len=args.span_len, page_size=ps, keep_frac=args.keep_frac,
            protect_recent=args.protect_recent, return_ref=True)
        rec["modes"]["full"] = full_res
        print(f"  [{fi+1}/{len(files)}] {rec['id']} {'full':13s} nll_corr={full_res['nll_corr']:.4f}")
        torch.cuda.empty_cache()
        for mode in ["evict", "patch_anchor", "patch_random"]:
            sig = OnlineSignature(args.scorer, ps, device)
            r = score_trace_patch(
                model, tokenizer, full_ids, P, list(refl_pages), refl_pages, sig, mode,
                ref_kv=ref_kv, span_len=args.span_len, page_size=ps,
                keep_frac=args.keep_frac, protect_recent=args.protect_recent)
            rec["modes"][mode] = r
            print(f"  [{fi+1}/{len(files)}] {rec['id']} {mode:13s} nll_corr={r['nll_corr']:.4f}")
        records.append(rec)
        del ref_kv; torch.cuda.empty_cache()
        _dump(records, args)
    _dump(records, args)
    _report(records)


def _summarize(records):
    summ = {}
    for mode in MODES:
        rows = [r["modes"][mode] for r in records if mode in r["modes"]]
        if not rows:
            continue
        nc = sum(x["nll_corr"] * x["n_corr"] for x in rows)
        ncn = sum(x["n_corr"] for x in rows)
        summ[mode] = {"nll_corr": nc / max(ncn, 1), "n_corr": ncn}
    if "full" in summ:
        base = summ["full"]["nll_corr"]
        for m in summ:
            summ[m]["dNLL_corr"] = summ[m]["nll_corr"] - base
    return summ


def _dump(records, args):
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "summary": _summarize(records), "records": records},
        indent=2, ensure_ascii=False))


def _report(records):
    summ = _summarize(records)
    print(f"\n=== T3 因果 patching（{len(records)} traces）===")
    print("| mode | NLL(纠错) | ΔNLL(纠错) |")
    print("|------|-----------|------------|")
    for m in MODES:
        if m not in summ:
            continue
        d = summ[m].get("dNLL_corr")
        print(f"| {m} | {summ[m]['nll_corr']:.4f} | "
              f"{d:+.4f} |" if d is not None else f"| {m} | {summ[m]['nll_corr']:.4f} | — |")
    if {"evict", "patch_anchor", "patch_random"} <= set(summ):
        ev, pa, pr = (summ[k]["dNLL_corr"] for k in ("evict", "patch_anchor", "patch_random"))
        rec = (ev - pa) / ev * 100 if ev else 0
        print(f"\npatch_anchor 相对 evict 恢复纠错 NLL 惩罚 {rec:.1f}%;"
              f" 对照 patch_random ΔNLL={pr:+.4f}（应明显大于 patch_anchor {pa:+.4f}）")
        print("判定:", "GO ✅ 锚点因果成立" if pa < pr and pa < ev else "存疑/NO-GO")


if __name__ == "__main__":
    main()
