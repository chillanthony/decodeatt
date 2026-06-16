"""Phase 3（稳健版）：Teacher-forced 因果探针。

在已生成好的长 trace 上,喂真 token 走五臂淘汰循环,测 reflection 后"纠错 span"token
的 NLL。无自由生成 -> 无连贯性崩溃混淆;臂间唯一差异是缓存内容。

核心假设(SCRL):若纠错锚点真被签名保住,则 rescue 臂在纠错 token 上的 NLL 涨幅
(相对 full)应**小于** evict / random / gsig。

用法:PYTHONPATH=. python scripts/step2_teacherforce.py --model <权重> \
        --trace-dir data/traces_long --keep-frac 0.2 --max-traces 12
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import yaml

from rkv.online_features import OnlineSignature
from rkv.runner_evict import ARMS, score_trace_teacherforced
from rkv.utils.reflection import mark_reflection_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scorer", default="results/signature_scorer.json")
    ap.add_argument("--trace-dir", default="data/traces_long")
    ap.add_argument("--keep-frac", type=float, default=0.2)
    ap.add_argument("--exempt-frac", type=float, default=0.05)
    ap.add_argument("--protect-recent", type=int, default=24)
    ap.add_argument("--span-len", type=int, default=32)
    ap.add_argument("--backend", default="window")
    ap.add_argument("--max-traces", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=8000, help="过长 trace 截断省时")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="results/step2_teacherforce.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"] = args.model
    cfg["attn_implementation"] = "eager"
    page_size = cfg.get("page_size", 16)
    arms = args.arms.split(",")

    from rkv.runner import load_model
    model, tokenizer = load_model(cfg)
    device = next(model.parameters()).device
    files = sorted(glob.glob(f"{args.trace_dir}/*.pt"))[: args.max_traces]
    print(f"[teacherforce] {len(files)} traces × {len(arms)} arms, "
          f"keep={args.keep_frac} pr={args.protect_recent}")

    records = []
    for fi, f in enumerate(files):
        tr = torch.load(f, weights_only=False)
        P = tr["prompt_ids"].shape[0]
        gen = tr["gen_ids"][: args.max_len]
        refl = mark_reflection_steps(gen, tokenizer,
                                     step_entropy=tr.get("step_entropy"),
                                     entropy_quantile=cfg.get("entropy_quantile"))
        if not refl:
            print(f"  skip {Path(f).name}（无 reflection）")
            continue
        full_ids = torch.cat([tr["prompt_ids"], gen]).unsqueeze(0)
        rec = {"id": tr.get("id", Path(f).stem), "n_refl": len(refl),
               "gen_len": int(gen.numel()), "arms": {}}
        for arm in arms:
            sig = OnlineSignature(args.scorer, page_size, device) if arm != "full" else None
            r = score_trace_teacherforced(
                model, tokenizer, full_ids, P, refl, sig, arm,
                span_len=args.span_len, page_size=page_size,
                keep_frac=args.keep_frac, protect_recent=args.protect_recent,
                exempt_frac=args.exempt_frac, backend=args.backend)
            rec["arms"][arm] = r
            print(f"  [{fi+1}/{len(files)}] {rec['id']} {arm:7s} "
                  f"nll_corr={r['nll_corr']:.4f} nll_non={r['nll_noncorr']:.4f}")
        records.append(rec)
        _dump(records, arms, args)
    _dump(records, arms, args)
    _report(records, arms)


def _summarize(records, arms):
    summ = {}
    base = {a: None for a in arms}
    # 每臂在纠错 / 非纠错 token 上的平均 NLL,以及相对 full 的涨幅
    for arm in arms:
        rows = [r["arms"][arm] for r in records if arm in r["arms"]]
        if not rows:
            continue
        nc = sum(x["nll_corr"] * x["n_corr"] for x in rows)
        ncn = sum(x["n_corr"] for x in rows)
        nn = sum(x["nll_noncorr"] * x["n_noncorr"] for x in rows)
        nnn = sum(x["n_noncorr"] for x in rows)
        summ[arm] = {"nll_corr": nc / max(ncn, 1), "nll_noncorr": nn / max(nnn, 1),
                     "n_corr": ncn, "n_noncorr": nnn}
    full = summ.get("full")
    if full:
        for arm in summ:
            summ[arm]["dNLL_corr"] = summ[arm]["nll_corr"] - full["nll_corr"]
            summ[arm]["dNLL_noncorr"] = summ[arm]["nll_noncorr"] - full["nll_noncorr"]
    return summ


def _dump(records, arms, args):
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "summary": _summarize(records, arms),
         "records": records}, indent=2, ensure_ascii=False))


def _report(records, arms):
    summ = _summarize(records, arms)
    print(f"\n=== Teacher-forced 因果探针（{len(records)} traces）===")
    print("| arm | NLL(纠错) | ΔNLL(纠错) | NLL(非纠错) | ΔNLL(非纠错) |")
    print("|-----|-----------|------------|-------------|--------------|")
    for arm in arms:
        if arm not in summ:
            continue
        s = summ[arm]
        dc = s.get("dNLL_corr"); dn = s.get("dNLL_noncorr")
        print(f"| {arm} | {s['nll_corr']:.4f} | {dc:+.4f} | "
              f"{s['nll_noncorr']:.4f} | {dn:+.4f} |"
              if dc is not None else
              f"| {arm} | {s['nll_corr']:.4f} | — | {s['nll_noncorr']:.4f} | — |")


if __name__ == "__main__":
    main()
