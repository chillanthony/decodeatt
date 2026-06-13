"""Phase 2：五臂端到端评测——签名豁免能否恢复纠错子集精度。

每题跑 full / evict / rescue / random / gsig 五臂（同预算），抽 \\boxed{} 答案、
判对错；分别汇报整体精度、**含 reflection 子集**精度、以及各臂的早期注意力质量
（receiver 可达性代理）。

用法：PYTHONPATH=. python scripts/step2_eval.py --model <权重> \
        --dataset mix --n 30 --max-new 8192 --keep-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml

from rkv.online_features import OnlineSignature
from rkv.runner_evict import ARMS, generate_with_evict
from scripts.gen_traces import load_dataset_problems

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_answer(text: str) -> str | None:
    m = list(_BOXED.finditer(text))
    if m:
        return _norm(m[-1].group(1))
    # 兜底：末尾的 "answer is X" 数字
    m2 = re.findall(r"(?:answer|=)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text[-200:], re.I)
    return _norm(m2[-1]) if m2 else None


def _norm(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.rstrip(".")
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[6:-1]
    return s


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    g = _norm(gold)
    if pred == g:
        return True
    try:
        return abs(float(pred) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


def has_reflection(text: str) -> bool:
    from rkv.utils.reflection import _REFLECTION_RE
    return bool(_REFLECTION_RE.search(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scorer", default="results/signature_scorer.json")
    ap.add_argument("--dataset", default="mix", choices=["aime", "math500", "mix"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--keep-frac", type=float, default=0.2)
    ap.add_argument("--exempt-frac", type=float, default=0.05)
    ap.add_argument("--backend", default="window", choices=["window", "h2o"])
    ap.add_argument("--evict-every", type=int, default=32)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="results/step2_eval.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"] = args.model
    cfg["attn_implementation"] = "eager"          # 自写循环需 output_attentions
    page_size = cfg.get("page_size", 16)
    arms = args.arms.split(",")

    from rkv.runner import load_model
    model, tokenizer = load_model(cfg)
    problems = load_dataset_problems(args.dataset, args.n)
    print(f"[step2] {len(problems)} problems × {len(arms)} arms, "
          f"keep={args.keep_frac} exempt={args.exempt_frac} backend={args.backend}")

    from rkv.utils.reflection import _REFLECTION_RE
    records = []
    for i, prob in enumerate(problems):
        ids = _build_ids(tokenizer, prob["question"], model)
        row = {"id": prob["id"], "gold": prob["answer"], "arms": {}}
        for arm in arms:
            sig = OnlineSignature(args.scorer, page_size=page_size,
                                  device=next(model.parameters()).device) \
                if arm != "full" else None
            r = generate_with_evict(
                model, tokenizer, ids, sig, arm, max_new=args.max_new,
                page_size=page_size, keep_frac=args.keep_frac,
                exempt_frac=args.exempt_frac, backend=args.backend,
                evict_every=args.evict_every,
                receiver_topk=cfg.get("receiver_topk", 8),
                exclude_recent_pages=cfg.get("exclude_recent_pages", 4))
            pred = extract_answer(r["text"])
            ok = is_correct(pred, prob["answer"])
            row["arms"][arm] = {
                "correct": ok, "pred": pred, "gen_len": len(r["gen_ids"]),
                "reflection": bool(_REFLECTION_RE.search(r["text"])),
                "n_evict": r["n_evict_rounds"], "n_refl": r["n_reflection"],
                "early_mass": (r["receiver_hits"] / r["receiver_total"]
                               if r["receiver_total"] else None),
            }
            print(f"  [{i+1}/{len(problems)}] {prob['id']} {arm:7s} "
                  f"{'✓' if ok else '✗'} pred={pred} len={len(r['gen_ids'])} "
                  f"evict={r['n_evict_rounds']}")
        records.append(row)
        _dump(records, arms, args)        # 边跑边存（防中断丢结果）

    _dump(records, arms, args)
    _report(records, arms)


def _build_ids(tokenizer, question, model):
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True, return_tensors="pt")
    else:
        ids = tokenizer(question, return_tensors="pt").input_ids
    return ids.to(next(model.parameters()).device)


def _summarize(records, arms):
    summ = {}
    # 含 reflection 子集：以 full 臂判断该题是否需要纠错（最公允）
    refl_ids = {r["id"] for r in records
                if r["arms"].get("full", {}).get("reflection")}
    for arm in arms:
        rows = [r["arms"][arm] for r in records if arm in r["arms"]]
        acc = sum(x["correct"] for x in rows) / max(len(rows), 1)
        refl_rows = [r["arms"][arm] for r in records
                     if r["id"] in refl_ids and arm in r["arms"]]
        acc_refl = (sum(x["correct"] for x in refl_rows) / len(refl_rows)
                    if refl_rows else None)
        masses = [x["early_mass"] for x in rows if x["early_mass"] is not None]
        lens = [x["gen_len"] for x in rows]
        summ[arm] = {
            "acc": acc, "acc_reflection": acc_refl,
            "n": len(rows), "n_reflection": len(refl_rows),
            "mean_early_mass": sum(masses) / len(masses) if masses else None,
            "mean_gen_len": sum(lens) / len(lens) if lens else None,
        }
    return summ, len(refl_ids)


def _dump(records, arms, args):
    summ, n_refl = _summarize(records, arms)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "n_reflection_subset": n_refl,
         "summary": summ, "records": records}, indent=2, ensure_ascii=False))


def _report(records, arms):
    summ, n_refl = _summarize(records, arms)
    print(f"\n=== Phase 2 五臂评测（{len(records)} 题，含 reflection 子集 {n_refl} 题）===")
    print("| arm | 整体精度 | 纠错子集精度 | 早期注意力质量 | 平均生成长度 |")
    print("|-----|----------|--------------|----------------|--------------|")
    for arm in arms:
        s = summ[arm]
        ar = f"{s['acc_reflection']:.3f}" if s["acc_reflection"] is not None else "—"
        em = f"{s['mean_early_mass']:.3f}" if s["mean_early_mass"] is not None else "—"
        ml = f"{s['mean_gen_len']:.0f}" if s["mean_gen_len"] else "—"
        print(f"| {arm} | {s['acc']:.3f} | {ar} | {em} | {ml} |")


if __name__ == "__main__":
    main()
