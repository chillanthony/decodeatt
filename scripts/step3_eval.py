"""Step 3:token 级后端 headroom 验证 + 锚点正交增益评测。

阶段 A(headroom):full vs rkv/snapkv 在多档 token 预算下的 MATH 精度,确认低预算≈满精度。
阶段 B(正交增益,--anchor):在选定后端/预算上比 none/anchor/random/lowent。

用法:PYTHONPATH=. python scripts/step3_eval.py --model <权重> --dataset math500 --n 24 \
        --budgets 1024,512,256 [--anchor --backend rkv --budget 512]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch

from rkv.runner_token import generate_token_evict
from scripts.gen_traces import load_dataset_problems
from scripts.step2_eval import extract_answer, is_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=4000)
    ap.add_argument("--budgets", default="1024,512,256")
    ap.add_argument("--recent", type=int, default=64)
    ap.add_argument("--backends", default="rkv,snapkv")
    ap.add_argument("--anchor", action="store_true", help="阶段B:比 none/anchor/random/lowent")
    ap.add_argument("--anchor-frac", type=float, default=0.05, help="anchor 额外保护预算占主 budget 的比例")
    ap.add_argument("--backend", default="rkv")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--no-full", action="store_true", help="跳过 full 臂(预算扫描复用已有 full)")
    ap.add_argument("--scorer", default="results/signature_scorer.json", help="完整签名权重(sig 臂用)")
    ap.add_argument("--out", default="results/step3_eval.json")
    ap.add_argument("--only-ids", default="", help="只跑指定题目 id,逗号分隔,如 aime-8,aime-1")
    ap.add_argument("--eval-arms", default="", help="只跑指定 arm,逗号分隔,如 none,sig,random")
    ap.add_argument("--debug-dir", default="", help="非空时为每题每臂保存 token 淘汰 debug JSON")
    ap.add_argument("--debug-topk", type=int, default=64, help="debug 中每次淘汰最多解码多少个 token 文本样例")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    probs = load_dataset_problems(args.dataset, args.n)
    if args.only_ids:
        keep_ids = set(args.only_ids.split(","))
        probs = [p for p in probs if p["id"] in keep_ids]

    def run(prob, arm_name, **kw):
        ids = tok.apply_chat_template([{"role": "user", "content": prob["question"]}],
                                      add_generation_prompt=True, return_tensors="pt").to("cuda")
        dbg = None
        if args.debug_dir:
            dbg = str(Path(args.debug_dir) / f"{prob['id']}_{arm_name}.json")
        r = generate_token_evict(model, tok, ids, max_new=args.max_new, recent=args.recent,
                                 anchor_frac=args.anchor_frac, do_sample=True, seed=0,
                                 debug_path=dbg, debug_topk=args.debug_topk, **kw)
        return is_correct(extract_answer(r["text"]), prob["answer"]), len(r["gen_ids"])

    if args.anchor:
        # 加载完整签名权重(原始空间线性,顺序=[entropy,cum_attn,position,concentration])
        sg = json.load(open(args.scorer))
        idx = [sg["feature_names"].index(f) for f in ["entropy", "cum_attn", "position", "concentration"]]
        sig = ([sg["w_raw"][i] for i in idx], sg["b_raw"])
        b = args.budget
        arms = [("none", dict(backend=args.backend, budget=b, anchor_mode="none")),
                ("sig", dict(backend=args.backend, budget=b, anchor_mode="sig", sig=sig)),
                ("anchor", dict(backend=args.backend, budget=b, anchor_mode="anchor")),
                ("random", dict(backend=args.backend, budget=b, anchor_mode="random")),
                ("lowent", dict(backend=args.backend, budget=b, anchor_mode="lowent"))]
        full_arm = not args.no_full
    else:
        arms = []
        for be in args.backends.split(","):
            for b in [int(x) for x in args.budgets.split(",")]:
                arms.append((f"{be}@{b}", dict(backend=be, budget=b, anchor_mode="none")))
        full_arm = not args.no_full
    if args.eval_arms:
        keep_arms = set(args.eval_arms.split(","))
        arms = [a for a in arms if a[0] in keep_arms]

    recs = []
    for i, prob in enumerate(probs):
        row = {"id": prob["id"], "gold": prob["answer"], "arms": {}}
        if full_arm:
            # full = 超大预算（不触发淘汰）
            ok, L = run(prob, "full", backend="snapkv", budget=10**9, anchor_mode="none")
            row["arms"]["full"] = {"ok": ok, "len": L}
        for name, kw in arms:
            ok, L = run(prob, name, **kw)
            row["arms"][name] = {"ok": ok, "len": L}
            print(f"[{i+1}/{len(probs)}] {prob['id']} {name:12s} ok={ok} len={L}", flush=True)
        recs.append(row)
        json.dump(recs, open(args.out, "w"), ensure_ascii=False, indent=1)

    print(f"\n=== Step3 {'正交增益' if args.anchor else 'headroom'} (MATH n={len(recs)}) ===")
    keys = ((["full"] if full_arm else []) + [a[0] for a in arms])
    for k in keys:
        if any(k not in r["arms"] for r in recs):
            continue
        acc = sum(r["arms"][k]["ok"] for r in recs) / len(recs)
        print(f"{k:14s} acc={acc:.3f} ({sum(r['arms'][k]['ok'] for r in recs)}/{len(recs)})")


if __name__ == "__main__":
    main()
