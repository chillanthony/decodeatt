"""Step3 NLL 兜底:token 级强后端(R-KV)上,完整签名 vs 熵 vs null 的纠错 NLL 正交增益。

在已有长 trace 上 teacher-forced,五臂 none/sig/anchor/random/lowent,测纠错 span NLL。
比 accuracy 敏感、便宜。用法:
  PYTHONPATH=. python scripts/step3_nll.py --model <权重> --budget 1024 --max-traces 12
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import torch, yaml
from rkv.runner_token import score_trace_token
from rkv.utils.reflection import mark_reflection_steps

ARMS = ["none", "sig", "anchor", "random", "lowent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scorer", default="results/signature_scorer.json")
    ap.add_argument("--trace-dir", default="data/traces_long")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--anchor-frac", type=float, default=0.05, help="anchor 额外保护预算占主 budget 的比例")
    ap.add_argument("--max-traces", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=16000)
    ap.add_argument("--out", default="results/step3_nll.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.model: cfg["model"] = args.model
    sg = json.load(open(args.scorer))
    idx = [sg["feature_names"].index(f) for f in ["entropy", "cum_attn", "position", "concentration"]]
    sig = ([sg["w_raw"][i] for i in idx], sg["b_raw"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    files = sorted(glob.glob(f"{args.trace_dir}/*.pt"))[: args.max_traces]
    print(f"[step3-nll] {len(files)} traces × {len(ARMS)} arms, budget={args.budget}")

    recs = []
    for fi, f in enumerate(files):
        tr = torch.load(f, weights_only=False)
        P = tr["prompt_ids"].shape[0]
        gen = tr["gen_ids"][: args.max_len]
        refl = mark_reflection_steps(gen, tok, step_entropy=tr.get("step_entropy"),
                                     entropy_quantile=cfg.get("entropy_quantile"))
        if not refl:
            print(f"  skip {Path(f).name}（无 reflection）"); continue
        full_ids = torch.cat([tr["prompt_ids"], gen]).unsqueeze(0)
        rec = {"id": tr.get("id", Path(f).stem), "arms": {}}
        for arm in ARMS:
            r = score_trace_token(model, tok, full_ids, P, refl, arm, budget=args.budget,
                                  anchor_frac=args.anchor_frac, sig=sig if arm == "sig" else None)
            rec["arms"][arm] = r
            print(f"  [{fi+1}/{len(files)}] {rec['id']} {arm:7s} nll_corr={r['nll_corr']:.4f}", flush=True)
        recs.append(rec)
        json.dump(recs, open(args.out, "w"), ensure_ascii=False, indent=1)

    print(f"\n=== Step3 NLL 兜底（{len(recs)} traces, budget={args.budget}）===")
    print("| arm | NLL(纠错) | NLL(非纠错) |")
    for arm in ARMS:
        rows = [r["arms"][arm] for r in recs if arm in r["arms"]]
        nc = sum(x["nll_corr"] * x["n_corr"] for x in rows) / max(sum(x["n_corr"] for x in rows), 1)
        nn = sum(x["nll_noncorr"] * x["n_noncorr"] for x in rows) / max(sum(x["n_noncorr"] for x in rows), 1)
        print(f"| {arm} | {nc:.4f} | {nn:.4f} |")


if __name__ == "__main__":
    main()
