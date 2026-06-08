"""Step ⑤：对一批 trace 出 GO/NO-GO 诊断。

串起 ①gen_traces 产物 + ②reflection + ③attn_probe + ④recurrence + scoring：
  对每条 trace：标 reflection 步 -> 探针 -> 反查 R_t -> 算 MRI；
  汇总 R_t（纠错回指 page）vs G（其余被访问 page）的 MRI 分布 + 通用淘汰器对 R_t 的错杀率。

GO：R_t 的 MRI 显著高于 G（KS 显著、中位数更大）且错杀率明显（>20-30%）。

用法：uv run python scripts/step0_diagnose.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import yaml

from rkv.runner import load_model
from rkv.attn_probe import probe_sequence, receiver_lookup
from rkv.rescue.recurrence import topk_pages_per_step, compute_mri
from rkv.utils.reflection import mark_reflection_steps
from rkv.utils import scoring


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def diagnose_trace(trace, tokenizer, model, cfg):
    """单条 trace -> (r_mri, g_mri, kill_rate)。无 reflection 步则返回 None。"""
    page_size = cfg.get("page_size", 16)
    prompt_ids = trace["prompt_ids"]
    gen_ids = trace["gen_ids"]
    prompt_len = prompt_ids.shape[0]
    full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)

    refl_gen = mark_reflection_steps(gen_ids, tokenizer)
    if not refl_gen:
        return None

    page_attn = probe_sequence(
        model, full, page_size=page_size,
        layers=cfg.get("probe_layers"), max_len=cfg.get("probe_max_len"),
    )
    L = page_attn.shape[0]

    topk = topk_pages_per_step(page_attn, cfg.get("mri_topk", 32))
    mri = compute_mri(topk)

    # R_t：所有 reflection 步反查到的早期 page（绝对位置 = prompt_len + gen_step）
    r_pages: set[int] = set()
    for t in refl_gen:
        pos = prompt_len + t
        if pos >= L:                      # 超出探针截断长度则跳过
            continue
        r_pages |= receiver_lookup(
            page_attn, pos, page_size=page_size,
            receiver_topk=cfg.get("receiver_topk", 8),
            exclude_recent_pages=cfg.get("exclude_recent_pages", 4),
        )
    if not r_pages:
        return None

    accessed = set(mri.keys())
    g_pages = accessed - r_pages

    r_mri = [mri.get(p, 0) for p in r_pages]
    g_mri = [mri[p] for p in g_pages]

    # 错杀率：通用淘汰器（按累计注意力保留 top keep_frac）丢掉的 R_t 比例
    evicted = scoring.evicted_by_budget(page_attn.sum(0), cfg.get("keep_frac", 0.2))
    kr = scoring.kill_rate(r_pages, evicted)

    return r_mri, g_mri, kr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="覆盖 config 的模型路径（用本地权重）")
    ap.add_argument("--max-traces", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = cfg.get("trace_dir", "data/traces")
    files = sorted(glob.glob(f"{trace_dir}/*.pt"))
    if args.max_traces:
        files = files[: args.max_traces]
    print(f"[step0] {len(files)} traces from {trace_dir}")

    model, tokenizer = load_model(cfg)

    all_r, all_g, kills = [], [], []
    used = 0
    for f in files:
        trace = torch.load(f, weights_only=False)
        res = diagnose_trace(trace, tokenizer, model, cfg)
        if res is None:
            print(f"  skip {Path(f).name} (无 reflection 步 / R_t 为空)")
            continue
        r_mri, g_mri, kr = res
        all_r += r_mri
        all_g += g_mri
        kills.append(kr)
        used += 1
        print(f"  {Path(f).name}: |R_t|={len(r_mri)} |G|={len(g_mri)} kill={kr:.2f}")

    if used == 0:
        print("无可用 trace（都没有 reflection 步）。换更长/更难的题重生成 trace。")
        return

    test = scoring.mri_distribution_test(all_r, all_g)
    mean_kill = sum(kills) / len(kills)
    scoring.plot_mri_cdf(all_r, all_g, str(results_dir / "mri_cdf.png"))

    summary = {**test, "mean_kill_rate": mean_kill, "n_traces": used}
    (results_dir / "step0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Step 0 诊断结果 ===")
    print(f"R_t MRI 中位数: {test['median_r']}  vs  G MRI 中位数: {test['median_g']}")
    print(f"KS={test['ks']}  p={test['p']}  (H1: R_t 的 MRI 更大)")
    print(f"平均错杀率: {mean_kill:.3f}")

    go = (
        test["p"] is not None and test["p"] < 0.05
        and test["median_r"] > test["median_g"]
        and mean_kill > 0.20
    )
    print(f"\n判定: {'GO ✅ (规律成立，继续 Step 1)' if go else 'NO-GO ❌ (前提不足，需重定位)'}")
    print(f"图/表已存到 {results_dir}/")


if __name__ == "__main__":
    main()
