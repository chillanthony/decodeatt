"""Step ①：用 R1-Distill-Qwen-7B 生成 CoT trace 并落盘。

每条 trace 落盘一个 .pt：prompt_ids / gen_ids / 每步熵 / 题目 / 标准答案 / 元信息。

用法：
  uv run python scripts/gen_traces.py --config configs/default.yaml --n 1
  # 烟雾测试（小模型、短生成）：
  uv run python scripts/gen_traces.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new 128 --n 1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from rkv.runner import load_model

# 内置示例题（仅作 --dataset sample 的兜底；正式实验用真 AIME/MATH）
SAMPLE_PROBLEMS = [
    {
        "id": "sample-aime-1",
        "question": (
            "Find the number of ordered pairs of positive integers (a, b) such that "
            "a + b = 1000 and neither a nor b has a zero digit."
        ),
        "answer": "738",
    },
    {
        "id": "sample-math-1",
        "question": "What is the remainder when 7^100 is divided by 13?",
        "answer": "9",
    },
]

# 数据集源（经 hf-mirror）：HF 上稳定可用的 AIME24 / MATH500
_DATASETS = {
    "aime": ("Maxwell-Jia/AIME_2024", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}
_Q_FIELDS = ["problem", "Problem", "question", "Question"]
_A_FIELDS = ["answer", "Answer", "solution", "Solution"]


def _pick(row: dict, fields: list[str]):
    for f in fields:
        if f in row and row[f] is not None:
            return row[f]
    raise KeyError(f"未找到字段 {fields}，实际列：{list(row.keys())}")


def load_dataset_problems(name: str, n: int) -> list[dict]:
    """从 HF（hf-mirror）加载真 AIME/MATH 题目；name=mix 则两者各半。"""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    if name == "mix":
        half = (n + 1) // 2
        return (load_dataset_problems("aime", half)
                + load_dataset_problems("math500", n - half))

    repo, split = _DATASETS[name]
    ds = load_dataset(repo, split=split)
    out = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        out.append({
            "id": f"{name}-{i}",
            "question": str(_pick(row, _Q_FIELDS)),
            "answer": str(_pick(row, _A_FIELDS)),
        })
    return out


def load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def build_input_ids(tokenizer, question: str, device):
    """优先用 chat template；没有则退化为纯文本。"""
    if tokenizer.chat_template:
        msgs = [{"role": "user", "content": question}]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        ids = tokenizer(question, return_tensors="pt").input_ids
    return ids.to(device)


@torch.no_grad()
def step_entropy(logits_steps) -> torch.Tensor:
    """从每步**原始** logits 算预测分布的熵（nats）。

    logits_steps: tuple[Tensor[batch, vocab]]（用 generate(output_logits=True)，
    是未经 temperature/top_p 截断的原始 logits）。返回 Tensor[num_steps]（batch=1）。
    """
    ent = []
    for logits in logits_steps:
        logp = torch.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        # 防护：截断/数值导致的 0*-inf -> 用 0 替代该项
        term = torch.where(p > 0, p * logp, torch.zeros_like(p))
        ent.append(-term.sum(-1).squeeze(0))
    return torch.stack(ent) if ent else torch.empty(0)


@torch.no_grad()
def generate_trace(model, tokenizer, problem: dict, cfg: dict, max_new: int):
    device = next(model.parameters()).device
    input_ids = build_input_ids(tokenizer, problem["question"], device)
    attention_mask = torch.ones_like(input_ids)

    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        do_sample=cfg.get("temperature", 0.0) > 0,
        temperature=cfg.get("temperature", 0.6),
        top_p=cfg.get("top_p", 0.95),
        return_dict_in_generate=True,
        output_logits=True,   # 原始 logits（未截断），用于算真实预测熵
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    prompt_len = input_ids.shape[1]
    gen_ids = out.sequences[0, prompt_len:].cpu()
    ent = step_entropy(out.logits).cpu()
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    return {
        "id": problem["id"],
        "question": problem["question"],
        "answer": problem["answer"],
        "model": cfg.get("model", "unknown"),
        "prompt_ids": input_ids[0].cpu(),
        "gen_ids": gen_ids,
        "step_entropy": ent,         # [num_steps]
        "gen_text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="覆盖 config 的模型（烟雾测试用小模型）")
    ap.add_argument("--max-new", type=int, default=None, help="覆盖 max_new_tokens")
    ap.add_argument("--n", type=int, default=1, help="生成几条 trace")
    ap.add_argument("--dataset", default="sample",
                    choices=["sample", "aime", "math500", "mix"],
                    help="题目来源：sample=内置兜底；aime/math500/mix=真数据集")
    ap.add_argument("--trace-dir", default=None, help="覆盖 config 的 trace 输出目录")
    ap.add_argument("--attn", default=None,
                    help="覆盖 attn_implementation（纯生成用 sdpa 提速；探针才需要 eager）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.attn:
        cfg["attn_implementation"] = args.attn
    max_new = args.max_new or cfg.get("max_new_tokens", 8192)

    trace_dir = Path(args.trace_dir or cfg.get("trace_dir", "data/traces"))
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen_traces] model={cfg['model']} max_new={max_new} n={args.n}")
    model, tokenizer = load_model(cfg)

    if args.dataset == "sample":
        problems = (SAMPLE_PROBLEMS * ((args.n // len(SAMPLE_PROBLEMS)) + 1))[: args.n]
    else:
        problems = load_dataset_problems(args.dataset, args.n)
    print(f"[gen_traces] dataset={args.dataset} -> {len(problems)} problems")
    for i, prob in enumerate(problems):
        trace = generate_trace(model, tokenizer, prob, cfg, max_new)
        out_path = trace_dir / f"{prob['id']}-{i}.pt"
        torch.save(trace, out_path)
        n_steps = trace["step_entropy"].numel()
        print(
            f"[{i+1}/{args.n}] {prob['id']}: gen={trace['gen_ids'].numel()} tok, "
            f"entropy_steps={n_steps}, mean_ent={trace['step_entropy'].mean():.3f} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
