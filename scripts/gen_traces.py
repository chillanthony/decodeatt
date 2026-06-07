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

# 内置示例题（无需下载数据集即可跑通；正式实验从 datasets 加载 AIME/MATH500）
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    max_new = args.max_new or cfg.get("max_new_tokens", 8192)

    trace_dir = Path(cfg.get("trace_dir", "data/traces"))
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen_traces] model={cfg['model']} max_new={max_new} n={args.n}")
    model, tokenizer = load_model(cfg)

    problems = (SAMPLE_PROBLEMS * ((args.n // len(SAMPLE_PROBLEMS)) + 1))[: args.n]
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
