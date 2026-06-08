"""模型加载与解码循环。

Step 0：黑盒 generate 即可（见 scripts/gen_traces.py）。
Step 1：在此实现自写解码循环 + KV 淘汰拦截 + 注意力探针。
"""
from __future__ import annotations

import torch

_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(cfg: dict):
    """按 config 加载 causal LM 与 tokenizer。返回 (model, tokenizer)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]
    dtype = _DTYPE[cfg.get("dtype", "bfloat16")]
    device = cfg.get("device", "cuda")

    kwargs = {}
    # 探针阶段需 eager 才能拿到 attentions；生成阶段可用 sdpa/flash 提速
    attn_impl = cfg.get("attn_implementation")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=device,
        **kwargs,
    )
    model.eval()
    return model, tokenizer
