"""Model and prompt helpers."""
from __future__ import annotations

import torch

_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_causal_lm(model_name: str, dtype: str = "bfloat16", device_map: str | dict = "cuda",
                   attn_implementation: str = "eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_DTYPE[dtype],
        device_map=device_map,
        attn_implementation=attn_implementation,
    ).eval()
    return model, tokenizer


def build_input_ids(tokenizer, question: str, device):
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        ids = tokenizer(question, return_tensors="pt").input_ids
    return ids.to(device)
