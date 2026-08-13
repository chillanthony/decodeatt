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


def pad_input_ids(tokenizer, rows: list[torch.Tensor], device):
    """Left-pad already-tokenized prompts and return ids, mask, and lengths."""
    if not rows:
        raise ValueError("cannot pad an empty prompt batch")
    flat_rows = [row.reshape(-1).to(device) for row in rows]
    lengths = torch.tensor([row.numel() for row in flat_rows], dtype=torch.long, device=device)
    max_length = int(lengths.max())
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    input_ids = torch.full(
        (len(flat_rows), max_length), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, row in enumerate(flat_rows):
        length = int(row.numel())
        input_ids[row_idx, max_length - length:] = row
        attention_mask[row_idx, max_length - length:] = 1
    return input_ids, attention_mask, lengths
