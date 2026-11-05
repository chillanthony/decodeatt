"""Shared fixtures: a tiny randomly initialized Llama (GQA: 4 query / 2 KV heads, CPU)."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_llama():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg).eval()
    prompt = torch.randint(1, 64, (1, 20), generator=torch.Generator().manual_seed(1))
    return model, prompt
