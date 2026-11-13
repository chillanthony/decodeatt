"""Grouped decode attention and batched top-p sampling."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kveviction import gqa_attention
from kveviction.sampling import RowUniforms, sample_batch


def _qkv(batch=3, heads=8, kv_heads=2, length=11, dim=16, q_len=1):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(batch, heads, q_len, dim, generator=g)
    k = torch.randn(batch, kv_heads, length, dim, generator=g)
    v = torch.randn(batch, kv_heads, length, dim, generator=g)
    return q, k, v


def _padding_mask(batch, length, q_len=1):
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[0, :3] = False                                   # row 0 is left padded
    return valid[:, None, None, :].expand(batch, 1, q_len, length)


def test_grouped_eager_matches_repeat_kv():
    module = SimpleNamespace(training=False)
    q, k, v = _qkv()
    additive = torch.zeros(3, 1, 1, 11).masked_fill(~_padding_mask(3, 11), torch.finfo(torch.float32).min)
    out, weights = gqa_attention.gqa_eager_attention_forward(module, q, k, v, additive, scaling=0.25)
    ref_out, ref_weights = gqa_attention._hf_eager(module, q, k, v, additive, scaling=0.25)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(weights, ref_weights)
    assert weights[0, :, 0, :3].abs().max() == 0          # padding gets no attention


def test_grouped_sdpa_matches_repeat_kv():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    module = SimpleNamespace(training=False, is_causal=True, num_key_value_groups=4)
    q, k, v = _qkv()
    mask = _padding_mask(3, 11)
    out, _ = gqa_attention.make_gqa_sdpa(sdpa_attention_forward)(module, q, k, v, mask, scaling=0.25)
    ref, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=0.25)
    torch.testing.assert_close(out, ref)


def test_prefill_keeps_the_hf_path():
    module = SimpleNamespace(training=False)
    q, k, v = _qkv(q_len=11)
    out, weights = gqa_attention.gqa_eager_attention_forward(module, q, k, v, None, scaling=0.25)
    ref_out, ref_weights = gqa_attention._hf_eager(module, q, k, v, None, scaling=0.25)
    assert torch.equal(out, ref_out) and torch.equal(weights, ref_weights)


def _uniforms(seeds):
    return RowUniforms([torch.Generator().manual_seed(s) for s in seeds])


@pytest.mark.parametrize("vocab", [40, 2000])          # 2000 > top-k: wide nuclei use the full sort
def test_sampling_follows_the_top_p_distribution(vocab):
    logits = torch.randn(1, vocab, generator=torch.Generator().manual_seed(1)) * (3.0 if vocab == 40 else 0.05)
    temperature, top_p, draws = 0.7, 0.9, 20000
    probs = torch.softmax(logits / temperature, -1)[0]
    sorted_probs, order = probs.sort(descending=True)
    keep = sorted_probs.cumsum(0) <= top_p
    keep[0] = True
    expected = torch.zeros_like(probs).scatter_(0, order[keep], sorted_probs[keep])
    expected /= expected.sum()

    uniforms = _uniforms([3])
    counts = torch.zeros(vocab)
    for _ in range(draws):
        counts[sample_batch(logits, True, temperature, top_p, uniforms)] += 1
    assert counts[expected == 0].sum() == 0                # never outside the nucleus
    top = expected.argsort(descending=True)[:5]
    torch.testing.assert_close(counts[top] / draws, expected[top], atol=0.015, rtol=0)


def test_each_row_samples_the_same_alone_or_in_a_batch():
    logits = torch.randn(3, 50, generator=torch.Generator().manual_seed(2)) * 2
    batch = _uniforms([10, 11, 12])
    alone = [_uniforms([s]) for s in (10, 11, 12)]
    for _ in range(300):                                   # crosses the 256-draw block boundary
        together = sample_batch(logits, True, 0.6, 0.95, batch)
        single = [sample_batch(logits[i:i + 1], True, 0.6, 0.95, alone[i]).item() for i in range(3)]
        assert together.tolist() == single


def _redundancy_reference(keys, threshold, retain_ratio, retain_direction, eps, chunk_size, valid_mask=None):
    """The R-KV redundancy as written before 2026-11-13 (one pass, int64 index matrix)."""
    prefix, n_cand = keys.shape[:-2], keys.shape[-2]
    keys = keys.reshape(-1, n_cand, keys.shape[-1])
    flat_heads = keys.shape[0]
    keys = keys.float() / (keys.float().norm(dim=-1, keepdim=True) + eps)
    flat_valid = None
    if valid_mask is not None:
        flat_valid = valid_mask.bool().unsqueeze(1).expand(prefix[0], prefix[1], n_cand).reshape(flat_heads, n_cand)
    col_ids = torch.arange(n_cand).view(1, 1, n_cand)
    col_sum = torch.zeros(flat_heads, n_cand)
    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        sim = keys[:, start:end] @ keys.transpose(1, 2)
        sim[:, torch.arange(end - start), torch.arange(start, end)] = 0.0
        if flat_valid is not None:
            sim = sim * flat_valid[:, start:end].unsqueeze(-1)
            sim = sim * flat_valid.unsqueeze(1)
        indices = torch.where(sim > threshold, col_ids, torch.zeros_like(sim, dtype=torch.long))
        k = max(1, int(n_cand * retain_ratio))
        retain = {
            "last": lambda: indices.max(-1).values,
            "first": lambda: indices.min(-1).values,
            "last_percent": lambda: indices.topk(k, dim=-1).values[:, :, 0],
            "first_percent": lambda: indices.topk(k, dim=-1, largest=False).values[:, :, -1],
        }[retain_direction]()
        sim.scatter_(-1, retain.unsqueeze(-1), 0)
        col_sum += sim.sum(dim=1)
    denominator = flat_valid.sum(-1, keepdim=True).clamp_min(1) if flat_valid is not None else n_cand
    redundancy = col_sum / denominator
    if flat_valid is not None:
        redundancy = redundancy.masked_fill(~flat_valid, torch.finfo(redundancy.dtype).min)
    return torch.softmax(redundancy, dim=-1).reshape(*prefix, n_cand)


@pytest.mark.parametrize("direction", ["last", "first", "last_percent", "first_percent"])
@pytest.mark.parametrize("padded", [False, True])
def test_grouped_redundancy_matches_reference(monkeypatch, direction, padded):
    from kveviction.strategies import rkv_official

    g = torch.Generator().manual_seed(4)
    base = torch.randn(3, 2, 6, 8, generator=g)
    keys = torch.cat([base, base + 0.05 * torch.randn(3, 2, 6, 8, generator=g)], dim=2)   # near-duplicates
    valid = torch.ones(3, 12, dtype=torch.bool)
    if padded:
        valid[0, :4] = False
    mask = valid if padded else None
    want = _redundancy_reference(keys, 0.5, 0.1, direction, 1e-8, 5, mask)
    monkeypatch.setattr(rkv_official, "_GROUP_BYTES", 1)            # one head per group
    got = rkv_official.rkv_redundancy(keys, 0.5, 0.1, direction, 1e-8, 5, valid_mask=mask)
    assert torch.equal(got, want)


def test_folded_observation_scores_match_broadcast():
    from kveviction.strategies.attention_utils import compute_attention_scores

    q, k, _ = _qkv(q_len=3)
    grouped = q.view(3, 2, 4, 3, 16)
    want = torch.matmul(grouped, k.unsqueeze(2).transpose(3, 4)) / 4.0
    torch.testing.assert_close(compute_attention_scores(q, k), want.max(dim=2).values)
    torch.testing.assert_close(compute_attention_scores(q, k, pooling="mean"), want.mean(dim=2))
