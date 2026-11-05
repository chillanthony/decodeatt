"""Token sampling and logit entropy helpers for the decoding loops."""
from __future__ import annotations

import torch


def sample(logits, do_sample, temperature, top_p):
    """采样或贪心。logits: [1, vocab]，使用全局 RNG。"""
    if not do_sample:
        return int(logits.argmax(-1))
    lg = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(lg, -1)[0]
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cdf = sp.cumsum(0)
        keep = cdf <= top_p
        keep[0] = True
        probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


def sample_batch(logits, do_sample, temperature, top_p, generators):
    """Sample one token per row while keeping an independent RNG stream."""
    if not do_sample:
        return logits.argmax(-1)
    lg = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(lg, -1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        keep = sorted_probs.cumsum(-1) <= top_p
        keep[:, 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_indices, sorted_probs * keep)
        probs = filtered / filtered.sum(-1, keepdim=True)
    return torch.stack([
        torch.multinomial(probs[row], 1, generator=generators[row]).squeeze(0)
        for row in range(probs.shape[0])
    ])


def entropy_batch(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return -(torch.where(p > 0, p * logp, torch.zeros_like(p))).sum(-1)


def entropy(logits: torch.Tensor) -> float:
    return float(entropy_batch(logits).sum())
