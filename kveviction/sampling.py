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


class RowUniforms:
    """Per-row uniform draws, taken from each row's own generator in blocks.

    Every row keeps an independent stream that depends only on its seed, so a
    request samples the same tokens alone or in any batch.  Drawing ``block``
    numbers per call replaces one tiny RNG launch per row per step.
    """

    def __init__(self, generators, block: int = 256):
        self.generators, self.block = generators, block
        self.buffer, self.pos = None, block

    def next(self) -> torch.Tensor:
        if self.pos == self.block:
            self.buffer = torch.stack([
                torch.rand(self.block, generator=g, device=g.device) for g in self.generators
            ], dim=1)                                        # [block, B]
            self.pos = 0
        self.pos += 1
        return self.buffer[self.pos - 1]


# Candidates kept before the nucleus cut; a row whose nucleus is wider falls back to a full sort.
_TOP_K = 256


def _nucleus(probs: torch.Tensor, top_p: float, k: int):
    """Sorted top-``k`` probabilities with the top-p mask (keep while cumsum <= top_p, always the first)."""
    values, indices = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    keep = values.cumsum(-1) <= top_p
    keep[:, 0] = True
    return values * keep, indices, keep


def sample_batch(logits, do_sample, temperature, top_p, uniforms: RowUniforms):
    """Sample one token per row by inverse CDF over the top-p nucleus.

    Same distribution as top-p + multinomial, but batched: one top-k instead of
    a full-vocabulary sort, and one uniform per row from ``uniforms``.  Rows
    whose nucleus does not fit in the top ``_TOP_K`` use a full sort.
    """
    if not do_sample:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / max(temperature, 1e-6), -1)
    u = uniforms.next().to(probs.device)
    if top_p >= 1.0:
        weights, indices = probs, None
    else:
        weights, indices, keep = _nucleus(probs, top_p, _TOP_K)
        if keep.shape[-1] < probs.shape[-1] and bool(keep[:, -1].any()):
            weights, indices, _ = _nucleus(probs, top_p, probs.shape[-1])
    cdf = weights.cumsum(-1)
    target = (u * cdf[:, -1]).unsqueeze(-1)
    choice = torch.searchsorted(cdf, target, right=True).clamp_(max=cdf.shape[-1] - 1)
    # Never land on a dropped (zero-weight) slot past the nucleus.
    choice = torch.minimum(choice, (weights > 0).sum(-1, keepdim=True) - 1)
    choice = choice.squeeze(-1)
    if indices is None:
        return choice
    return indices.gather(1, choice.unsqueeze(-1)).squeeze(-1)


def entropy_batch(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return -(torch.where(p > 0, p * logp, torch.zeros_like(p))).sum(-1)


def entropy(logits: torch.Tensor) -> float:
    return float(entropy_batch(logits).sum())
