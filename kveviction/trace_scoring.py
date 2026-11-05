"""Teacher-forced NLL scoring under token-level KV eviction (TideProbe diagnostics).

``score_trace_token`` replays a fixed generation trace through the same decode
loop as ``generate_token_evict`` (``runner_token._decode`` in teacher-forcing
mode), so eviction schedule, observation windows and slot features match
generation exactly, and accumulates NLL on correction spans versus the rest
of the trace.
"""
from __future__ import annotations

import torch

from kveviction.backends import ExecutionBackend
from kveviction.cache_manager import CacheManager
from kveviction.selection import is_headwise_selection


def _position_count_pairs(positions: torch.Tensor) -> list[list[int]]:
    """Compress exact per-layer/head positions into ``[position, count]`` rows."""
    if positions.numel() == 0:
        return []
    unique, counts = torch.unique(positions.reshape(-1), sorted=True, return_counts=True)
    return [
        [int(position), int(count)]
        for position, count in zip(unique.detach().cpu(), counts.detach().cpu())
    ]


def _selection_position_debug(position_maps, n: int, idx) -> tuple[list, list, list]:
    """Return kept maps plus exact kept/evicted true-position multiplicities."""
    kept_maps = []
    kept_positions = []
    evicted_positions = []
    headwise = is_headwise_selection(idx)
    for layer_index, position_map in enumerate(position_maps):
        num_heads = position_map.shape[0]
        if headwise:
            layer_idx = idx[layer_index]
            if layer_idx.ndim == 3:
                layer_idx = layer_idx[0]
        else:
            layer_idx = idx.unsqueeze(0).expand(num_heads, -1)
        layer_idx = layer_idx.to(position_map.device)
        kept = position_map[:, :n].gather(1, layer_idx)
        keep_mask = torch.zeros((num_heads, n), dtype=torch.bool, device=position_map.device)
        keep_mask.scatter_(1, layer_idx, True)
        kept_maps.append(kept)
        kept_positions.append(kept.reshape(-1))
        evicted_positions.append(position_map[:, :n][~keep_mask])
    return (
        kept_maps,
        _position_count_pairs(torch.cat(kept_positions)),
        _position_count_pairs(torch.cat(evicted_positions)),
    )


class _PositionTracker:
    """Track the true position held by every (layer, head, slot) for debug events."""

    def __init__(self, total_len: int):
        self.total_len = total_len
        self.maps = None
        self.events = []

    def prefill(self, cache, prompt_len: int) -> None:
        self.maps = []
        for layer in cache.layers:
            device = layer.keys.device
            mapping = torch.empty((layer.keys.shape[1], self.total_len), dtype=torch.long, device=device)
            mapping[:, :prompt_len] = torch.arange(prompt_len, device=device)
            self.maps.append(mapping)

    def append(self, slot: int, position: int) -> None:
        for mapping in self.maps:
            mapping[:, slot] = position

    def compact(self, step: int, before_n: int, idx, after_n: int) -> None:
        kept_maps, kept_counts, evicted_counts = _selection_position_debug(self.maps, before_n, idx)
        for mapping, kept in zip(self.maps, kept_maps):
            mapping[:, :after_n] = kept
        if before_n > after_n:
            self.events.append({
                "eviction_step": step + 1,
                "cache_len_before": before_n,
                "cache_len_after": after_n,
                "evicted": before_n - after_n,
                "kept_positions": [position for position, _ in kept_counts],
                "evicted_positions": [position for position, _ in evicted_counts],
                "kept_position_counts": kept_counts,
                "evicted_position_counts": evicted_counts,
            })


@torch.no_grad()
def score_trace_token(
    model,
    tokenizer,
    full_ids,
    prompt_len,
    refl_steps,
    anchor_mode,
    budget=1024,
    recent=64,
    sink=8,
    backend="rkv",
    anchor_frac=0.05,
    evict_every=128,
    obs_window=16,
    obs_decay=0.9,
    span_len=32,
    sig=None,
    policy_params=None,
    debug=False,
    seed=0,
    execution_backend: ExecutionBackend | None = None,
    cache_manager: CacheManager | None = None,
):
    """Teacher-force ``full_ids[:, prompt_len:]`` and measure NLL under eviction.

    Tokens in ``[s, s + span_len)`` for every ``s`` in ``refl_steps`` count as
    correction spans (``nll_corr``); the rest as ``nll_noncorr``.  ``debug``
    adds per-token NLL and per-eviction kept/evicted true positions.
    """
    from kveviction.runner_token import _decode

    device = next(model.parameters()).device
    torch.manual_seed(int(seed))
    full_ids = full_ids.to(device)
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    P = int(prompt_len)
    gen_ids = full_ids[:, P:]
    Lgen = int(gen_ids.shape[1])
    tracker = _PositionTracker(P + Lgen) if debug else None
    run = _decode(
        model, tokenizer, full_ids[:, :P], batch_size=1, solo=True,
        budget=budget, recent=recent, sink=sink, backend=backend,
        anchor_mode=anchor_mode, anchor_frac=anchor_frac, evict_every=evict_every,
        obs_window=obs_window, obs_decay=obs_decay, max_new=Lgen,
        do_sample=False, temperature=1.0, top_p=1.0, seeds=[seed],
        sig=sig, policy_params=policy_params, input_attention_mask=None,
        debug_path=None, debug_topk=0,
        execution_backend=execution_backend, cache_manager=cache_manager,
        forced_ids=gen_ids, trace_hook=tracker,
    )
    nll = run["forced_nll"][:, 0].cpu() if Lgen else torch.zeros(0)
    corr = torch.zeros(Lgen, dtype=torch.bool)
    for s in refl_steps:
        corr[s: min(s + span_len, Lgen)] = True
    n_corr = int(corr.sum())
    n_noncorr = Lgen - n_corr
    result = {
        "nll_corr": sum(nll[corr].tolist()) / max(n_corr, 1),
        "n_corr": n_corr,
        "nll_noncorr": sum(nll[~corr].tolist()) / max(n_noncorr, 1),
        "n_noncorr": n_noncorr,
    }
    if debug:
        result.update({
            "per_token_nll": nll.float(),
            "eviction_events": tracker.events,
            "final_cache_len": run["n"],
            "seed": int(seed),
        })
    return result
