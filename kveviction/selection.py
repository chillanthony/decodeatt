"""Eviction scheduling and slot selection shared by the decoding loops."""
from __future__ import annotations

import torch

from kveviction.cache_manager import _is_headwise as is_headwise_selection
from kveviction.cache_manager import _representative_indices as representative_indices
from kveviction.cache_manager import _representative_indices_batch as representative_indices_batch
from kveviction.controller import EvictionController

__all__ = [
    "is_headwise_selection", "representative_indices", "representative_indices_batch",
    "selection_length", "is_cache_aware", "eviction_due",
    "should_collect_observation", "select_tokens", "select_tokens_batch",
]


def selection_length(idx) -> int:
    if is_headwise_selection(idx):
        return int(idx[0].shape[-1])
    return int(idx.numel())


def is_cache_aware(policy) -> bool:
    return policy.cache_aware


def _over_trigger(policy, cache_len: int, budget: int, anchor_k: int) -> bool:
    """Cache-aware policies compact at ``budget``; others leave room for anchors."""
    if is_cache_aware(policy):
        return cache_len >= budget
    return cache_len > budget + anchor_k


def eviction_due(policy, *, track: bool, step: int, evict_every: int,
                 cache_len: int, budget: int, anchor_k: int) -> bool:
    """Whether to compact after decode ``step`` with ``cache_len`` live slots."""
    return (
        track
        and (step + 1) % evict_every == 0
        and _over_trigger(policy, cache_len, budget, anchor_k)
    )


def should_collect_observation(
    policy,
    *,
    track: bool,
    policy_params: dict,
    obs_window: int,
    evict_every: int,
    step: int,
    cache_len: int,
    budget: int,
    anchor_k: int,
) -> tuple[bool, int]:
    """Open attention observation only inside the window before a real compaction."""
    active_obs_window = policy.observation_window(policy_params, obs_window)
    if not track or active_obs_window <= 0:
        return False, active_obs_window

    to_evict = evict_every - ((step + 1) % evict_every or evict_every)
    if to_evict >= active_obs_window:
        return False, active_obs_window

    predicted_len_at_evict = cache_len + to_evict + 1
    return _over_trigger(policy, predicted_len_at_evict, budget, anchor_k), active_obs_window


@torch.no_grad()
def select_tokens(policy, ctx, *, cache=None, attn_history=None, return_debug=False):
    """Return kept slot indices for one request (head-wise list for cache-aware policies).

    ``policy`` may be wrapped with anchors (``strategies.anchors.with_anchors``).
    With ``return_debug`` also returns a dict of ``backend_keep`` (base policy
    choice), ``anchor_extra`` (rescued slots), ``policy_score`` and ``sig_score``.
    """
    n = ctx.slot_pos.numel()
    dev = ctx.slot_pos.device
    if n <= ctx.budget:
        idx = torch.arange(n, device=dev)
        if return_debug:
            return idx, {
                "backend_keep": idx,
                "anchor_extra": torch.empty(0, dtype=torch.long, device=dev),
                "sig_score": None,
                "policy_score": ctx.importance,
            }
        return idx

    decision = EvictionController(policy).decide(
        ctx, cache=cache, attn_history=attn_history, return_debug=return_debug,
    )
    idx = decision.keep_indices
    if not return_debug:
        return idx
    policy_score = decision.policy_score
    if policy_score is None:
        policy_score = torch.full((n,), float("nan"), device=dev)
    return idx, {
        "backend_keep": representative_indices(decision.base_indices),
        "anchor_extra": decision.rescue_indices,
        "sig_score": decision.anchor_score,
        "policy_score": policy_score,
    }


@torch.no_grad()
def select_tokens_batch(policy, features, n, *, budget, recent, sink, policy_params,
                        cache, attn_history):
    """Batched selection over ``features`` (a ``[B, capacity]`` ``SlotFeatures``).

    Cache-aware policies select for the whole batch at once; slot-level
    policies select per request over that request's valid slots.
    """
    valid_mask = features.valid[:, :n]
    if policy.cache_aware:
        params = dict(policy_params or {})
        params["_valid_mask"] = valid_mask
        return policy.select_from_cache(cache, attn_history, n, budget, params)

    selections = []
    for row in range(valid_mask.shape[0]):
        valid_idx = valid_mask[row].nonzero(as_tuple=True)[0]
        ctx = features.context(
            n, budget=budget, recent=recent, sink=sink, params=policy_params,
            row=row, slots=valid_idx,
        )
        selections.append(valid_idx[select_tokens(policy, ctx)])

    # Rows keep different numbers of valid slots (e.g. a finished request with
    # masked filler, or anchors rescuing fewer than anchor_k tokens).  The
    # dense batched cache needs one length, so pad short rows with masked slots.
    target_length = max(int(item.numel()) for item in selections)
    indices = []
    for row, selected in enumerate(selections):
        if selected.numel() < target_length:
            invalid_idx = (~valid_mask[row]).nonzero(as_tuple=True)[0]
            selected = torch.cat([selected, invalid_idx[: target_length - selected.numel()]])
        indices.append(selected)
    lengths = {int(item.numel()) for item in indices}
    if len(lengths) != 1:
        raise RuntimeError(f"batched selections must have equal lengths, got {sorted(lengths)}")
    return torch.stack(indices)
