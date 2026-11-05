"""Eviction scheduling and slot selection shared by the decoding loops."""
from __future__ import annotations

import torch

from kveviction.attention_logits import model_layers
from kveviction.cache_manager import _is_headwise as is_headwise_selection
from kveviction.cache_manager import _representative_indices as representative_indices
from kveviction.cache_manager import _representative_indices_batch as representative_indices_batch
from kveviction.controller import EvictionController
from kveviction.strategies.token import SelectionContext, get_policy

__all__ = [
    "is_headwise_selection", "representative_indices", "representative_indices_batch",
    "selection_length", "key_reps", "is_cache_aware", "eviction_due",
    "should_collect_observation", "select_tokens", "select_tokens_batch",
]


def selection_length(idx) -> int:
    if is_headwise_selection(idx):
        return int(idx[0].shape[-1])
    return int(idx.numel())


def key_reps(cache, device):
    """每槽代表 key 向量 [n, D]（层/KV头平均）。"""
    acc = None
    for layer in cache.layers:
        k = layer.keys[0].mean(0).float()    # [n, D]
        acc = k if acc is None else acc + k
    return acc / len(cache.layers)


def is_cache_aware(policy) -> bool:
    return policy.needs_cache or policy.needs_attn_history


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
def select_tokens(imp, cum_imp, win_imp, ent, key_rep, slot_pos, budget, recent, sink,
                  backend, anchor_mode, anchor_k, rng, cum=None, con=None, sig=None,
                  policy_params=None, cache=None, attn_history=None,
                  model=None, return_debug=False):
    """返回应保留的**槽下标** LongTensor。imp/ent/key_rep/slot_pos 均按当前缓存槽对齐。
    anchor_mode="sig" 时用完整 4 维签名(熵,cum_attn,位置,concentration)打分,
    sig=(w[4], b);cum/con 为每槽 cum_attn/concentration。"""
    n = slot_pos.numel()
    dev = slot_pos.device
    if n <= budget:
        idx = torch.arange(n, device=dev)
        if return_debug:
            return idx, {
                "backend_keep": idx,
                "anchor_extra": torch.empty(0, dtype=torch.long, device=dev),
                "sig_score": None,
                "policy_score": imp,
            }
        return idx

    policy = get_policy(backend)
    runtime_params = dict(policy_params or {})
    if model is not None:
        runtime_params["_model"] = model
        layers = model_layers(model)
        if layers is not None:
            runtime_params["_layers"] = layers
    ctx = SelectionContext(
        importance=imp,
        cumulative_attention=cum_imp,
        window_attention=win_imp,
        key_reps=key_rep,
        slot_pos=slot_pos,
        budget=budget,
        recent=recent,
        sink=sink,
        params=runtime_params,
    )
    decision = EvictionController(backend, params=runtime_params).decide(
        ctx, cache=cache, attn_history=attn_history, model=model,
        anchor_mode=anchor_mode, anchor_k=anchor_k,
        anchor_features={"entropy": ent, "cumulative": cum, "concentration": con},
        sig=sig,
        return_debug=return_debug,
    )
    backend_idx = decision.keep_indices
    policy_score = decision.policy_score
    if is_headwise_selection(backend_idx):
        if anchor_mode != "none" and anchor_k > 0:
            raise ValueError("anchor modes are not supported for head-wise cache-aware strategies")
        if return_debug:
            return backend_idx, {
                "backend_keep": representative_indices(backend_idx),
                "anchor_extra": torch.empty(0, dtype=torch.long, device=dev),
                "sig_score": None,
                "policy_score": policy_score
                if policy_score is not None
                else torch.full((n,), float("nan"), device=dev),
            }
        return backend_idx

    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    keep[backend_idx] = True
    idx = keep.nonzero(as_tuple=True)[0]
    if not return_debug:
        return idx
    if policy_score is None:
        policy_score = policy.scores(ctx)
    return idx, {
        "backend_keep": idx,
        "anchor_extra": decision.rescue_indices,
        "sig_score": None,
        "policy_score": policy_score,
    }


@torch.no_grad()
def select_tokens_batch(
    imp,
    cum_imp,
    win_imp,
    ent,
    slot_pos,
    budget,
    recent,
    sink,
    backend,
    anchor_mode,
    anchor_k,
    *,
    policy_params,
    cache,
    attn_history,
    model,
    valid_mask=None,
):
    policy = get_policy(backend)
    if is_cache_aware(policy):
        if anchor_mode != "none" and anchor_k > 0:
            raise ValueError("anchor modes are not supported for batched cache-aware strategies")
        runtime_params = dict(policy_params or {})
        runtime_params["_model"] = model
        if valid_mask is not None:
            runtime_params["_valid_mask"] = valid_mask
        return policy.select_from_cache(
            cache, attn_history, slot_pos.shape[-1], budget, runtime_params
        )

    indices = []
    for row in range(slot_pos.shape[0]):
        if valid_mask is None:
            valid_idx = torch.arange(slot_pos.shape[1], device=slot_pos.device)
        else:
            valid_idx = valid_mask[row].nonzero(as_tuple=True)[0]
        selected_valid = select_tokens(
            imp[row, valid_idx], cum_imp[row, valid_idx], win_imp[row, valid_idx],
            ent[row, valid_idx], None, slot_pos[row, valid_idx],
            budget, recent, sink, backend, anchor_mode, anchor_k, None,
            policy_params=policy_params, cache=None, attn_history=None, model=model,
        )
        selected = valid_idx[selected_valid]
        target_length = min(budget + anchor_k, slot_pos.shape[1])
        if selected.numel() < target_length:
            invalid_idx = (
                (~valid_mask[row]).nonzero(as_tuple=True)[0]
                if valid_mask is not None
                else torch.empty(0, dtype=torch.long, device=slot_pos.device)
            )
            selected = torch.cat([selected, invalid_idx[: target_length - selected.numel()]])
        indices.append(selected)
    lengths = {int(item.numel()) for item in indices}
    if len(lengths) != 1:
        raise RuntimeError(f"batched selections must have equal lengths, got {sorted(lengths)}")
    return torch.stack(indices)
