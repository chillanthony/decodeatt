"""Teacher-forced NLL scoring under token-level KV eviction (TideProbe diagnostics).

``score_trace_token`` replays a fixed generation trace with the same eviction
schedule as ``generate_token_evict`` and accumulates NLL on correction spans
versus the rest of the trace.
"""
from __future__ import annotations

import torch

from kveviction.attention_logits import attention_logit_rows, supports_attention_logits
from kveviction.backends import ExecutionBackend, TransformersExecutionBackend
from kveviction.cache_manager import CacheManager, TransformersCacheManager
from kveviction.observations import AttentionObservationProvider
from kveviction.sampling import entropy
from kveviction.selection import (
    eviction_due,
    is_headwise_selection,
    representative_indices,
    select_tokens,
)
from kveviction.slot_features import SlotFeatures
from kveviction.strategies.anchors import with_anchors
from kveviction.strategies.token import get_policy


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
    """NLL 兜底:token 级淘汰下 teacher-forced 喂真 trace,测纠错 span 的 NLL。
    与 generate_token_evict 同淘汰逻辑,但不采样、喂 gen_ids,累计纠错/非纠错 NLL。"""
    device = next(model.parameters()).device
    execution_backend = execution_backend or TransformersExecutionBackend(model)
    cache_manager = cache_manager or TransformersCacheManager()
    torch.manual_seed(int(seed))
    full_ids = full_ids.to(device)
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    P = prompt_len
    gen_ids = full_ids[0, P:]
    Lgen = gen_ids.numel()
    corr = torch.zeros(Lgen, dtype=torch.bool)
    for s in refl_steps:
        corr[s: min(s + span_len, Lgen)] = True
    anchor_k = int(anchor_frac * budget)
    use_sig = anchor_mode == "sig"
    sig_t = (torch.tensor(sig[0], device=device), float(sig[1])) if use_sig else None
    policy = with_anchors(get_policy(backend), anchor_mode, anchor_k, sig_t)
    policy_params = policy_params or {}
    track = budget < 10 ** 8 and not policy.never_evict
    use_attention_logits = policy.needs_attn_history and supports_attention_logits(model)
    observation_provider = AttentionObservationProvider(
        decay=obs_decay, needs_history=policy.needs_attn_history
    )

    N = P + Lgen
    feats = SlotFeatures(N, device, track_sig=use_sig)
    feats.slot_pos[:P] = torch.arange(P, device=device)
    feats.slot_ids[:P] = full_ids[0, :P]
    n = P
    out = execution_backend.prefill(
        full_ids[:, :P], use_cache=True, output_attentions=False
    )
    cache = out.past_key_values
    logits = out.logits[:, -1]
    nll_c, nll_n, c_c, c_n = 0.0, 0.0, 0, 0
    attn_history = []
    nll_values = []
    eviction_events = []
    position_maps = None
    if debug:
        cache_view = TransformersCacheManager._view(cache)
        position_maps = []
        initial_positions = torch.arange(P, device=device)
        for layer in cache_view.layers:
            mapping = torch.empty((layer.keys.shape[1], N), dtype=torch.long, device=device)
            mapping[:, :P] = initial_positions
            position_maps.append(mapping)

    for t in range(Lgen):
        real = int(gen_ids[t])
        nll = -float(torch.log_softmax(logits.float(), -1)[0, real])
        if debug:
            nll_values.append(nll)
        if corr[t]: nll_c += nll; c_c += 1
        else: nll_n += nll; c_n += 1
        true_pos = P + t
        to_evict = evict_every - ((t + 1) % evict_every or evict_every)
        active_obs_window = policy.observation_window(policy_params, obs_window)
        want_observe = track and active_obs_window > 0 and to_evict < active_obs_window
        want_logits = want_observe and policy.needs_attn_history and use_attention_logits
        want_attn = want_observe and not want_logits
        position_ids = torch.tensor([[true_pos]], device=device)
        out = execution_backend.decode(
            full_ids[:, P + t:P + t + 1],
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
            output_hidden_states=want_logits,
            attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
            position_ids=position_ids,
            cache_position=torch.tensor([n], device=device),
        )
        cache = out.past_key_values
        logits = out.logits[:, -1]
        # anchor criterion approximates current logit entropy
        feats.append(n, pos=true_pos, token=real, entropy=entropy(logits))
        if position_maps is not None:
            for mapping in position_maps:
                mapping[:, n] = true_pos
        n += 1
        if want_logits:
            rows = attention_logit_rows(model, out, cache, position_ids, n)
            if rows is None:
                raise RuntimeError(
                    "attention-logit extraction failed; use a supported RoPE decoder or disable cache-aware baselines"
                )
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:
            observed = observation_provider.observe(
                out, cache=None, n=n, position_ids=position_ids, model=model,
            )
            if policy.needs_attn_history:
                attn_history.append(observed["history"])
                attn_history = attn_history[-active_obs_window:]
            feats.observe(n, observed["importance"], obs_decay)
        if eviction_due(policy, track=track, step=t, evict_every=evict_every,
                        cache_len=n, budget=budget, anchor_k=anchor_k):
            selection_cache = TransformersCacheManager._view(cache)
            ctx = feats.context(n, budget=budget, recent=recent, sink=sink, params=policy_params)
            idx = select_tokens(policy, ctx, cache=selection_cache, attn_history=attn_history)
            before_n = n
            kept_maps = kept_counts = evicted_counts = None
            if position_maps is not None:
                kept_maps, kept_counts, evicted_counts = _selection_position_debug(
                    position_maps, n, idx
                )
            cache, k = cache_manager.compact(cache, idx)
            feats.compact(representative_indices(idx), before_n)
            if position_maps is not None:
                for mapping, kept in zip(position_maps, kept_maps):
                    mapping[:, :k] = kept
                if before_n > k:
                    eviction_events.append(
                        {
                            "eviction_step": t + 1,
                            "cache_len_before": before_n,
                            "cache_len_after": k,
                            "evicted": before_n - k,
                            "kept_positions": [position for position, _ in kept_counts],
                            "evicted_positions": [position for position, _ in evicted_counts],
                            "kept_position_counts": kept_counts,
                            "evicted_position_counts": evicted_counts,
                        }
                    )
            n = k
            if policy.needs_attn_history:
                attn_history = []
    result = {
        "nll_corr": nll_c / max(c_c, 1),
        "n_corr": c_c,
        "nll_noncorr": nll_n / max(c_n, 1),
        "n_noncorr": c_n,
    }
    if debug:
        result.update(
            {
                "per_token_nll": torch.tensor(nll_values, dtype=torch.float32),
                "eviction_events": eviction_events,
                "final_cache_len": n,
                "seed": int(seed),
            }
        )
    return result
