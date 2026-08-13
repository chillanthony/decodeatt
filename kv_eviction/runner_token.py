"""Token-level KV eviction decoding loop.

后端:
  snapkv —— 重要性(观察窗注意力 maxpool)top-B + recent window + sink。
  rkv    —— 论文版 R-KV: observation attention importance + per-head key redundancy。
  h2o    —— 累计注意力 top-B + recent window + sink。
  window —— 观察窗注意力 top-B + recent window + sink。
  random —— 随机中段 token + recent window + sink。
锚点保护(anchor=高熵 forking token):在预算外额外钉 top-k 高熵 token;
  对照 mode:none / anchor / random / lowent。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from kv_eviction.strategies.rkv_official import compute_attention_scores
from kv_eviction.strategies.token import SelectionContext, get_policy


def _sample(logits, do_sample, temperature, top_p):
    """采样或贪心。logits: [1, vocab]。"""
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


def _sample_batch(logits, do_sample, temperature, top_p, generators):
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


def _entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return float(-(torch.where(p > 0, p * logp, torch.zeros_like(p))).sum())


def _entropy_batch(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return -(torch.where(p > 0, p * logp, torch.zeros_like(p))).sum(-1)


@torch.no_grad()
def _select_tokens(imp, cum_imp, win_imp, ent, key_rep, slot_pos, budget, recent, sink,
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
        layers = _model_layers(model)
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
    if policy.needs_cache or policy.needs_attn_history:
        if return_debug:
            backend_idx, policy_debug = policy.select_from_cache(
                cache, attn_history, n, budget, runtime_params, ctx=ctx, return_debug=True
            )
            policy_score = policy_debug["policy_score"]
        else:
            backend_idx = policy.select_from_cache(
                cache, attn_history, n, budget, runtime_params, ctx=ctx
            )
            policy_score = None
    else:
        backend_idx = policy.select_keep(ctx)
        policy_score = policy.scores(ctx) if return_debug else None
    if _is_headwise_selection(backend_idx):
        if anchor_mode != "none" and anchor_k > 0:
            raise ValueError("anchor modes are not supported for head-wise cache-aware strategies")
        idx = backend_idx
        if return_debug:
            return idx, {
                "backend_keep": _representative_indices(backend_idx),
                "anchor_extra": torch.empty(0, dtype=torch.long, device=dev),
                "sig_score": None,
                "policy_score": policy_score
                if policy_score is not None
                else torch.full((n,), float("nan"), device=dev),
            }
        return idx

    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    keep[backend_idx] = True
    backend_keep = keep.clone()
    anchor_extra = torch.empty(0, dtype=torch.long, device=dev)
    sig_score = None
    if return_debug and policy_score is None:
        policy_score = policy.scores(ctx)

    # 锚点保护:预算外额外钉 anchor_k 个槽
    if anchor_mode != "none" and anchor_k > 0:
        drop = (~keep).nonzero(as_tuple=True)[0]
        if drop.numel():
            if anchor_mode == "anchor":                 # 熵单维(旧)
                extra = drop[torch.argsort(ent[drop], descending=True)][:anchor_k]
            elif anchor_mode == "sig":                  # 完整 4 维签名(新)
                n = slot_pos.numel()
                pos = slot_pos.float() / max(int(slot_pos.max()), 1)
                feat = torch.stack([ent, cum, pos, con], dim=1)  # [n,4]
                sig_score = feat @ sig[0] + sig[1]
                extra = drop[torch.argsort(sig_score[drop], descending=True)][:anchor_k]
            elif anchor_mode == "lowent":
                extra = drop[torch.argsort(ent[drop])][:anchor_k]
            else:                            # random
                extra = drop[torch.randperm(drop.numel(), device=dev)][:anchor_k]
            anchor_extra = extra
            keep[extra] = True
    idx = keep.nonzero(as_tuple=True)[0]
    if return_debug:
        return idx, {
            "backend_keep": backend_keep.nonzero(as_tuple=True)[0],
            "anchor_extra": anchor_extra,
            "sig_score": sig_score,
            "policy_score": policy_score,
        }
    return idx


def _take_float(x):
    return [float(v) for v in x.detach().float().cpu().tolist()]


def _take_int(x):
    return [int(v) for v in x.detach().cpu().tolist()]


def _token_samples(tokenizer, slot_ids, idx, limit=64):
    if limit <= 0:
        return []
    idx = idx[:limit].detach().cpu().tolist()
    ids = slot_ids.detach().cpu().tolist()
    return [{"slot": int(i), "token_id": int(ids[i]),
             "text": tokenizer.decode([int(ids[i])], skip_special_tokens=False)}
            for i in idx]


def _key_reps(cache, device):
    """每槽代表 key 向量 [n, D]（层/KV头平均）。"""
    acc = None
    for layer in cache.layers:
        k = layer.keys[0].mean(0).float()    # [n, D]
        acc = k if acc is None else acc + k
    return acc / len(cache.layers)


def _is_headwise_selection(idx) -> bool:
    return isinstance(idx, list)


def _representative_indices(idx):
    if _is_headwise_selection(idx):
        return idx[0][0]
    return idx


def _representative_indices_batch(idx):
    """Return one token-index row per request from global or head-wise indices."""
    if _is_headwise_selection(idx):
        layer_idx = idx[0]
        if layer_idx.ndim == 2:
            return layer_idx[0].unsqueeze(0)
        return layer_idx[:, 0, :]
    if idx.ndim == 1:
        return idx.unsqueeze(0)
    return idx


class _CacheLayerView:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _CacheView:
    def __init__(self, layers: list[_CacheLayerView]):
        self.layers = layers


def _cache_view(cache):
    if cache is None or hasattr(cache, "layers"):
        return cache
    if isinstance(cache, tuple):
        return _CacheView([_CacheLayerView(keys, values) for keys, values in cache])
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return _CacheView([
            _CacheLayerView(keys, values)
            for keys, values in zip(cache.key_cache, cache.value_cache)
        ])
    return cache


def _should_collect_observation(
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
    active_obs_window = policy.observation_window(policy_params, obs_window)
    if not track or active_obs_window <= 0:
        return False, active_obs_window

    to_evict = evict_every - ((step + 1) % evict_every or evict_every)
    if to_evict >= active_obs_window:
        return False, active_obs_window

    cache_aware = policy.needs_cache or policy.needs_attn_history
    trigger_len = budget if cache_aware else budget + anchor_k
    predicted_len_at_evict = cache_len + to_evict + 1
    will_trigger = (
        predicted_len_at_evict >= trigger_len
        if cache_aware
        else predicted_len_at_evict > trigger_len
    )
    return will_trigger, active_obs_window


def _selection_length(idx) -> int:
    if _is_headwise_selection(idx):
        return int(idx[0].shape[-1])
    return int(idx.numel())


def _compact_cache(cache, idx):
    if _is_headwise_selection(idx):
        for layer, layer_idx in zip(cache.layers, idx):
            layer_idx = layer_idx.to(layer.keys.device)
            head_dim = layer.keys.shape[-1]
            gather_idx = layer_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, head_dim)
            layer.keys = layer.keys.gather(2, gather_idx).contiguous()
            layer.values = layer.values.gather(2, gather_idx).contiguous()
        return int(idx[0].shape[-1])

    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
    return int(idx.numel())


def _compact_cache_update(cache, idx):
    if hasattr(cache, "layers"):
        return cache, _compact_cache(cache, idx)

    if isinstance(cache, tuple):
        compacted = []
        if _is_headwise_selection(idx):
            for (keys, values), layer_idx in zip(cache, idx):
                layer_idx = layer_idx.to(keys.device)
                head_dim = keys.shape[-1]
                gather_idx = layer_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, head_dim)
                compacted.append((
                    keys.gather(2, gather_idx).contiguous(),
                    values.gather(2, gather_idx).contiguous(),
                ))
            return tuple(compacted), int(idx[0].shape[-1])

        for keys, values in cache:
            local_idx = idx.to(keys.device)
            compacted.append((
                keys.index_select(2, local_idx).contiguous(),
                values.index_select(2, local_idx).contiguous(),
            ))
        return tuple(compacted), int(idx.numel())

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if _is_headwise_selection(idx):
            for layer_i, layer_idx in enumerate(idx):
                keys = cache.key_cache[layer_i]
                values = cache.value_cache[layer_i]
                layer_idx = layer_idx.to(keys.device)
                head_dim = keys.shape[-1]
                gather_idx = layer_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, head_dim)
                cache.key_cache[layer_i] = keys.gather(2, gather_idx).contiguous()
                cache.value_cache[layer_i] = values.gather(2, gather_idx).contiguous()
            return cache, int(idx[0].shape[-1])

        for layer_i, keys in enumerate(cache.key_cache):
            local_idx = idx.to(keys.device)
            cache.key_cache[layer_i] = keys.index_select(2, local_idx).contiguous()
            cache.value_cache[layer_i] = cache.value_cache[layer_i].index_select(2, local_idx).contiguous()
        return cache, int(idx.numel())

    return cache, _compact_cache(cache, idx)


def _compact_cache_update_batch(cache, idx):
    """Compact a batched cache with per-request, optionally per-head indices."""
    cache_view = _cache_view(cache)
    batch_size = cache_view.layers[0].keys.shape[0]
    tuple_layers = [] if isinstance(cache, tuple) else None
    for layer_i, layer in enumerate(cache_view.layers):
        keys = layer.keys
        values = layer.values
        if _is_headwise_selection(idx):
            layer_idx = idx[layer_i]
            if layer_idx.ndim == 2:
                layer_idx = layer_idx.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            layer_idx = idx
            if layer_idx.ndim == 1:
                layer_idx = layer_idx.unsqueeze(0).expand(batch_size, -1)
            layer_idx = layer_idx.unsqueeze(1).expand(-1, keys.shape[1], -1)
        gather_idx = layer_idx.to(keys.device).unsqueeze(-1).expand(
            -1, -1, -1, keys.shape[-1]
        )
        compact_keys = keys.gather(2, gather_idx).contiguous()
        compact_values = values.gather(2, gather_idx).contiguous()
        if hasattr(cache, "layers"):
            cache.layers[layer_i].keys = compact_keys
            cache.layers[layer_i].values = compact_values
        elif isinstance(cache, tuple):
            tuple_layers.append((compact_keys, compact_values))
        else:
            cache.key_cache[layer_i] = compact_keys
            cache.value_cache[layer_i] = compact_values
    if tuple_layers is not None:
        cache = tuple(tuple_layers)
    return cache, int(_representative_indices_batch(idx).shape[-1])


def _base_model(model):
    return getattr(model, "model", getattr(model, "transformer", model))


def _model_layers(model):
    base = _base_model(model)
    return getattr(base, "layers", getattr(base, "h", None))


def _cache_length_summary(cache, physical_len: int) -> dict:
    cache = _cache_view(cache)
    if cache is None:
        lengths = torch.tensor([float(physical_len)])
    else:
        lengths = torch.cat([
            torch.full((layer.keys.shape[1],), float(physical_len), device=layer.keys.device)
            for layer in cache.layers
        ])

    total = float(lengths.sum())
    probs = lengths / max(total, 1e-12)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum()) if lengths.numel() else 0.0
    max_len = float(lengths.max()) if lengths.numel() else 0.0
    return {
        "mean_effective_cache_len": float(lengths.mean()) if lengths.numel() else 0.0,
        "max_effective_cache_len": max_len,
        "min_effective_cache_len": float(lengths.min()) if lengths.numel() else 0.0,
        "total_effective_kv_tokens": total,
        "effective_kv_tokens_per_layer_head": [
            [int(physical_len)] * int(layer.keys.shape[1])
            for layer in cache.layers
        ] if cache is not None else [[int(physical_len)]],
        "head_budget_mean": float(lengths.mean()) if lengths.numel() else 0.0,
        "head_budget_std": float(lengths.std(unbiased=False)) if lengths.numel() else 0.0,
        "head_budget_min": float(lengths.min()) if lengths.numel() else 0.0,
        "head_budget_max": max_len,
        "head_budget_entropy": entropy,
        "num_underfilled_heads": int((lengths < max_len).sum()) if lengths.numel() else 0,
    }


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_q(query_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query_states * cos) + (_rotate_half(query_states) * sin)


def _supports_attention_logits(model) -> bool:
    layers = _model_layers(model)
    base = _base_model(model)
    if layers is None or not hasattr(base, "rotary_emb"):
        return False
    if len(layers) == 0:
        return False
    attn = getattr(layers[0], "self_attn", None)
    return attn is not None and hasattr(attn, "q_proj") and hasattr(attn, "head_dim")


def _attention_logit_rows(model, outputs, cache, position_ids, n):
    """Compute official pre-softmax attention rows from layer inputs and cache keys."""
    if not getattr(outputs, "hidden_states", None):
        return None
    layers = _model_layers(model)
    base = _base_model(model)
    if layers is None or not hasattr(base, "rotary_emb"):
        return None

    rows = []
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj") or not hasattr(attn, "head_dim"):
            return None
        hidden = outputs.hidden_states[layer_idx][:, -1:, :]
        q = attn.q_proj(hidden)
        q = q.view(*hidden.shape[:-1], -1, attn.head_dim).transpose(1, 2)
        cos, sin = base.rotary_emb(hidden, position_ids)
        q = _apply_rotary_q(q, cos, sin)
        keys = cache.layers[layer_idx].keys[:, :, :n, :]
        rows.append(compute_attention_scores(q, keys)[0, :, -1, :].detach().float())
    return rows


def _attention_logit_rows_batch(model, outputs, cache, position_ids, n):
    """Batched counterpart returning one ``[B,H,N]`` row per layer."""
    if not getattr(outputs, "hidden_states", None):
        return None
    layers = _model_layers(model)
    base = _base_model(model)
    if layers is None or not hasattr(base, "rotary_emb"):
        return None

    rows = []
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj") or not hasattr(attn, "head_dim"):
            return None
        hidden = outputs.hidden_states[layer_idx][:, -1:, :]
        q = attn.q_proj(hidden)
        q = q.view(*hidden.shape[:-1], -1, attn.head_dim).transpose(1, 2)
        cos, sin = base.rotary_emb(hidden, position_ids)
        q = _apply_rotary_q(q, cos, sin)
        keys = cache.layers[layer_idx].keys[:, :, :n, :]
        rows.append(compute_attention_scores(q, keys)[:, :, -1, :].detach().float())
    return rows


@torch.no_grad()
def _select_tokens_batch(
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
    if policy.needs_cache or policy.needs_attn_history:
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
        selected_valid = _select_tokens(
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


@torch.no_grad()
def generate_token_evict(
    model, tokenizer, input_ids, budget=1024, recent=64, sink=8,
    backend="rkv", anchor_mode="none", anchor_frac=0.05, evict_every=128,
    obs_window=16, obs_decay=0.9, max_new=4096,
    do_sample=True, temperature=0.6, top_p=0.95, seed=0, sig=None,
    policy_params=None,
    debug_path=None, debug_topk=0,
):
    """token 级淘汰生成(忠实 R-KV 架构 + 支持长生成)。

    速度/忠实兼顾:平时不取观测信号; cache-aware 官方 baseline 只在压缩前
    observation window 内用 hidden states + q_proj + cache keys 重算 attention logits。
    缓冲预分配(slot_pos/imp/ent),消掉逐步 torch.cat 的 O(n²),解锁 16k 长生成。
    返回 dict(text, gen_ids, n_evict, final_cache_len)。"""
    device = next(model.parameters()).device
    start_time = time.perf_counter()
    prefill_sec = 0.0
    decode_forward_sec = 0.0
    attention_observation_sec = 0.0
    eviction_sec_total = 0.0
    use_cuda_memory = device.type == "cuda"
    if use_cuda_memory:
        torch.cuda.reset_peak_memory_stats(device)
    if do_sample:
        torch.manual_seed(seed)
    elif backend == "random":
        torch.manual_seed(seed)
    input_ids = input_ids.to(device)
    P = input_ids.shape[1]
    anchor_k = int(anchor_frac * budget)
    policy = get_policy(backend)
    track = budget < 10 ** 8 and not policy.never_evict
    policy_params = policy_params or {}
    use_attention_logits = policy.needs_attn_history and _supports_attention_logits(model)

    use_sig = anchor_mode == "sig"                    # 完整 4 维签名锚点
    N = P + max_new
    slot_pos = torch.zeros(N, dtype=torch.long, device=device); slot_pos[:P] = torch.arange(P, device=device)
    slot_ids = torch.full((N,), -1, dtype=torch.long, device=device); slot_ids[:P] = input_ids[0]
    imp = torch.zeros(N, device=device)
    cum_imp = torch.zeros(N, device=device)
    win_imp = torch.zeros(N, device=device)
    ent = torch.zeros(N, device=device)
    # 完整签名需 cum_attn/concentration → 维护每槽被访问注意力的 sum/cnt/max
    att_sum = torch.zeros(N, device=device); att_cnt = torch.zeros(N, device=device)
    att_max = torch.zeros(N, device=device)
    sig_t = (torch.tensor(sig[0], device=device), float(sig[1])) if use_sig else None
    n = P                                             # 当前缓存长度

    prefill_start = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True, output_attentions=False)
    prefill_sec = time.perf_counter() - prefill_start
    cache = out.past_key_values
    logits = out.logits[:, -1]
    eos = tokenizer.eos_token_id
    gen, n_evict = [], 0
    evict_events = []
    attn_history = []
    debug = {
        "config": {
            "budget": budget, "recent": recent, "sink": sink, "backend": backend,
            "anchor_mode": anchor_mode, "anchor_frac": anchor_frac,
            "anchor_k": anchor_k, "evict_every": evict_every,
            "obs_window": obs_window, "obs_decay": obs_decay, "max_new": max_new,
            "policy_params": policy_params,
        },
        "prompt_len": P,
        "evictions": [],
    } if debug_path else None

    for step in range(max_new):
        true_pos = P + step
        e = _entropy(logits)
        nxt = _sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            break
        # 只在下一次会真正触发压缩前的 observation window 内打开注意力观测。
        want_observe, active_obs_window = _should_collect_observation(
            policy,
            track=track,
            policy_params=policy_params,
            obs_window=obs_window,
            evict_every=evict_every,
            step=step,
            cache_len=n,
            budget=budget,
            anchor_k=anchor_k,
        )
        want_logits = want_observe and policy.needs_attn_history and use_attention_logits
        want_attn = want_observe and not want_logits
        position_ids = torch.tensor([[true_pos]], device=device)
        decode_start = time.perf_counter()
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    use_cache=True, output_attentions=want_attn,
                    output_hidden_states=want_logits,
                    attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=position_ids,
                    cache_position=torch.tensor([n], device=device))
        decode_forward_sec += time.perf_counter() - decode_start
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos[n] = true_pos; slot_ids[n] = nxt; ent[n] = e; imp[n] = 0.0
        cum_imp[n] = 0.0; win_imp[n] = 0.0
        if use_sig:
            att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_logits:
            obs_start = time.perf_counter()
            rows = _attention_logit_rows(model, out, _cache_view(cache), position_ids, n)
            attention_observation_sec += time.perf_counter() - obs_start
            if rows is None:
                raise RuntimeError(
                    "attention-logit extraction failed; use a supported RoPE decoder or disable cache-aware baselines"
                )
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:                                 # 观察窗:全层注意力均值 maxpool 进 imp
            obs_start = time.perf_counter()
            if policy.needs_attn_history:
                attn_history.append([
                    torch.log(a[0, :, -1, :n].detach().float().clamp_min(1e-30))
                    for a in out.attentions
                ])
                attn_history = attn_history[-active_obs_window:]
            row = torch.zeros(n, device=device)
            for a in out.attentions:
                row += a[0, :, -1, :].mean(0).float()
            row /= len(out.attentions)
            imp[:n] = torch.maximum(imp[:n] * obs_decay, row)
            cum_imp[:n] += row
            win_imp[:n] = win_imp[:n] * obs_decay + row
            if use_sig:                               # cum_attn/concentration 统计
                att_sum[:n] += row; att_cnt[:n] += 1.0
                att_max[:n] = torch.maximum(att_max[:n], row)
            attention_observation_sec += time.perf_counter() - obs_start

        cache_aware = policy.needs_cache or policy.needs_attn_history
        trigger_len = budget if cache_aware else budget + anchor_k
        over_trigger = n >= trigger_len if cache_aware else n > trigger_len
        if track and (step + 1) % evict_every == 0 and over_trigger:
            evict_start = time.perf_counter()
            selection_cache = _cache_view(cache)
            before_summary = _cache_length_summary(selection_cache, n)
            key_rep = None if policy.needs_cache else (_key_reps(selection_cache, device) if policy.needs_key_reps else None)
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1)
                con = att_max[:n] / (cum + 1e-9)
            if debug is not None:
                idx, dbg = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                          key_rep, slot_pos[:n], budget, recent,
                                          sink, backend, anchor_mode, anchor_k, None,
                                          cum=cum, con=con, sig=sig_t,
                                          policy_params=policy_params, cache=selection_cache,
                                          attn_history=attn_history,
                                          model=model,
                                          return_debug=True)
            else:
                idx = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                     key_rep, slot_pos[:n], budget, recent,
                                     sink, backend, anchor_mode, anchor_k, None,
                                     cum=cum, con=con, sig=sig_t,
                                     policy_params=policy_params, cache=selection_cache,
                                     attn_history=attn_history, model=model)
            if debug is not None:
                idx_debug = _representative_indices(idx)
                final_keep = torch.zeros(n, dtype=torch.bool, device=device)
                final_keep[idx_debug] = True
                evicted = (~final_keep).nonzero(as_tuple=True)[0]
                score = dbg["sig_score"]
                if score is None:
                    score = torch.full((n,), float("nan"), device=device)
                rescued = dbg["anchor_extra"]
                evicted_by_sig = (
                    evicted[torch.argsort(score[evicted], descending=True)]
                    if evicted.numel() else evicted
                )
                debug["evictions"].append({
                    "step": step + 1,
                    "cache_len_before": n,
                    "cache_len_after": _selection_length(idx),
                    "slot_pos": _take_int(slot_pos[:n]),
                    "token_ids": _take_int(slot_ids[:n]),
                    "imp": _take_float(imp[:n]),
                    "cum_imp": _take_float(cum_imp[:n]),
                    "win_imp": _take_float(win_imp[:n]),
                    "ent": _take_float(ent[:n]),
                    "cum_attn": _take_float(cum) if cum is not None else None,
                    "concentration": _take_float(con) if con is not None else None,
                    "sig_score": _take_float(score),
                    "policy_score": _take_float(dbg["policy_score"]),
                    "backend_keep_slots": _take_int(dbg["backend_keep"]),
                    "anchor_extra_slots": _take_int(rescued),
                    "final_keep_slots": _take_int(idx_debug),
                    "evicted_slots": _take_int(evicted),
                    "anchor_extra_text": _token_samples(tokenizer, slot_ids[:n], rescued, debug_topk),
                    "evicted_top_sig_text": _token_samples(tokenizer, slot_ids[:n], evicted_by_sig, debug_topk),
                })
            before_n = n
            idx_slots = _representative_indices(idx)
            cache, k = _compact_cache_update(cache, idx)
            after_summary = _cache_length_summary(cache, k)
            slot_pos[:k] = slot_pos[idx_slots]; slot_ids[:k] = slot_ids[idx_slots]; imp[:k] = imp[idx_slots]
            cum_imp[:k] = cum_imp[idx_slots]; win_imp[:k] = win_imp[idx_slots]; ent[:k] = ent[idx_slots]
            if use_sig:
                att_sum[:k] = att_sum[idx_slots]; att_cnt[:k] = att_cnt[idx_slots]; att_max[:k] = att_max[idx_slots]
            n = k
            if policy.needs_attn_history:
                attn_history = []
            n_evict += 1
            event_eviction_sec = time.perf_counter() - evict_start
            eviction_sec_total += event_eviction_sec
            evict_events.append({
                "step": step + 1,
                "cache_len_before": int(before_n),
                "cache_len_after": int(k),
                "compression_ratio": float(k / max(before_n, 1)),
                "evicted": int(before_n - k),
                "effective_cache_len_before_mean": before_summary["mean_effective_cache_len"],
                "effective_cache_len_after_mean": after_summary["mean_effective_cache_len"],
                "effective_cache_len_before_max": before_summary["max_effective_cache_len"],
                "effective_cache_len_after_max": after_summary["max_effective_cache_len"],
                "effective_compression_ratio": float(
                    after_summary["total_effective_kv_tokens"]
                    / max(before_summary["total_effective_kv_tokens"], 1.0)
                ),
                "effective_evicted": float(
                    before_summary["total_effective_kv_tokens"]
                    - after_summary["total_effective_kv_tokens"]
                ),
                "head_budget_mean": after_summary["head_budget_mean"],
                "head_budget_std": after_summary["head_budget_std"],
                "head_budget_min": after_summary["head_budget_min"],
                "head_budget_max": after_summary["head_budget_max"],
                "head_budget_entropy": after_summary["head_budget_entropy"],
                "num_underfilled_heads": after_summary["num_underfilled_heads"],
                "eviction_sec": event_eviction_sec,
            })

    text = tokenizer.decode(gen, skip_special_tokens=True)
    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if use_cuda_memory else None
    mean_compression = (
        sum(event["compression_ratio"] for event in evict_events) / len(evict_events)
        if evict_events else 1.0
    )
    final_length_summary = _cache_length_summary(cache, n)
    compression_ratios = [event["compression_ratio"] for event in evict_events]
    effective_compression_ratios = [event["effective_compression_ratio"] for event in evict_events]
    total_evicted_tokens = sum(event["evicted"] for event in evict_events)
    total_effective_evicted_tokens = sum(event["effective_evicted"] for event in evict_events)
    mean_evicted_per_event = _mean([event["evicted"] for event in evict_events])
    mean_effective_evicted_per_event = _mean([event["effective_evicted"] for event in evict_events])
    cache_len_curve = [event["cache_len_after"] for event in evict_events]
    effective_cache_len_curve = [event["effective_cache_len_after_mean"] for event in evict_events]
    eviction_sec_mean = eviction_sec_total / n_evict if n_evict else 0.0
    decode_sec = elapsed - prefill_sec
    timing = {
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "decode_forward_sec": decode_forward_sec,
        "attention_observation_sec": attention_observation_sec,
        "eviction_sec_total": eviction_sec_total,
        "eviction_sec_mean": eviction_sec_mean,
        "other_decode_sec": max(0.0, decode_sec - decode_forward_sec - attention_observation_sec - eviction_sec_total),
    }
    extra_metrics = {
        **final_length_summary,
        "total_evicted_tokens": int(total_evicted_tokens),
        "total_effective_evicted_tokens": float(total_effective_evicted_tokens),
        "mean_evicted_per_event": float(mean_evicted_per_event),
        "mean_effective_evicted_per_event": float(mean_effective_evicted_per_event),
        "min_compression_ratio": min(compression_ratios) if compression_ratios else 1.0,
        "max_compression_ratio": max(compression_ratios) if compression_ratios else 1.0,
        "mean_effective_compression_ratio": _mean(effective_compression_ratios, 1.0),
        "min_effective_compression_ratio": min(effective_compression_ratios) if effective_compression_ratios else 1.0,
        "max_effective_compression_ratio": max(effective_compression_ratios) if effective_compression_ratios else 1.0,
        "cache_len_curve": cache_len_curve,
        "effective_cache_len_curve": effective_cache_len_curve,
        **timing,
    }
    if debug is not None:
        debug["generated_text"] = text
        debug["gen_ids"] = [int(x) for x in gen]
        debug["n_evict"] = n_evict
        debug["final_cache_len"] = n
        debug["evict_events"] = evict_events
        debug["elapsed_sec"] = elapsed
        debug["tokens_per_sec"] = len(gen) / elapsed if elapsed > 0 else 0.0
        debug["peak_memory_bytes"] = int(peak_memory) if peak_memory is not None else None
        debug["mean_compression_ratio"] = mean_compression
        debug.update(extra_metrics)
        path = Path(debug_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(debug, ensure_ascii=False, indent=1))
    return {"text": text, "gen_ids": gen,
            "n_evict": n_evict, "final_cache_len": n,
            "elapsed_sec": elapsed,
            "tokens_per_sec": len(gen) / elapsed if elapsed > 0 else 0.0,
            "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
            "mean_compression_ratio": mean_compression,
            "evict_events": evict_events,
            **extra_metrics}


@torch.no_grad()
def generate_token_evict_batch(
    model,
    tokenizer,
    input_ids,
    batch_size,
    budget=1024,
    recent=64,
    sink=8,
    backend="rkv",
    anchor_mode="none",
    anchor_frac=0.05,
    evict_every=128,
    obs_window=16,
    obs_decay=0.9,
    max_new=4096,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    seeds=None,
    policy_params=None,
    input_attention_mask=None,
):
    """Generate independent candidates for one or more prompts in a static batch.

    Heterogeneous prompts use left padding plus a per-request valid-cache mask.
    Cache-aware policies may retain different slots for every request, layer,
    and KV head. Finished requests append masked filler slots while the remaining
    requests continue decoding.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if batch_size == 1 and input_ids.shape[0] == 1 and input_attention_mask is None:
        seed = int((seeds or [0])[0])
        return {
            "candidates": [generate_token_evict(
                model,
                tokenizer,
                input_ids,
                budget=budget,
                recent=recent,
                sink=sink,
                backend=backend,
                anchor_mode=anchor_mode,
                anchor_frac=anchor_frac,
                evict_every=evict_every,
                obs_window=obs_window,
                obs_decay=obs_decay,
                max_new=max_new,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                policy_params=policy_params,
            )]
        }

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if input_ids.shape[0] not in {1, batch_size}:
        raise ValueError(
            f"input batch must contain 1 or {batch_size} prompts, got {input_ids.shape[0]}"
        )
    seeds = list(seeds if seeds is not None else range(batch_size))
    if len(seeds) != batch_size:
        raise ValueError(f"expected {batch_size} seeds, got {len(seeds)}")
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)

    policy = get_policy(backend)
    policy_params = policy_params or {}
    track = budget < 10 ** 8 and not policy.never_evict
    use_attention_logits = policy.needs_attn_history and _supports_attention_logits(model)
    anchor_k = int(anchor_frac * budget)
    physical_prompt_len = int(input_ids.shape[1])
    capacity = physical_prompt_len + max_new
    batched_input_ids = input_ids.expand(batch_size, -1)
    if input_attention_mask is None:
        prompt_attention_mask = torch.ones_like(batched_input_ids, dtype=torch.long)
    else:
        input_attention_mask = input_attention_mask.to(device=device, dtype=torch.long)
        if input_attention_mask.shape[0] == 1 and batch_size > 1:
            input_attention_mask = input_attention_mask.expand(batch_size, -1)
        if input_attention_mask.shape != batched_input_ids.shape:
            raise ValueError(
                "input_attention_mask must match the expanded input_ids shape, got "
                f"{tuple(input_attention_mask.shape)} vs {tuple(batched_input_ids.shape)}"
            )
        prompt_attention_mask = input_attention_mask
    prompt_lengths = prompt_attention_mask.sum(-1).long()
    prefill_position_ids = prompt_attention_mask.cumsum(-1) - 1
    prefill_position_ids.masked_fill_(prompt_attention_mask == 0, 0)
    slot_pos = torch.zeros(batch_size, capacity, dtype=torch.long, device=device)
    slot_pos[:, :physical_prompt_len] = prefill_position_ids
    slot_ids = torch.full((batch_size, capacity), -1, dtype=torch.long, device=device)
    slot_ids[:, :physical_prompt_len] = batched_input_ids
    valid_cache_mask = torch.zeros(batch_size, capacity, dtype=torch.bool, device=device)
    valid_cache_mask[:, :physical_prompt_len] = prompt_attention_mask.bool()
    imp = torch.zeros(batch_size, capacity, device=device)
    cum_imp = torch.zeros_like(imp)
    win_imp = torch.zeros_like(imp)
    ent = torch.zeros_like(imp)
    n = physical_prompt_len

    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefill_start = time.perf_counter()
    out = model(
        input_ids=batched_input_ids,
        attention_mask=prompt_attention_mask,
        position_ids=prefill_position_ids,
        use_cache=True,
        output_attentions=False,
    )
    prefill_sec = time.perf_counter() - prefill_start
    cache = out.past_key_values
    logits = out.logits[:, -1]
    eos = tokenizer.eos_token_id
    filler = eos if eos is not None else tokenizer.pad_token_id
    if filler is None:
        filler = 0
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated = [[] for _ in range(batch_size)]
    attn_history = []
    n_evict = 0
    evict_events = []
    decode_forward_sec = 0.0
    attention_observation_sec = 0.0
    eviction_sec_total = 0.0

    for step in range(max_new):
        active = ~finished
        step_valid = active.clone()
        token_entropy = _entropy_batch(logits)
        next_tokens = _sample_batch(logits, do_sample, temperature, top_p, generators)
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, filler))
        for row in active.nonzero(as_tuple=True)[0].tolist():
            generated[row].append(int(next_tokens[row]))
        if eos is not None:
            finished |= active & next_tokens.eq(eos)
        if bool(finished.all()):
            break

        want_observe, active_obs_window = _should_collect_observation(
            policy,
            track=track,
            policy_params=policy_params,
            obs_window=obs_window,
            evict_every=evict_every,
            step=step,
            cache_len=n,
            budget=budget,
            anchor_k=anchor_k,
        )
        want_logits = want_observe and policy.needs_attn_history and use_attention_logits
        want_attn = want_observe and not want_logits
        position_ids = (prompt_lengths + step).unsqueeze(1)
        step_attention_mask = torch.cat(
            [valid_cache_mask[:, :n], step_valid.unsqueeze(1)], dim=1
        ).long()
        decode_start = time.perf_counter()
        out = model(
            input_ids=next_tokens.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
            output_hidden_states=want_logits,
            attention_mask=step_attention_mask,
            position_ids=position_ids,
            cache_position=torch.tensor([n], device=device),
        )
        decode_forward_sec += time.perf_counter() - decode_start
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos[:, n] = position_ids.squeeze(1)
        slot_ids[:, n] = next_tokens
        valid_cache_mask[:, n] = step_valid
        ent[:, n] = token_entropy
        imp[:, n] = 0.0
        cum_imp[:, n] = 0.0
        win_imp[:, n] = 0.0
        n += 1

        if want_logits:
            observation_start = time.perf_counter()
            rows = _attention_logit_rows_batch(
                model, out, _cache_view(cache), position_ids, n
            )
            attention_observation_sec += time.perf_counter() - observation_start
            if rows is None:
                raise RuntimeError("batched attention-logit extraction failed")
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:
            observation_start = time.perf_counter()
            if policy.needs_attn_history:
                attn_history.append([
                    torch.log(a[:, :, -1, :n].detach().float().clamp_min(1e-30))
                    for a in out.attentions
                ])
                attn_history = attn_history[-active_obs_window:]
            row_score = torch.zeros(batch_size, n, device=device)
            for attention in out.attentions:
                row_score += attention[:, :, -1, :n].mean(1).float()
            row_score /= len(out.attentions)
            imp[:, :n] = torch.maximum(imp[:, :n] * obs_decay, row_score)
            cum_imp[:, :n] += row_score
            win_imp[:, :n] = win_imp[:, :n] * obs_decay + row_score
            attention_observation_sec += time.perf_counter() - observation_start

        cache_aware = policy.needs_cache or policy.needs_attn_history
        trigger_len = budget if cache_aware else budget + anchor_k
        over_trigger = n >= trigger_len if cache_aware else n > trigger_len
        if track and (step + 1) % evict_every == 0 and over_trigger:
            eviction_start = time.perf_counter()
            before_n = n
            idx = _select_tokens_batch(
                imp[:, :n],
                cum_imp[:, :n],
                win_imp[:, :n],
                ent[:, :n],
                slot_pos[:, :n],
                budget,
                recent,
                sink,
                backend,
                anchor_mode,
                anchor_k,
                policy_params=policy_params,
                cache=_cache_view(cache),
                attn_history=attn_history,
                model=model,
                valid_mask=valid_cache_mask[:, :n],
            )
            representative = _representative_indices_batch(idx)
            cache, n = _compact_cache_update_batch(cache, idx)
            slot_pos[:, :n] = torch.gather(slot_pos[:, :before_n], 1, representative)
            slot_ids[:, :n] = torch.gather(slot_ids[:, :before_n], 1, representative)
            imp[:, :n] = torch.gather(imp[:, :before_n], 1, representative)
            cum_imp[:, :n] = torch.gather(cum_imp[:, :before_n], 1, representative)
            win_imp[:, :n] = torch.gather(win_imp[:, :before_n], 1, representative)
            ent[:, :n] = torch.gather(ent[:, :before_n], 1, representative)
            valid_cache_mask[:, :n] = torch.gather(
                valid_cache_mask[:, :before_n], 1, representative
            )
            if policy.needs_attn_history:
                attn_history = []
            n_evict += 1
            event_sec = time.perf_counter() - eviction_start
            eviction_sec_total += event_sec
            evict_events.append({
                "step": step + 1,
                "cache_len_before": int(before_n),
                "cache_len_after": int(n),
                "compression_ratio": float(n / max(before_n, 1)),
                "evicted": int(before_n - n),
                "eviction_sec": event_sec,
            })

    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    batch_total_tokens = sum(len(ids) for ids in generated)
    amortized_elapsed = elapsed / batch_size
    candidates = []
    for row, ids in enumerate(generated):
        candidates.append({
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "gen_ids": ids,
            "seed": int(seeds[row]),
            "prompt_len": int(prompt_lengths[row]),
            "n_evict": n_evict,
            "final_cache_len": n,
            "elapsed_sec": amortized_elapsed,
            "tokens_per_sec": len(ids) / amortized_elapsed if amortized_elapsed > 0 else 0.0,
            "batch_elapsed_sec": elapsed,
            "batch_tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
            "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
            "mean_compression_ratio": (
                sum(event["compression_ratio"] for event in evict_events) / len(evict_events)
                if evict_events else 1.0
            ),
            "evict_events": evict_events,
            "prefill_sec": prefill_sec / batch_size,
            "decode_sec": max(0.0, elapsed - prefill_sec) / batch_size,
            "decode_forward_sec": decode_forward_sec / batch_size,
            "attention_observation_sec": attention_observation_sec / batch_size,
            "eviction_sec_total": eviction_sec_total / batch_size,
            "other_decode_sec": max(
                0.0,
                elapsed - prefill_sec - decode_forward_sec
                - attention_observation_sec - eviction_sec_total,
            ) / batch_size,
        })
    return {
        "candidates": candidates,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
    }


@torch.no_grad()
def score_trace_token(model, tokenizer, full_ids, prompt_len, refl_steps, anchor_mode,
                      budget=1024, recent=64, sink=8, backend="rkv", anchor_frac=0.05,
                      evict_every=128, obs_window=16, obs_decay=0.9, span_len=32, sig=None,
                      policy_params=None):
    """NLL 兜底:token 级淘汰下 teacher-forced 喂真 trace,测纠错 span 的 NLL。
    与 generate_token_evict 同淘汰逻辑,但不采样、喂 gen_ids,累计纠错/非纠错 NLL。"""
    device = next(model.parameters()).device
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
    policy = get_policy(backend)
    policy_params = policy_params or {}
    track = budget < 10 ** 8 and not policy.never_evict
    use_attention_logits = policy.needs_attn_history and _supports_attention_logits(model)

    N = P + Lgen
    slot_pos = torch.zeros(N, dtype=torch.long, device=device); slot_pos[:P] = torch.arange(P, device=device)
    imp = torch.zeros(N, device=device); cum_imp = torch.zeros(N, device=device)
    win_imp = torch.zeros(N, device=device); ent = torch.zeros(N, device=device)
    att_sum = torch.zeros(N, device=device); att_cnt = torch.zeros(N, device=device); att_max = torch.zeros(N, device=device)
    n = P
    out = model(input_ids=full_ids[:, :P], use_cache=True, output_attentions=False)
    cache = out.past_key_values; logits = out.logits[:, -1]
    nll_c, nll_n, c_c, c_n = 0.0, 0.0, 0, 0
    attn_history = []

    for t in range(Lgen):
        real = int(gen_ids[t])
        nll = -float(torch.log_softmax(logits.float(), -1)[0, real])
        if corr[t]: nll_c += nll; c_c += 1
        else: nll_n += nll; c_n += 1
        true_pos = P + t
        to_evict = evict_every - ((t + 1) % evict_every or evict_every)
        active_obs_window = policy.observation_window(policy_params, obs_window)
        want_observe = track and active_obs_window > 0 and to_evict < active_obs_window
        want_logits = want_observe and policy.needs_attn_history and use_attention_logits
        want_attn = want_observe and not want_logits
        position_ids = torch.tensor([[true_pos]], device=device)
        out = model(input_ids=full_ids[:, P + t:P + t + 1], past_key_values=cache, use_cache=True,
                    output_attentions=want_attn, output_hidden_states=want_logits,
                    attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=position_ids, cache_position=torch.tensor([n], device=device))
        cache = out.past_key_values; logits = out.logits[:, -1]
        slot_pos[n] = true_pos; ent[n] = _entropy(logits)   # 锚点判据用当前 logits 熵近似
        imp[n] = 0.0; cum_imp[n] = 0.0; win_imp[n] = 0.0
        att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_logits:
            rows = _attention_logit_rows(model, out, cache, position_ids, n)
            if rows is None:
                raise RuntimeError(
                    "attention-logit extraction failed; use a supported RoPE decoder or disable cache-aware baselines"
                )
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:
            if policy.needs_attn_history:
                attn_history.append([
                    torch.log(a[0, :, -1, :n].detach().float().clamp_min(1e-30))
                    for a in out.attentions
                ])
                attn_history = attn_history[-active_obs_window:]
            row = torch.zeros(n, device=device)
            for a in out.attentions: row += a[0, :, -1, :].mean(0).float()
            row /= len(out.attentions)
            imp[:n] = torch.maximum(imp[:n] * obs_decay, row)
            cum_imp[:n] += row
            win_imp[:n] = win_imp[:n] * obs_decay + row
            if use_sig:
                att_sum[:n] += row; att_cnt[:n] += 1.0; att_max[:n] = torch.maximum(att_max[:n], row)
        cache_aware = policy.needs_cache or policy.needs_attn_history
        trigger_len = budget if cache_aware else budget + anchor_k
        over_trigger = n >= trigger_len if cache_aware else n > trigger_len
        if track and (t + 1) % evict_every == 0 and over_trigger:
            key_rep = None if policy.needs_cache else (_key_reps(cache, device) if policy.needs_key_reps else None)
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1); con = att_max[:n] / (cum + 1e-9)
            idx = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                 key_rep, slot_pos[:n], budget, recent, sink,
                                 backend, anchor_mode, anchor_k, None, cum=cum, con=con, sig=sig_t,
                                 policy_params=policy_params, cache=cache, attn_history=attn_history,
                                 model=model)
            idx_slots = _representative_indices(idx)
            k = _compact_cache(cache, idx)
            slot_pos[:k] = slot_pos[idx_slots]; imp[:k] = imp[idx_slots]
            cum_imp[:k] = cum_imp[idx_slots]; win_imp[:k] = win_imp[idx_slots]; ent[:k] = ent[idx_slots]
            att_sum[:k] = att_sum[idx_slots]; att_cnt[:k] = att_cnt[idx_slots]; att_max[:k] = att_max[idx_slots]
            n = k
            if policy.needs_attn_history:
                attn_history = []
    return {"nll_corr": nll_c / max(c_c, 1), "n_corr": c_c,
            "nll_noncorr": nll_n / max(c_n, 1), "n_noncorr": c_n}


def _smoke(model_path, max_new=200):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    ids = tok.apply_chat_template([{"role": "user", "content": "Compute 12*13 then 7*8."}],
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    for be in ["snapkv", "rkv", "h2o", "window", "random"]:
        for am in ["none", "anchor"]:
            r = generate_token_evict(model, tok, ids, budget=128, recent=32, sink=4,
                                     backend=be, anchor_mode=am, evict_every=32, max_new=max_new)
            print(f"backend={be:6s} anchor={am:6s} gen={len(r['gen_ids'])} "
                  f"evict={r['n_evict']} cache={r['final_cache_len']} (budget128)")
    print("TOKEN SMOKE OK")


if __name__ == "__main__":
    import sys
    _smoke(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
