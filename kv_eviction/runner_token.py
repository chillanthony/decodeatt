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
import types
from pathlib import Path

import torch

from kv_eviction.strategies.rkv_official import compute_attention_scores
from kv_eviction.strategies.token import HeadwiseSelection, SelectionContext, get_policy


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


def _entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return float(-(torch.where(p > 0, p * logp, torch.zeros_like(p))).sum())


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
    return isinstance(idx, (list, HeadwiseSelection))


def _selection_indices(idx):
    return idx.indices if isinstance(idx, HeadwiseSelection) else idx


def _selection_valid_masks(idx):
    return idx.valid_masks if isinstance(idx, HeadwiseSelection) else None


def _representative_indices(idx):
    if _is_headwise_selection(idx):
        indices = _selection_indices(idx)
        return indices[0][0]
    return idx


def _selection_length(idx) -> int:
    if _is_headwise_selection(idx):
        return max(int(layer_idx.shape[-1]) for layer_idx in _selection_indices(idx))
    return int(idx.numel())


def _compact_cache(cache, idx, return_valid_masks: bool = False):
    if _is_headwise_selection(idx):
        indices = _selection_indices(idx)
        valid_masks = _selection_valid_masks(idx)
        if valid_masks is None:
            valid_masks = [
                torch.ones_like(layer_idx, dtype=torch.bool, device=layer_idx.device)
                for layer_idx in indices
            ]
        global_len = max(int(layer_idx.shape[-1]) for layer_idx in indices)
        out_masks = []
        for layer, layer_idx, valid in zip(cache.layers, indices, valid_masks):
            layer_idx = layer_idx.to(layer.keys.device)
            valid = valid.to(layer.keys.device)
            if layer_idx.shape[-1] < global_len:
                pad_len = global_len - layer_idx.shape[-1]
                pad_src = torch.where(valid, layer_idx, torch.zeros_like(layer_idx))
                fallback = pad_src[:, -1:].expand(-1, pad_len)
                layer_idx = torch.cat([layer_idx, fallback], dim=-1)
                valid = torch.cat(
                    [valid, torch.zeros(valid.shape[0], pad_len, dtype=torch.bool, device=valid.device)],
                    dim=-1,
                )
            head_dim = layer.keys.shape[-1]
            gather_idx = layer_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, head_dim)
            layer.keys = layer.keys.gather(2, gather_idx).contiguous()
            layer.values = layer.values.gather(2, gather_idx).contiguous()
            out_masks.append(valid.contiguous())
        if return_valid_masks:
            return global_len, out_masks
        return global_len

    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
    if return_valid_masks:
        return int(idx.numel()), None
    return int(idx.numel())


def _base_model(model):
    return getattr(model, "model", getattr(model, "transformer", model))


def _model_layers(model):
    base = _base_model(model)
    return getattr(base, "layers", getattr(base, "h", None))


def _ensure_headwise_mask_patch(model):
    layers = _model_layers(model)
    if layers is None:
        return False
    patched = False
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_kv_eviction_mask_patched", False):
            patched = True
            continue
        original_forward = attn.forward

        def forward_with_head_mask(self, *args, **kwargs):
            valid = getattr(self, "_kv_eviction_valid_mask", None)
            if valid is not None:
                if "attention_mask" in kwargs:
                    attention_mask = kwargs["attention_mask"]
                elif len(args) >= 3:
                    attention_mask = args[2]
                else:
                    attention_mask = None
                hidden_states = kwargs.get("hidden_states", args[0] if args else None)
                q_len = 1 if hidden_states is None else hidden_states.shape[1]
                groups = int(getattr(self, "num_key_value_groups", 1))
                q_mask = valid.to(device=hidden_states.device if hidden_states is not None else valid.device)
                q_mask = q_mask.repeat_interleave(groups, dim=0).unsqueeze(0).unsqueeze(2)
                q_mask = q_mask.expand(-1, -1, q_len, -1)
                dtype = (
                    attention_mask.dtype
                    if attention_mask is not None and attention_mask.is_floating_point()
                    else (hidden_states.dtype if hidden_states is not None else torch.float32)
                )
                pad = torch.zeros(q_mask.shape, dtype=dtype, device=q_mask.device)
                pad = pad.masked_fill(~q_mask, torch.finfo(dtype).min)
                if attention_mask is None:
                    attention_mask = pad
                else:
                    if attention_mask.shape[1] == 1 and pad.shape[1] != 1:
                        attention_mask = attention_mask.expand(-1, pad.shape[1], -1, -1)
                    attention_mask = attention_mask + pad[:, :, :, : attention_mask.shape[-1]]
                if "attention_mask" in kwargs:
                    kwargs["attention_mask"] = attention_mask
                else:
                    args = list(args)
                    if len(args) >= 3:
                        args[2] = attention_mask
                    else:
                        kwargs["attention_mask"] = attention_mask
                    args = tuple(args)
            return self._kv_eviction_original_forward(*args, **kwargs)

        attn._kv_eviction_original_forward = original_forward
        attn.forward = types.MethodType(forward_with_head_mask, attn)
        attn._kv_eviction_mask_patched = True
        patched = True
    return patched


def _set_headwise_valid_masks(model, valid_masks):
    layers = _model_layers(model)
    if layers is None:
        return
    if valid_masks is None and not any(
        getattr(getattr(layer, "self_attn", None), "_kv_eviction_mask_patched", False)
        for layer in layers
    ):
        return
    _ensure_headwise_mask_patch(model)
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if valid_masks is None:
            attn._kv_eviction_valid_mask = None
        else:
            attn._kv_eviction_valid_mask = valid_masks[layer_idx]


def _append_valid_token(valid_masks):
    if valid_masks is None:
        return None
    return [
        torch.cat(
            [mask, torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device)],
            dim=-1,
        )
        for mask in valid_masks
    ]


def _cache_length_summary(cache, valid_masks, physical_len: int) -> dict:
    if cache is None:
        lengths = torch.tensor([float(physical_len)])
    elif valid_masks is None:
        lengths = torch.cat([
            torch.full((layer.keys.shape[1],), float(physical_len), device=layer.keys.device)
            for layer in cache.layers
        ])
    else:
        lengths = torch.cat([mask[:, :physical_len].sum(dim=-1).float() for mask in valid_masks])

    total = float(lengths.sum())
    probs = lengths / max(total, 1e-12)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum()) if lengths.numel() else 0.0
    max_len = float(lengths.max()) if lengths.numel() else 0.0
    return {
        "mean_effective_cache_len": float(lengths.mean()) if lengths.numel() else 0.0,
        "max_effective_cache_len": max_len,
        "min_effective_cache_len": float(lengths.min()) if lengths.numel() else 0.0,
        "total_effective_kv_tokens": total,
        "effective_kv_tokens_per_layer_head": (
            [
                [int(v) for v in mask[:, :physical_len].sum(dim=-1).detach().cpu().tolist()]
                for mask in valid_masks
            ]
            if valid_masks is not None
            else [
                [int(physical_len)] * int(layer.keys.shape[1])
                for layer in cache.layers
            ] if cache is not None else [[int(physical_len)]]
        ),
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


def _attention_logit_rows(model, outputs, cache, position_ids, n, valid_masks=None):
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
        score = compute_attention_scores(q, keys)[0, :, -1, :].detach().float()
        if valid_masks is not None:
            valid = valid_masks[layer_idx][:, :n].to(score.device)
            score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        rows.append(score)
    return rows


@torch.no_grad()
def generate_token_evict(
    model, tokenizer, input_ids, budget=1024, recent=64, sink=8,
    backend="rkv", anchor_mode="none", anchor_frac=0.05, evict_every=128,
    obs_window=16, obs_decay=0.9, max_new=4096,
    do_sample=True, temperature=0.6, top_p=0.95, seed=0, sig=None,
    policy_params=None,
    debug_path=None, debug_topk=64,
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
    cache_valid_masks = None
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
        # 只在"下次压缩前 obs_window 步内"开注意力,算全层重要性
        to_evict = evict_every - ((step + 1) % evict_every or evict_every)
        active_obs_window = policy.observation_window(policy_params, obs_window)
        want_observe = track and active_obs_window > 0 and to_evict < active_obs_window
        want_logits = want_observe and policy.needs_attn_history and use_attention_logits
        want_attn = want_observe and not want_logits
        position_ids = torch.tensor([[true_pos]], device=device)
        forward_valid_masks = _append_valid_token(cache_valid_masks)
        if forward_valid_masks is not None:
            _set_headwise_valid_masks(model, forward_valid_masks)
        decode_start = time.perf_counter()
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    use_cache=True, output_attentions=want_attn,
                    output_hidden_states=want_logits,
                    attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=position_ids,
                    cache_position=torch.tensor([n], device=device))
        decode_forward_sec += time.perf_counter() - decode_start
        cache = out.past_key_values
        cache_valid_masks = forward_valid_masks
        logits = out.logits[:, -1]
        slot_pos[n] = true_pos; slot_ids[n] = nxt; ent[n] = e; imp[n] = 0.0
        cum_imp[n] = 0.0; win_imp[n] = 0.0
        if use_sig:
            att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_logits:
            obs_start = time.perf_counter()
            rows = _attention_logit_rows(model, out, cache, position_ids, n, cache_valid_masks)
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
            before_summary = _cache_length_summary(cache, cache_valid_masks, n)
            key_rep = None if policy.needs_cache else (_key_reps(cache, device) if policy.needs_key_reps else None)
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1)
                con = att_max[:n] / (cum + 1e-9)
            if debug is not None:
                idx, dbg = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                          key_rep, slot_pos[:n], budget, recent,
                                          sink, backend, anchor_mode, anchor_k, None,
                                          cum=cum, con=con, sig=sig_t,
                                          policy_params=policy_params, cache=cache,
                                          attn_history=attn_history,
                                          model=model,
                                          return_debug=True)
            else:
                idx = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                     key_rep, slot_pos[:n], budget, recent,
                                     sink, backend, anchor_mode, anchor_k, None,
                                     cum=cum, con=con, sig=sig_t,
                                     policy_params=policy_params, cache=cache,
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
            k, cache_valid_masks = _compact_cache(cache, idx, return_valid_masks=True)
            after_summary = _cache_length_summary(cache, cache_valid_masks, k)
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
    _set_headwise_valid_masks(model, None)
    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if use_cuda_memory else None
    mean_compression = (
        sum(event["compression_ratio"] for event in evict_events) / len(evict_events)
        if evict_events else 1.0
    )
    final_length_summary = _cache_length_summary(cache, cache_valid_masks, n)
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
    cache_valid_masks = None
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
        forward_valid_masks = _append_valid_token(cache_valid_masks)
        if forward_valid_masks is not None:
            _set_headwise_valid_masks(model, forward_valid_masks)
        out = model(input_ids=full_ids[:, P + t:P + t + 1], past_key_values=cache, use_cache=True,
                    output_attentions=want_attn, output_hidden_states=want_logits,
                    attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=position_ids, cache_position=torch.tensor([n], device=device))
        cache = out.past_key_values; cache_valid_masks = forward_valid_masks; logits = out.logits[:, -1]
        slot_pos[n] = true_pos; ent[n] = _entropy(logits)   # 锚点判据用当前 logits 熵近似
        imp[n] = 0.0; cum_imp[n] = 0.0; win_imp[n] = 0.0
        att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_logits:
            rows = _attention_logit_rows(model, out, cache, position_ids, n, cache_valid_masks)
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
            k, cache_valid_masks = _compact_cache(cache, idx, return_valid_masks=True)
            slot_pos[:k] = slot_pos[idx_slots]; imp[:k] = imp[idx_slots]
            cum_imp[:k] = cum_imp[idx_slots]; win_imp[:k] = win_imp[idx_slots]; ent[:k] = ent[idx_slots]
            att_sum[:k] = att_sum[idx_slots]; att_cnt[:k] = att_cnt[idx_slots]; att_max[:k] = att_max[idx_slots]
            n = k
            if policy.needs_attn_history:
                attn_history = []
    _set_headwise_valid_masks(model, None)
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
