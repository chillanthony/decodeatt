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

from rkv.policies import SelectionContext, get_policy


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
                   policy_params=None,
                   return_debug=False):
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
    ctx = SelectionContext(
        importance=imp,
        cumulative_attention=cum_imp,
        window_attention=win_imp,
        key_reps=key_rep,
        slot_pos=slot_pos,
        budget=budget,
        recent=recent,
        sink=sink,
        params=policy_params or {},
    )
    backend_idx = policy.select_keep(ctx)
    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    keep[backend_idx] = True
    backend_keep = keep.clone()
    anchor_extra = torch.empty(0, dtype=torch.long, device=dev)
    sig_score = None
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


def _aggregate_gqa_attention(attn: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Map query-head attention rows to KV heads.

    HF returns attention probabilities per query head. For GQA models, several
    query heads share one KV head; we approximate the paper's group max-pooling
    using the returned attention probabilities.
    """
    q_heads = attn.shape[0]
    if q_heads == kv_heads:
        return attn
    if q_heads % kv_heads == 0:
        group = q_heads // kv_heads
        return attn.view(kv_heads, group, attn.shape[-1]).max(1).values
    return attn.mean(0, keepdim=True).expand(kv_heads, attn.shape[-1])


def _max_pool_importance(attn: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1 or attn.numel() == 0:
        return attn
    pad = kernel // 2
    pooled = torch.nn.functional.max_pool1d(
        attn.unsqueeze(1), kernel_size=kernel, stride=1, padding=pad
    ).squeeze(1)
    return pooled[:, : attn.shape[-1]]


def _rkv_paper_importance(
    attn_history: list[list[torch.Tensor]],
    layer_idx: int,
    kv_heads: int,
    n_cand: int,
    n_total: int,
    pool_kernel: int,
    device,
) -> torch.Tensor:
    rows = []
    for step_rows in attn_history:
        if layer_idx >= len(step_rows):
            continue
        row = step_rows[layer_idx].to(device=device, dtype=torch.float32)
        row = _aggregate_gqa_attention(row, kv_heads)
        if row.shape[-1] < n_total:
            row = torch.nn.functional.pad(row, (0, n_total - row.shape[-1]))
        rows.append(row[:, :n_cand])
    if not rows:
        return torch.zeros(kv_heads, n_cand, dtype=torch.float32, device=device)
    attn = torch.stack(rows, dim=1).mean(1)
    return _max_pool_importance(attn, pool_kernel)


def _rkv_paper_redundancy(
    keys: torch.Tensor,
    threshold: float,
    beta: int,
    eps: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute paper Eq. (6) per KV head without materializing all heads at once."""
    kv_heads, n_cand, _ = keys.shape
    if n_cand == 0:
        return torch.empty(kv_heads, 0, dtype=torch.float32, device=keys.device)
    keys = torch.nn.functional.normalize(keys.float(), dim=-1, eps=eps)
    means = []
    col_ids = torch.arange(n_cand, device=keys.device).view(1, 1, n_cand)
    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        sim = keys[:, start:end] @ keys.transpose(1, 2)
        local = torch.arange(start, end, device=keys.device)
        sim[:, torch.arange(end - start, device=keys.device), local] = 0.0
        if beta > 0:
            high = sim > threshold
            recent_ids = torch.where(high, col_ids, torch.full_like(col_ids, -1))
            recent = recent_ids.topk(min(beta, n_cand), dim=-1).values
            for rank in range(recent.shape[-1]):
                target = recent[..., rank]
                valid = target >= 0
                if valid.any():
                    head_idx, row_idx = torch.where(valid)
                    sim[head_idx, row_idx, target[valid]] = 0.0
        means.append(sim.mean(-1))
    avg_sim = torch.cat(means, dim=-1)
    return torch.softmax(avg_sim, dim=-1)


def _select_rkv_paper(cache, attn_history, n, budget, params, return_debug=False):
    """Paper-faithful R-KV selection.

    ``budget`` is the paper's Bbudget. The final cache keeps the selected
    Bbudget candidate tokens plus the last alpha observation tokens.
    """
    device = cache.layers[0].keys.device
    alpha = int(params.get("alpha", 8))
    lam = float(params.get("lambda", params.get("redundancy_lambda", 0.1)))
    pool_kernel = int(params.get("pool_kernel", 5))
    threshold = float(params.get("similarity_threshold", 0.9))
    beta = int(params.get("beta", 8))
    eps = float(params.get("eps", 1e-8))
    chunk_size = int(params.get("redundancy_chunk", 512))

    obs = min(alpha, n)
    n_cand = n - obs
    if n_cand <= budget:
        idx = torch.arange(n, device=device)
        if return_debug:
            score = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
            return idx, {"backend_keep": idx, "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
                         "sig_score": None, "policy_score": score}
        return idx

    agg_score = torch.zeros(n_cand, dtype=torch.float32, device=device)
    n_heads_total = 0
    for layer_idx, layer in enumerate(cache.layers):
        keys = layer.keys[0, :, :n_cand, :]
        kv_heads = keys.shape[0]
        importance = _rkv_paper_importance(
            attn_history, layer_idx, kv_heads, n_cand, n, pool_kernel, device
        )
        redundancy = _rkv_paper_redundancy(keys, threshold, beta, eps, chunk_size)
        score = lam * importance - (1.0 - lam) * redundancy
        agg_score += score.sum(0)
        n_heads_total += kv_heads
    agg_score /= max(n_heads_total, 1)

    selected = torch.argsort(agg_score, descending=True)[:budget]
    obs_idx = torch.arange(n_cand, n, device=device)
    idx = torch.cat([selected, obs_idx]).sort().values
    if return_debug:
        policy_score = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
        policy_score[:n_cand] = agg_score
        return idx, {
            "backend_keep": idx,
            "anchor_extra": torch.empty(0, dtype=torch.long, device=device),
            "sig_score": None,
            "policy_score": policy_score,
        }
    return idx


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

    速度/忠实兼顾:平时 output_attentions=False(模型若以 sdpa 加载则走快算子),
    **只在每次压缩前的 obs_window 步**用全层注意力算重要性(SnapKV/R-KV 的观察窗思想)。
    缓冲预分配(slot_pos/imp/ent),消掉逐步 torch.cat 的 O(n²),解锁 16k 长生成。
    返回 dict(text, gen_ids, n_evict, final_cache_len)。"""
    device = next(model.parameters()).device
    start_time = time.perf_counter()
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
    track = budget < 10 ** 8                          # full 臂超大预算→不淘汰,全程快路径
    policy_params = policy_params or {}
    paper_rkv = backend in {"rkv", "rkv-paper"}
    paper_alpha = int(policy_params.get("alpha", 8))

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

    out = model(input_ids=input_ids, use_cache=True, output_attentions=False)
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
        # 只在"下次压缩前 obs_window 步内"开注意力,算全层重要性
        to_evict = evict_every - ((step + 1) % evict_every or evict_every)
        active_obs_window = paper_alpha if paper_rkv else obs_window
        want_attn = track and to_evict < active_obs_window
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    use_cache=True, output_attentions=want_attn,
                    attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=torch.tensor([[true_pos]], device=device),
                    cache_position=torch.tensor([n], device=device))
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos[n] = true_pos; slot_ids[n] = nxt; ent[n] = e; imp[n] = 0.0
        cum_imp[n] = 0.0; win_imp[n] = 0.0
        if use_sig:
            att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_attn:                                 # 观察窗:全层注意力均值 maxpool 进 imp
            if paper_rkv:
                attn_history.append([a[0, :, -1, :n].detach().float() for a in out.attentions])
                attn_history = attn_history[-paper_alpha:]
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

        trigger_len = budget + (paper_alpha if paper_rkv else anchor_k)
        if track and (step + 1) % evict_every == 0 and n > trigger_len:
            policy = get_policy(backend)
            key_rep = None if paper_rkv else (_key_reps(cache, device) if policy.needs_key_reps else None)
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1)
                con = att_max[:n] / (cum + 1e-9)
            if paper_rkv:
                if debug is not None:
                    idx, dbg = _select_rkv_paper(
                        cache, attn_history, n, budget, policy_params, return_debug=True
                    )
                else:
                    idx = _select_rkv_paper(cache, attn_history, n, budget, policy_params)
            elif debug is not None:
                idx, dbg = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                          key_rep, slot_pos[:n], budget, recent,
                                          sink, backend, anchor_mode, anchor_k, None,
                                          cum=cum, con=con, sig=sig_t,
                                          policy_params=policy_params,
                                          return_debug=True)
            else:
                idx = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                     key_rep, slot_pos[:n], budget, recent,
                                     sink, backend, anchor_mode, anchor_k, None,
                                     cum=cum, con=con, sig=sig_t,
                                     policy_params=policy_params)
            if debug is not None:
                final_keep = torch.zeros(n, dtype=torch.bool, device=device)
                final_keep[idx] = True
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
                    "cache_len_after": int(idx.numel()),
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
                    "final_keep_slots": _take_int(idx),
                    "evicted_slots": _take_int(evicted),
                    "anchor_extra_text": _token_samples(tokenizer, slot_ids[:n], rescued, debug_topk),
                    "evicted_top_sig_text": _token_samples(tokenizer, slot_ids[:n], evicted_by_sig, debug_topk),
                })
            before_n = n
            for layer in cache.layers:
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
            k = idx.numel()
            slot_pos[:k] = slot_pos[idx]; slot_ids[:k] = slot_ids[idx]; imp[:k] = imp[idx]
            cum_imp[:k] = cum_imp[idx]; win_imp[:k] = win_imp[idx]; ent[:k] = ent[idx]
            if use_sig:
                att_sum[:k] = att_sum[idx]; att_cnt[:k] = att_cnt[idx]; att_max[:k] = att_max[idx]
            n = k
            if paper_rkv:
                attn_history = []
            n_evict += 1
            evict_events.append({
                "step": step + 1,
                "cache_len_before": int(before_n),
                "cache_len_after": int(k),
                "compression_ratio": float(k / max(before_n, 1)),
                "evicted": int(before_n - k),
            })

    text = tokenizer.decode(gen, skip_special_tokens=True)
    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if use_cuda_memory else None
    mean_compression = (
        sum(event["compression_ratio"] for event in evict_events) / len(evict_events)
        if evict_events else 1.0
    )
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
        path = Path(debug_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(debug, ensure_ascii=False, indent=1))
    return {"text": text, "gen_ids": gen,
            "n_evict": n_evict, "final_cache_len": n,
            "elapsed_sec": elapsed,
            "tokens_per_sec": len(gen) / elapsed if elapsed > 0 else 0.0,
            "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
            "mean_compression_ratio": mean_compression,
            "evict_events": evict_events}


@torch.no_grad()
def score_trace_token(model, tokenizer, full_ids, prompt_len, refl_steps, anchor_mode,
                      budget=1024, recent=64, sink=8, backend="rkv", anchor_frac=0.05,
                      evict_every=128, obs_window=16, obs_decay=0.9, span_len=32, sig=None):
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

    N = P + Lgen
    slot_pos = torch.zeros(N, dtype=torch.long, device=device); slot_pos[:P] = torch.arange(P, device=device)
    imp = torch.zeros(N, device=device); cum_imp = torch.zeros(N, device=device)
    win_imp = torch.zeros(N, device=device); ent = torch.zeros(N, device=device)
    att_sum = torch.zeros(N, device=device); att_cnt = torch.zeros(N, device=device); att_max = torch.zeros(N, device=device)
    n = P
    out = model(input_ids=full_ids[:, :P], use_cache=True, output_attentions=False)
    cache = out.past_key_values; logits = out.logits[:, -1]
    nll_c, nll_n, c_c, c_n = 0.0, 0.0, 0, 0

    for t in range(Lgen):
        real = int(gen_ids[t])
        nll = -float(torch.log_softmax(logits.float(), -1)[0, real])
        if corr[t]: nll_c += nll; c_c += 1
        else: nll_n += nll; c_n += 1
        true_pos = P + t
        to_evict = evict_every - ((t + 1) % evict_every or evict_every)
        want_attn = to_evict < obs_window
        out = model(input_ids=full_ids[:, P + t:P + t + 1], past_key_values=cache, use_cache=True,
                    output_attentions=want_attn, attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=torch.tensor([[true_pos]], device=device), cache_position=torch.tensor([n], device=device))
        cache = out.past_key_values; logits = out.logits[:, -1]
        slot_pos[n] = true_pos; ent[n] = _entropy(logits)   # 锚点判据用当前 logits 熵近似
        imp[n] = 0.0; cum_imp[n] = 0.0; win_imp[n] = 0.0
        att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_attn:
            row = torch.zeros(n, device=device)
            for a in out.attentions: row += a[0, :, -1, :].mean(0).float()
            row /= len(out.attentions)
            imp[:n] = torch.maximum(imp[:n] * obs_decay, row)
            cum_imp[:n] += row
            win_imp[:n] = win_imp[:n] * obs_decay + row
            if use_sig:
                att_sum[:n] += row; att_cnt[:n] += 1.0; att_max[:n] = torch.maximum(att_max[:n], row)
        if (t + 1) % evict_every == 0 and n > budget + anchor_k:
            policy = get_policy(backend)
            key_rep = _key_reps(cache, device) if policy.needs_key_reps else None
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1); con = att_max[:n] / (cum + 1e-9)
            idx = _select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                 key_rep, slot_pos[:n], budget, recent, sink,
                                 backend, anchor_mode, anchor_k, None, cum=cum, con=con, sig=sig_t)
            for layer in cache.layers:
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
            k = idx.numel()
            slot_pos[:k] = slot_pos[idx]; imp[:k] = imp[idx]
            cum_imp[:k] = cum_imp[idx]; win_imp[:k] = win_imp[idx]; ent[:k] = ent[idx]
            att_sum[:k] = att_sum[idx]; att_cnt[:k] = att_cnt[idx]; att_max[:k] = att_max[idx]
            n = k
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
