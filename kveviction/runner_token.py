"""Token-level KV eviction decoding loops (HuggingFace reference runner).

后端:
  snapkv —— 重要性(观察窗注意力 maxpool)top-B + recent window + sink。
  rkv    —— 论文版 R-KV: observation attention importance + per-head key redundancy。
  h2o    —— 累计注意力 top-B + recent window + sink。
  window —— 观察窗注意力 top-B + recent window + sink。
  random —— 随机中段 token + recent window + sink。
锚点保护(anchor=高熵 forking token):在预算外额外钉 top-k 高熵 token;
  对照 mode:none / anchor / random / lowent。

辅助模块: ``sampling``(采样/熵)、``attention_logits``(R-KV logit 重算)、
``selection``(淘汰调度与选槽)、``eviction_stats``(cache 统计与指标)、
``trace_scoring``(teacher-forced NLL 诊断)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from kveviction.attention_logits import (
    attention_logit_rows,
    attention_logit_rows_batch,
    supports_attention_logits,
)
from kveviction.backends import ExecutionBackend, TransformersExecutionBackend
from kveviction.cache_manager import CacheManager, TransformersCacheManager
from kveviction.eviction_stats import (
    batch_cache_length_summaries,
    cache_length_summary,
    eviction_event,
    eviction_metrics,
    take_float,
    take_int,
    token_samples,
)
from kveviction.observations import (
    AttentionLogitObservationProvider,
    AttentionObservationProvider,
)
from kveviction.results import TimingSink
from kveviction.sampling import entropy, entropy_batch, sample, sample_batch
from kveviction.selection import (
    eviction_due,
    key_reps,
    representative_indices,
    representative_indices_batch,
    select_tokens,
    select_tokens_batch,
    selection_length,
    should_collect_observation,
)
from kveviction.strategies.token import get_policy

_cache_view = TransformersCacheManager._view


@torch.no_grad()
def generate_token_evict(
    model, tokenizer, input_ids, budget=1024, recent=64, sink=8,
    backend="rkv", anchor_mode="none", anchor_frac=0.05, evict_every=128,
    obs_window=16, obs_decay=0.9, max_new=4096,
    do_sample=True, temperature=0.6, top_p=0.95, seed=0, sig=None,
    policy_params=None,
    debug_path=None, debug_topk=0,
    execution_backend: ExecutionBackend | None = None,
    cache_manager: CacheManager | None = None,
    timing_sink: TimingSink | None = None,
):
    """token 级淘汰生成(忠实 R-KV 架构 + 支持长生成)。

    速度/忠实兼顾:平时不取观测信号; cache-aware 官方 baseline 只在压缩前
    observation window 内用 hidden states + q_proj + cache keys 重算 attention logits。
    缓冲预分配(slot_pos/imp/ent),消掉逐步 torch.cat 的 O(n²),解锁 16k 长生成。
    返回 dict(text, gen_ids, n_evict, final_cache_len)。"""
    device = next(model.parameters()).device
    execution_backend = execution_backend or TransformersExecutionBackend(model)
    cache_manager = cache_manager or TransformersCacheManager()
    timing_sink = timing_sink or TimingSink()
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
    observation_provider = AttentionObservationProvider(
        decay=obs_decay, needs_history=policy.needs_attn_history
    )
    logit_provider = AttentionLogitObservationProvider(attention_logit_rows)
    track = budget < 10 ** 8 and not policy.never_evict
    policy_params = policy_params or {}
    use_attention_logits = policy.needs_attn_history and supports_attention_logits(model)

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
    out = execution_backend.prefill(input_ids, use_cache=True, output_attentions=False)
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
        e = entropy(logits)
        nxt = sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            break
        # 只在下一次会真正触发压缩前的 observation window 内打开注意力观测。
        want_observe, active_obs_window = should_collect_observation(
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
        out = execution_backend.decode(
            torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
            output_hidden_states=want_logits,
            attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
            position_ids=position_ids,
            cache_position=torch.tensor([n], device=device),
        )
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
            rows = logit_provider.observe(
                out, cache=_cache_view(cache), n=n,
                position_ids=position_ids, model=model,
            )["history"]
            attention_observation_sec += time.perf_counter() - obs_start
            if rows is None:
                raise RuntimeError(
                    "attention-logit extraction failed; use a supported RoPE decoder or disable cache-aware baselines"
                )
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:                                 # 观察窗:全层注意力均值 maxpool 进 imp
            obs_start = time.perf_counter()
            observed = observation_provider.observe(
                out, cache=_cache_view(cache), n=n,
                position_ids=position_ids, model=model,
            )
            if policy.needs_attn_history and observed["history"] is not None:
                attn_history.append(observed["history"])
                attn_history = attn_history[-active_obs_window:]
            row = observed["importance"]
            imp[:n] = torch.maximum(imp[:n] * obs_decay, row)
            cum_imp[:n] += row
            win_imp[:n] = win_imp[:n] * obs_decay + row
            if use_sig:                               # cum_attn/concentration 统计
                att_sum[:n] += row; att_cnt[:n] += 1.0
                att_max[:n] = torch.maximum(att_max[:n], row)
            attention_observation_sec += time.perf_counter() - obs_start

        if eviction_due(policy, track=track, step=step, evict_every=evict_every,
                        cache_len=n, budget=budget, anchor_k=anchor_k):
            evict_start = time.perf_counter()
            selection_cache = _cache_view(cache)
            before_summary = cache_length_summary(selection_cache, n)
            key_rep = None if policy.needs_cache else (key_reps(selection_cache, device) if policy.needs_key_reps else None)
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1)
                con = att_max[:n] / (cum + 1e-9)
            selected = select_tokens(imp[:n], cum_imp[:n], win_imp[:n], ent[:n],
                                     key_rep, slot_pos[:n], budget, recent,
                                     sink, backend, anchor_mode, anchor_k, None,
                                     cum=cum, con=con, sig=sig_t,
                                     policy_params=policy_params, cache=selection_cache,
                                     attn_history=attn_history, model=model,
                                     return_debug=debug is not None)
            idx, dbg = selected if debug is not None else (selected, None)
            if debug is not None:
                idx_debug = representative_indices(idx)
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
                    "cache_len_after": selection_length(idx),
                    "slot_pos": take_int(slot_pos[:n]),
                    "token_ids": take_int(slot_ids[:n]),
                    "imp": take_float(imp[:n]),
                    "cum_imp": take_float(cum_imp[:n]),
                    "win_imp": take_float(win_imp[:n]),
                    "ent": take_float(ent[:n]),
                    "cum_attn": take_float(cum) if cum is not None else None,
                    "concentration": take_float(con) if con is not None else None,
                    "sig_score": take_float(score),
                    "policy_score": take_float(dbg["policy_score"]),
                    "backend_keep_slots": take_int(dbg["backend_keep"]),
                    "anchor_extra_slots": take_int(rescued),
                    "final_keep_slots": take_int(idx_debug),
                    "evicted_slots": take_int(evicted),
                    "anchor_extra_text": token_samples(tokenizer, slot_ids[:n], rescued, debug_topk),
                    "evicted_top_sig_text": token_samples(tokenizer, slot_ids[:n], evicted_by_sig, debug_topk),
                })
            before_n = n
            idx_slots = representative_indices(idx)
            cache, k = cache_manager.compact(cache, idx)
            after_summary = cache_length_summary(cache, k)
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
            evict_events.append(eviction_event(
                step=step + 1, before_n=before_n, after_n=k,
                before=before_summary, after=after_summary,
                eviction_sec=event_eviction_sec,
            ))

    text = tokenizer.decode(gen, skip_special_tokens=True)
    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if use_cuda_memory else None
    final_length_summary = cache_length_summary(cache, n)
    event_metrics = eviction_metrics(evict_events)
    mean_compression = event_metrics.pop("mean_compression_ratio")
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
    for name, value in timing.items():
        timing_sink.add(name, value)
    extra_metrics = {
        **final_length_summary,
        **event_metrics,
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
            "timing": dict(timing_sink.values),
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
    execution_backend: ExecutionBackend | None = None,
    cache_manager: CacheManager | None = None,
    timing_sink: TimingSink | None = None,
):
    """Generate independent candidates for one or more prompts in a static batch.

    Heterogeneous prompts use left padding plus a per-request valid-cache mask.
    Cache-aware policies may retain different slots for every request, layer,
    and KV head. Finished requests append masked filler slots while the remaining
    requests continue decoding.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    execution_backend = execution_backend or TransformersExecutionBackend(model)
    cache_manager = cache_manager or TransformersCacheManager()
    timing_sink = timing_sink or TimingSink()
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
                execution_backend=execution_backend,
                cache_manager=cache_manager,
                timing_sink=timing_sink,
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
    observation_provider = AttentionObservationProvider(
        decay=obs_decay, needs_history=policy.needs_attn_history
    )
    logit_provider = AttentionLogitObservationProvider(attention_logit_rows_batch)
    policy_params = policy_params or {}
    track = budget < 10 ** 8 and not policy.never_evict
    use_attention_logits = policy.needs_attn_history and supports_attention_logits(model)
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
    out = execution_backend.prefill(
        batched_input_ids,
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
    evict_events = [[] for _ in range(batch_size)]
    decode_forward_sec = 0.0
    attention_observation_sec = 0.0
    eviction_sec_total = 0.0

    for step in range(max_new):
        active = ~finished
        step_valid = active.clone()
        token_entropy = entropy_batch(logits)
        next_tokens = sample_batch(logits, do_sample, temperature, top_p, generators)
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, filler))
        for row in active.nonzero(as_tuple=True)[0].tolist():
            generated[row].append(int(next_tokens[row]))
        if eos is not None:
            finished |= active & next_tokens.eq(eos)
        if bool(finished.all()):
            break

        want_observe, active_obs_window = should_collect_observation(
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
        out = execution_backend.decode(
            next_tokens.unsqueeze(1),
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
            rows = logit_provider.observe(
                out, cache=_cache_view(cache), n=n,
                position_ids=position_ids, model=model,
            )["history"]
            attention_observation_sec += time.perf_counter() - observation_start
            if rows is None:
                raise RuntimeError("batched attention-logit extraction failed")
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:
            observation_start = time.perf_counter()
            observed = observation_provider.observe(
                out, cache=_cache_view(cache), n=n,
                position_ids=position_ids, model=model,
            )
            if policy.needs_attn_history and observed["history"] is not None:
                attn_history.append(observed["history"])
                attn_history = attn_history[-active_obs_window:]
            # stack().mean() keeps the historical reduction order (bitwise reproducible).
            row_score = torch.stack([
                attention[:, :, -1, :n].mean(1).float()
                for attention in out.attentions
            ]).mean(0)
            imp[:, :n] = torch.maximum(imp[:, :n] * obs_decay, row_score)
            cum_imp[:, :n] += row_score
            win_imp[:, :n] = win_imp[:, :n] * obs_decay + row_score
            attention_observation_sec += time.perf_counter() - observation_start

        if eviction_due(policy, track=track, step=step, evict_every=evict_every,
                        cache_len=n, budget=budget, anchor_k=anchor_k):
            eviction_start = time.perf_counter()
            before_n = n
            before_summaries = batch_cache_length_summaries(
                cache, valid_cache_mask[:, :before_n]
            )
            idx = select_tokens_batch(
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
            representative = representative_indices_batch(idx)
            cache, n = cache_manager.compact_batch(cache, idx)
            slot_pos[:, :n] = torch.gather(slot_pos[:, :before_n], 1, representative)
            slot_ids[:, :n] = torch.gather(slot_ids[:, :before_n], 1, representative)
            imp[:, :n] = torch.gather(imp[:, :before_n], 1, representative)
            cum_imp[:, :n] = torch.gather(cum_imp[:, :before_n], 1, representative)
            win_imp[:, :n] = torch.gather(win_imp[:, :before_n], 1, representative)
            ent[:, :n] = torch.gather(ent[:, :before_n], 1, representative)
            valid_cache_mask[:, :n] = torch.gather(
                valid_cache_mask[:, :before_n], 1, representative
            )
            after_summaries = batch_cache_length_summaries(
                cache, valid_cache_mask[:, :n]
            )
            if policy.needs_attn_history:
                attn_history = []
            n_evict += 1
            event_sec = time.perf_counter() - eviction_start
            eviction_sec_total += event_sec
            for row in range(batch_size):
                evict_events[row].append(eviction_event(
                    step=step + 1, before_n=before_n, after_n=n,
                    before=before_summaries[row], after=after_summaries[row],
                    eviction_sec=event_sec,
                ))

    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    timing_sink.add("prefill_sec", prefill_sec)
    timing_sink.add("decode_sec", max(0.0, elapsed - prefill_sec))
    timing_sink.add("decode_forward_sec", decode_forward_sec)
    timing_sink.add("attention_observation_sec", attention_observation_sec)
    timing_sink.add("eviction_sec_total", eviction_sec_total)
    batch_total_tokens = sum(len(ids) for ids in generated)
    amortized_elapsed = elapsed / batch_size
    final_summaries = batch_cache_length_summaries(cache, valid_cache_mask[:, :n])
    candidates = []
    for row, ids in enumerate(generated):
        row_events = evict_events[row]
        row_metrics = eviction_metrics(row_events)
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
            "mean_compression_ratio": row_metrics.pop("mean_compression_ratio"),
            "evict_events": row_events,
            **final_summaries[row],
            **row_metrics,
            "prefill_sec": prefill_sec / batch_size,
            "decode_sec": max(0.0, elapsed - prefill_sec) / batch_size,
            "decode_forward_sec": decode_forward_sec / batch_size,
            "attention_observation_sec": attention_observation_sec / batch_size,
            "eviction_sec_total": eviction_sec_total / batch_size,
            "eviction_sec_mean": (
                eviction_sec_total / (batch_size * n_evict) if n_evict else 0.0
            ),
            "other_decode_sec": max(
                0.0,
                elapsed - prefill_sec - decode_forward_sec
                - attention_observation_sec - eviction_sec_total,
            ) / batch_size,
            "timing": dict(timing_sink.values),
        })
    return {
        "candidates": candidates,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
        "batch_tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
        "timing": dict(timing_sink.values),
        "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
    }
