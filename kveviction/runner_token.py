"""Token-level KV eviction decoding loop (HuggingFace reference runner).

后端:
  snapkv —— 重要性(观察窗注意力 maxpool)top-B + recent window + sink。
  rkv    —— 论文版 R-KV: observation attention importance + per-head key redundancy。
  h2o    —— 累计注意力 top-B + recent window + sink。
  window —— 观察窗注意力 top-B + recent window + sink。
  random —— 随机中段 token + recent window + sink。
锚点保护(anchor=高熵 forking token):在预算外额外钉 top-k 高熵 token;
  对照 mode:none / anchor / random / lowent / sig。

``generate_token_evict`` (one request) and ``generate_token_evict_batch``
(static batch) are thin wrappers over a single decoding loop, ``_decode``.
A single unpadded request ("solo" mode) keeps the dense per-request semantics:
head-wise cache-aware selection on ``[H, N]`` rows, an optional per-eviction
debug dump, and an unmasked prefill.  Batches select per request over valid
slots and pad finished/left-padded rows with masked filler.  Every request
draws sampling and policy randomness from its own seeded generator, so a batch
row seeded with ``s`` reproduces a solo run seeded with ``s``.

辅助模块: ``sampling``(采样/熵)、``attention_logits``(R-KV logit 重算)、
``selection``(淘汰调度与选槽)、``slot_features``(每槽特征缓冲)、
``eviction_stats``(cache 统计与指标)、``trace_scoring``(teacher-forced NLL 诊断)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from kveviction.attention_logits import (
    attention_logit_rows,
    attention_logit_rows_batch,
    require_eager_attention,
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
from kveviction.sampling import RowUniforms, entropy_batch, sample_batch
from kveviction.selection import (
    eviction_due,
    representative_indices,
    representative_indices_batch,
    select_tokens,
    select_tokens_batch,
    selection_length,
    should_collect_observation,
)
from kveviction.slot_features import SlotFeatures
from kveviction.strategies.anchors import with_anchors
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
    sync_timing: bool = False,
    cache_impl: str = "growable",
):
    """token 级淘汰生成(忠实 R-KV 架构 + 支持长生成)。

    速度/忠实兼顾:平时不取观测信号; cache-aware 官方 baseline 只在压缩前
    observation window 内用 hidden states + q_proj + cache keys 重算 attention logits。
    每槽特征缓冲(SlotFeatures)预分配,消掉逐步 torch.cat 的 O(n²),解锁 16k 长生成。
    返回 dict(text, gen_ids, n_evict, final_cache_len, ...)。"""
    if do_sample or backend == "random":
        # Kept for callers relying on the global RNG state after a call; the
        # loop itself draws from a generator seeded identically.
        torch.manual_seed(seed)
    run = _decode(
        model, tokenizer, input_ids, batch_size=1, solo=True,
        budget=budget, recent=recent, sink=sink, backend=backend,
        anchor_mode=anchor_mode, anchor_frac=anchor_frac, evict_every=evict_every,
        obs_window=obs_window, obs_decay=obs_decay, max_new=max_new,
        do_sample=do_sample, temperature=temperature, top_p=top_p, seeds=[seed],
        sig=sig, policy_params=policy_params, input_attention_mask=None,
        debug_path=debug_path, debug_topk=debug_topk,
        execution_backend=execution_backend, cache_manager=cache_manager,
        sync_timing=sync_timing, cache_impl=cache_impl,
    )
    gen = run["generated"][0]
    events = run["evict_events"][0]
    elapsed = run["elapsed"]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    metrics = eviction_metrics(events)
    mean_compression = metrics.pop("mean_compression_ratio")
    n_evict = run["n_evict"]
    decode_sec = elapsed - run["prefill_sec"]
    timing = {
        "prefill_sec": run["prefill_sec"],
        "decode_sec": decode_sec,
        "decode_forward_sec": run["decode_forward_sec"],
        "attention_observation_sec": run["attention_observation_sec"],
        "eviction_sec_total": run["eviction_sec_total"],
        "eviction_sec_mean": run["eviction_sec_total"] / n_evict if n_evict else 0.0,
        "other_decode_sec": max(
            0.0,
            decode_sec - run["decode_forward_sec"]
            - run["attention_observation_sec"] - run["eviction_sec_total"],
        ),
    }
    extra_metrics = {**cache_length_summary(run["cache"], run["n"]), **metrics, **timing}
    peak_memory = run["peak_memory"]
    tokens_per_sec = len(gen) / elapsed if elapsed > 0 else 0.0
    debug = run["debug"]
    if debug is not None:
        debug.update({
            "generated_text": text,
            "gen_ids": [int(x) for x in gen],
            "n_evict": n_evict,
            "final_cache_len": run["n"],
            "evict_events": events,
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
            "mean_compression_ratio": mean_compression,
            **extra_metrics,
        })
        path = Path(debug_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(debug, ensure_ascii=False, indent=1))
    return {"text": text, "gen_ids": gen,
            "finished": run["finished"][0],
            "seed": int(seed),
            "prompt_len": int(run["prompt_lengths"][0]),
            "n_evict": n_evict, "final_cache_len": run["n"],
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "batch_elapsed_sec": elapsed,
            "batch_tokens_per_sec": tokens_per_sec,
            "batch_size": 1,
            "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
            "mean_compression_ratio": mean_compression,
            "evict_events": events,
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
    sig=None,
    sync_timing: bool = False,
    cache_impl: str = "growable",
):
    """Generate independent candidates for one or more prompts in a static batch.

    Heterogeneous prompts use left padding plus a per-request valid-cache mask.
    Cache-aware policies may retain different slots for every request, layer,
    and KV head. Finished requests append masked filler slots while the remaining
    requests continue decoding.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    common = dict(
        budget=budget, recent=recent, sink=sink, backend=backend,
        anchor_mode=anchor_mode, anchor_frac=anchor_frac, evict_every=evict_every,
        obs_window=obs_window, obs_decay=obs_decay, max_new=max_new,
        do_sample=do_sample, temperature=temperature, top_p=top_p, sig=sig,
        policy_params=policy_params, execution_backend=execution_backend,
        cache_manager=cache_manager, sync_timing=sync_timing, cache_impl=cache_impl,
    )
    if batch_size == 1 and input_ids.shape[0] == 1:
        # A single row runs on the solo path; drop its left padding first (the
        # batched selectors treat one row as unbatched and mis-shape the mask).
        if input_attention_mask is not None:
            keep = input_attention_mask[0].to(input_ids.device).bool()
            input_ids = input_ids[:, keep]
        seed = int((seeds or [0])[0])
        return {"candidates": [generate_token_evict(
            model, tokenizer, input_ids, seed=seed, **common,
        )]}

    if input_ids.shape[0] not in {1, batch_size}:
        raise ValueError(
            f"input batch must contain 1 or {batch_size} prompts, got {input_ids.shape[0]}"
        )
    seeds = list(seeds if seeds is not None else range(batch_size))
    if len(seeds) != batch_size:
        raise ValueError(f"expected {batch_size} seeds, got {len(seeds)}")
    run = _decode(
        model, tokenizer, input_ids, batch_size=batch_size, solo=False, seeds=seeds,
        input_attention_mask=input_attention_mask, debug_path=None, debug_topk=0,
        **common,
    )

    elapsed, prefill_sec, n_evict = run["elapsed"], run["prefill_sec"], run["n_evict"]
    forward_sec = run["decode_forward_sec"]
    observe_sec = run["attention_observation_sec"]
    evict_sec = run["eviction_sec_total"]
    peak_memory = run["peak_memory"]
    generated = run["generated"]
    batch_total_tokens = sum(len(ids) for ids in generated)
    amortized_elapsed = elapsed / batch_size
    final_summaries = batch_cache_length_summaries(run["cache"], run["feats"].valid[:, :run["n"]])
    candidates = []
    for row, ids in enumerate(generated):
        row_events = run["evict_events"][row]
        row_metrics = eviction_metrics(row_events)
        candidates.append({
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "gen_ids": ids,
            "finished": run["finished"][row],
            "seed": int(seeds[row]),
            "prompt_len": int(run["prompt_lengths"][row]),
            "n_evict": len(row_events),
            "final_cache_len": final_summaries[row]["max_effective_cache_len"],
            "elapsed_sec": amortized_elapsed,
            "tokens_per_sec": len(ids) / amortized_elapsed if amortized_elapsed > 0 else 0.0,
            "batch_elapsed_sec": elapsed,
            "batch_tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
            "batch_size": batch_size,
            "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
            "mean_compression_ratio": row_metrics.pop("mean_compression_ratio"),
            "evict_events": row_events,
            **final_summaries[row],
            **row_metrics,
            "prefill_sec": prefill_sec / batch_size,
            "decode_sec": max(0.0, elapsed - prefill_sec) / batch_size,
            "decode_forward_sec": forward_sec / batch_size,
            "attention_observation_sec": observe_sec / batch_size,
            "eviction_sec_total": evict_sec / batch_size,
            "eviction_sec_mean": (
                evict_sec / (batch_size * n_evict) if n_evict else 0.0
            ),
            "other_decode_sec": max(
                0.0, elapsed - prefill_sec - forward_sec - observe_sec - evict_sec,
            ) / batch_size,
        })
    return {
        "candidates": candidates,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
        "batch_tokens_per_sec": batch_total_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_memory_bytes": int(peak_memory) if peak_memory is not None else None,
    }


CACHE_IMPLS = ("growable", "dynamic")


def _new_cache(model, cache_impl: str) -> dict:
    """Prefill kwargs selecting the KV cache implementation.

    ``growable`` (default) uses ``GrowableCache``: amortized growth and
    in-place compaction instead of ``DynamicCache``'s per-step ``torch.cat``.
    Generations are identical; only speed and peak memory differ.  Models
    that are not HF ``PreTrainedModel``s (test doubles) keep their own cache.
    """
    if cache_impl not in CACHE_IMPLS:
        raise ValueError(f"cache_impl must be one of {CACHE_IMPLS}, got {cache_impl!r}")
    if cache_impl == "dynamic":
        return {}
    from transformers import PreTrainedModel

    if not isinstance(model, PreTrainedModel):
        return {}
    from kveviction.growable_cache import GrowableCache

    return {"past_key_values": GrowableCache()}


def _synced_clock(device):
    """``perf_counter`` after waiting for queued GPU work, so phase timings are real.

    CUDA kernels run asynchronously: without a sync a timer around a forward
    pass only measures the launch, and the GPU time lands in whichever phase
    next blocks.  Syncing makes every phase timing exact at a small throughput
    cost, so it is opt-in (``sync_timing``).
    """
    if device.type != "cuda":
        return time.perf_counter

    def clock():
        torch.cuda.synchronize(device)
        return time.perf_counter()

    return clock


def _accepts_logits_to_keep(model) -> bool:
    import inspect

    params = inspect.signature(model.forward).parameters
    return "logits_to_keep" in params


class _PhaseTimer:
    """Per-phase durations for the decode loop (prefill, forward, observation, eviction).

    With ``sync`` every boundary waits for the GPU (exact, slower).  Otherwise on
    CUDA each phase is bracketed by CUDA events, which do not block: their GPU
    timestamps are read once in ``resolve`` at the end.  A host timer around an
    async forward would only measure the kernel launch and push the GPU time
    into whichever later phase blocks.  On CPU plain ``perf_counter`` is exact.
    """

    def __init__(self, device, sync: bool):
        self.use_events = device.type == "cuda" and not sync
        self.clock = _synced_clock(device) if sync else time.perf_counter
        self.totals = {"prefill": 0.0, "forward": 0.0, "observe": 0.0, "evict": 0.0}
        self._pending = []

    def start(self):
        if self.use_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return self.clock()

    def stop(self, phase: str, start, records: list[dict] | None = None) -> None:
        """Add the time since ``start`` to ``phase``; also set ``eviction_sec`` on ``records``."""
        if self.use_events:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((phase, start, end, records))
            return
        self._add(phase, self.clock() - start, records)

    def _add(self, phase, seconds, records):
        self.totals[phase] += seconds
        for record in records or ():
            record["eviction_sec"] = seconds

    def resolve(self) -> dict:
        if self._pending:
            self._pending[-1][2].synchronize()
            for phase, start, end, records in self._pending:
                self._add(phase, start.elapsed_time(end) / 1000.0, records)
            self._pending = []
        return self.totals


def _decode(
    model, tokenizer, input_ids, *, batch_size, solo, budget, recent, sink, backend,
    anchor_mode, anchor_frac, evict_every, obs_window, obs_decay, max_new,
    do_sample, temperature, top_p, seeds, sig, policy_params, input_attention_mask,
    debug_path, debug_topk, execution_backend, cache_manager,
    sync_timing=False, forced_ids=None, trace_hook=None, cache_impl="growable",
):
    """Shared decoding loop; returns raw per-request state for the wrappers to format."""
    device = next(model.parameters()).device
    execution_backend = execution_backend or TransformersExecutionBackend(model)
    cache_manager = cache_manager or TransformersCacheManager()
    input_ids = input_ids.to(device)
    B = batch_size
    clock = _synced_clock(device) if sync_timing else time.perf_counter
    timer = _PhaseTimer(device, sync_timing)

    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
    uniforms = RowUniforms(generators)

    anchor_k = int(anchor_frac * budget)
    use_sig = anchor_mode == "sig"                    # 完整 4 维签名锚点
    sig_t = (torch.tensor(sig[0], device=device), float(sig[1])) if use_sig else None
    policy = with_anchors(get_policy(backend), anchor_mode, anchor_k, sig_t)
    policy_params = policy_params or {}
    observation_provider = AttentionObservationProvider(
        decay=obs_decay, needs_history=policy.needs_attn_history
    )
    # Solo runs feed strategies [H, N] rows; batches feed [B, H, N].
    logit_provider = AttentionLogitObservationProvider(
        attention_logit_rows if solo else attention_logit_rows_batch
    )
    track = budget < 10 ** 8 and not policy.never_evict
    use_attention_logits = policy.needs_attn_history and supports_attention_logits(model)
    require_eager_attention(model, policy, use_attention_logits=use_attention_logits, track=track)

    P = int(input_ids.shape[1])
    batched_input_ids = input_ids.expand(B, -1)
    if input_attention_mask is None:
        prompt_attention_mask = torch.ones_like(batched_input_ids, dtype=torch.long)
    else:
        input_attention_mask = input_attention_mask.to(device=device, dtype=torch.long)
        if input_attention_mask.shape[0] == 1 and B > 1:
            input_attention_mask = input_attention_mask.expand(B, -1)
        if input_attention_mask.shape != batched_input_ids.shape:
            raise ValueError(
                "input_attention_mask must match the expanded input_ids shape, got "
                f"{tuple(input_attention_mask.shape)} vs {tuple(batched_input_ids.shape)}"
            )
        prompt_attention_mask = input_attention_mask
    prompt_lengths = prompt_attention_mask.sum(-1).long()
    prefill_position_ids = prompt_attention_mask.cumsum(-1) - 1
    prefill_position_ids.masked_fill_(prompt_attention_mask == 0, 0)

    # 每槽特征缓冲预分配到 P + max_new;sig 锚点额外维护被访问注意力的 sum/cnt/max。
    feats = SlotFeatures((B, P + max_new), device, track_sig=use_sig, track_valid=True)
    feats.slot_pos[:, :P] = prefill_position_ids
    feats.slot_ids[:, :P] = batched_input_ids
    feats.valid[:, :P] = prompt_attention_mask.bool()
    n = P                                             # 当前缓存长度

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
    if (forced_ids is not None or trace_hook is not None) and not solo:
        raise ValueError("teacher forcing is only supported for single unpadded requests")
    if debug is not None and not solo:
        raise ValueError("debug dumps are only supported for single unpadded requests")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = clock()
    prefill_start = timer.start()
    prefill_cache = _new_cache(model, cache_impl)
    # Only the last position's logits are used; without this HF materializes
    # [batch, prompt, vocab] logits, the largest per-row memory spike in prefill.
    last_logits = {"logits_to_keep": 1} if _accepts_logits_to_keep(model) else {}
    if solo:
        out = execution_backend.prefill(
            input_ids, use_cache=True, output_attentions=False, **prefill_cache, **last_logits
        )
    else:
        out = execution_backend.prefill(
            batched_input_ids,
            attention_mask=prompt_attention_mask,
            position_ids=prefill_position_ids,
            use_cache=True,
            output_attentions=False,
            **prefill_cache,
            **last_logits,
        )
    timer.stop("prefill", prefill_start)
    cache = out.past_key_values
    logits = out.logits[:, -1]
    eos = tokenizer.eos_token_id
    filler = eos if eos is not None else tokenizer.pad_token_id
    if filler is None:
        filler = 0
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated = [[] for _ in range(B)]
    evict_events = [[] for _ in range(B)]
    attn_history = []
    n_evict = 0

    if trace_hook is not None:
        trace_hook.prefill(_cache_view(cache), P)
    forced_nll = []
    need_entropy = policy.needs_entropy or debug is not None
    for step in range(max_new):
        active = ~finished
        step_valid = active.clone()
        # Slot entropy is only read by entropy-based anchors and debug dumps.
        token_entropy = entropy_batch(logits) if need_entropy else 0.0
        if forced_ids is None:
            next_tokens = sample_batch(logits, do_sample, temperature, top_p, uniforms)
        else:
            # Teacher forcing: feed the given trace and record its NLL instead of sampling.
            next_tokens = forced_ids[:, step]
            forced_nll.append(
                -torch.log_softmax(logits.float(), -1).gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            )
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, filler))
        if eos is not None and forced_ids is None:
            finished |= active & next_tokens.eq(eos)
        # One device->host copy per step for tokens, active rows and the stop check.
        host_tokens, host_active, host_finished = torch.stack(
            [next_tokens.long(), active.long(), finished.long()]
        ).tolist()
        for row, (token, is_active) in enumerate(zip(host_tokens, host_active)):
            if is_active:
                generated[row].append(token)
        if all(host_finished):
            break

        # 只在下一次会真正触发压缩前的 observation window 内打开注意力观测。
        want_observe, active_obs_window = should_collect_observation(
            policy, track=track, policy_params=policy_params, obs_window=obs_window,
            evict_every=evict_every, step=step, cache_len=n, budget=budget,
            anchor_k=anchor_k,
        )
        want_logits = want_observe and policy.needs_attn_history and use_attention_logits
        want_attn = want_observe and not want_logits
        position_ids = (prompt_lengths + step).unsqueeze(1)
        decode_start = timer.start()
        out = execution_backend.decode(
            next_tokens.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
            output_hidden_states=want_logits,
            attention_mask=torch.cat([feats.valid[:, :n], step_valid.unsqueeze(1)], dim=1).long(),
            position_ids=position_ids,
            cache_position=torch.tensor([n], device=device),
        )
        timer.stop("forward", decode_start)
        cache = out.past_key_values
        logits = out.logits[:, -1]
        feats.append(n, pos=position_ids.squeeze(1), token=next_tokens,
                     entropy=token_entropy, valid=step_valid)
        if trace_hook is not None:
            trace_hook.append(n, int(position_ids[0, 0]))
        n += 1

        if want_logits:
            obs_start = timer.start()
            rows = logit_provider.observe(
                out, cache=_cache_view(cache), n=n, position_ids=position_ids, model=model,
            )["history"]
            timer.stop("observe", obs_start)
            if rows is None:
                raise RuntimeError(
                    "attention-logit extraction failed; use a supported RoPE decoder "
                    "or disable cache-aware baselines"
                )
            attn_history.append(rows)
            attn_history = attn_history[-active_obs_window:]
        if want_attn:                                 # 观察窗:全层注意力均值进 imp/cum/win
            obs_start = timer.start()
            observed = observation_provider.observe(
                out, cache=_cache_view(cache), n=n, position_ids=position_ids, model=model,
            )
            if policy.needs_attn_history and observed["history"] is not None:
                attn_history.append(observed["history"])
                attn_history = attn_history[-active_obs_window:]
            feats.observe(n, observed["importance"], obs_decay)
            timer.stop("observe", obs_start)

        if not eviction_due(policy, track=track, step=step, evict_every=evict_every,
                            cache_len=n, budget=budget, anchor_k=anchor_k):
            continue
        evict_start = timer.start()
        before_n = n
        selection_cache = _cache_view(cache)
        if solo:
            before = [cache_length_summary(selection_cache, n)]
            ctx = feats.context(n, budget=budget, recent=recent, sink=sink,
                                params=policy_params, row=0, generator=generators[0])
            selected = select_tokens(policy, ctx, cache=selection_cache,
                                     attn_history=attn_history,
                                     return_debug=debug is not None)
            idx, dbg = selected if debug is not None else (selected, None)
            if debug is not None:
                debug["evictions"].append(
                    _debug_eviction(step, n, idx, dbg, ctx, feats, tokenizer, debug_topk)
                )
            cache, n = cache_manager.compact(cache, idx)
            feats.compact(representative_indices(idx).unsqueeze(0), before_n)
            if trace_hook is not None:
                trace_hook.compact(step, before_n, idx, n)
            after = [cache_length_summary(cache, n)]
        else:
            before = batch_cache_length_summaries(cache, feats.valid[:, :n])
            idx = select_tokens_batch(
                policy, feats, n, budget=budget, recent=recent, sink=sink,
                policy_params=policy_params, cache=selection_cache,
                attn_history=attn_history, generators=generators,
            )
            cache, n = cache_manager.compact_batch(cache, idx)
            feats.compact(representative_indices_batch(idx), before_n)
            after = batch_cache_length_summaries(cache, feats.valid[:, :n])
        if policy.needs_attn_history:
            attn_history = []
        n_evict += 1
        # Record the event only for rows still generating, each with its own
        # cache length: in a batch the shared slot count ``n`` includes left
        # padding and rows that already finished.
        records = []
        for row in range(B):
            if solo:
                row_before, row_after = before_n, n
            elif host_finished[row]:
                continue
            else:
                row_before = int(before[row]["max_effective_cache_len"])
                row_after = int(after[row]["max_effective_cache_len"])
            records.append(eviction_event(
                step=step + 1, before_n=row_before, after_n=row_after,
                before=before[row], after=after[row], eviction_sec=0.0,
            ))
            evict_events[row].append(records[-1])
        timer.stop("evict", evict_start, records)

    elapsed = clock() - start_time
    phase_sec = timer.resolve()
    return {
        "generated": generated,
        "forced_nll": torch.stack(forced_nll) if forced_nll else None,   # [steps, B]
        # True when the request ended with EOS, False when it hit max_new.
        "finished": [bool(x) for x in finished.tolist()],
        "evict_events": evict_events,
        "n_evict": n_evict,
        "n": n,
        "cache": cache,
        "feats": feats,
        "prompt_lengths": prompt_lengths,
        "prefill_sec": phase_sec["prefill"],
        "decode_forward_sec": phase_sec["forward"],
        "attention_observation_sec": phase_sec["observe"],
        "eviction_sec_total": phase_sec["evict"],
        "elapsed": elapsed,
        "peak_memory": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "debug": debug,
    }


def _debug_eviction(step, n, idx, dbg, ctx, feats, tokenizer, debug_topk) -> dict:
    """Per-eviction dump for solo runs (``debug_path``)."""
    device = ctx.slot_pos.device
    idx_debug = representative_indices(idx)
    final_keep = torch.zeros(n, dtype=torch.bool, device=device)
    final_keep[idx_debug] = True
    evicted = (~final_keep).nonzero(as_tuple=True)[0]
    score = dbg["sig_score"]
    if score is None:
        score = torch.full((n,), float("nan"), device=device)
    rescued = dbg["anchor_extra"]
    evicted_by_sig = (
        evicted[torch.argsort(score[evicted], descending=True)] if evicted.numel() else evicted
    )
    slot_ids = feats.slot_ids[0, :n]
    return {
        "step": step + 1,
        "cache_len_before": n,
        "cache_len_after": selection_length(idx),
        "slot_pos": take_int(ctx.slot_pos),
        "token_ids": take_int(slot_ids),
        "imp": take_float(ctx.importance),
        "cum_imp": take_float(ctx.cumulative_attention),
        "win_imp": take_float(ctx.window_attention),
        "ent": take_float(ctx.entropy),
        "cum_attn": take_float(ctx.cumulative_mean) if ctx.cumulative_mean is not None else None,
        "concentration": take_float(ctx.concentration) if ctx.concentration is not None else None,
        "sig_score": take_float(score),
        "policy_score": take_float(dbg["policy_score"]),
        "backend_keep_slots": take_int(dbg["backend_keep"]),
        "anchor_extra_slots": take_int(rescued),
        "final_keep_slots": take_int(idx_debug),
        "evicted_slots": take_int(evicted),
        "anchor_extra_text": token_samples(tokenizer, slot_ids, rescued, debug_topk),
        "evicted_top_sig_text": token_samples(tokenizer, slot_ids, evicted_by_sig, debug_topk),
    }
