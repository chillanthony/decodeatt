"""Phase 2：自写解码循环 + 页级 KV 淘汰 + 在线签名豁免（RescueKV v0）。

每步：单 token 前向（output_attentions）→ 取当前 query 对各页注意力 → 更新在线签名 →
每 evict_every 步按后端策略淘汰页（始终保最近 protect_recent 页；按 arm 额外豁免
≤exempt_frac 页）。淘汰直接压缩 HF DynamicCache 的每层 keys/values，并维护
slot_pos（每个缓存槽的真实序列位置），保证 RoPE 相位与页映射正确。

五臂（arm）：
  full   ：不淘汰（上界基线）
  evict  ：纯淘汰（无豁免）
  rescue ：淘汰 + 豁免签名分最高的页（RescueKV v0）
  random ：淘汰 + 随机豁免同等页数（null）
  gsig   ：淘汰 + 豁免签名分最低的页（最像 G，反向 null）

诊断：receiver 命中率——reflection 触发时反查早期高注意力页，统计其是否仍存活。
"""
from __future__ import annotations

import random

import torch

from rkv.online_features import OnlineSignature
from rkv.utils.reflection import _REFLECTION_RE

ARMS = ["full", "evict", "rescue", "random", "gsig", "rescue_buf"]


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


def _attn_row_to_pages(attn_layers, slot_pos: torch.Tensor, num_pages: int,
                       page_size: int, device) -> torch.Tensor:
    """把各层 [B,H,1,kv] 的当前 query 注意力（层/头平均）按槽位映射聚合到页 [num_pages]。"""
    acc = None
    for a in attn_layers:
        row = a[0, :, -1, :].mean(0).float()        # [kv]
        acc = row if acc is None else acc + row
    acc = acc / len(attn_layers)
    assert acc.numel() == slot_pos.numel(), \
        f"attn kv_len {acc.numel()} != slot_pos {slot_pos.numel()}（淘汰后错配）"
    pages = torch.zeros(num_pages, device=device)
    slot_page = (slot_pos // page_size).to(device)
    pages.scatter_add_(0, slot_page, acc.to(device))
    return pages


@torch.no_grad()
def generate_with_evict(
    model, tokenizer, input_ids, scorer: OnlineSignature | None, arm: str,
    max_new: int = 4096, page_size: int = 16, keep_frac: float = 0.2,
    protect_recent: int = 4, exempt_frac: float = 0.05, evict_every: int = 32,
    window: int = 8, backend: str = "window", receiver_topk: int = 8,
    exclude_recent_pages: int = 4, seed: int = 0,
    do_sample: bool = False, temperature: float = 0.6, top_p: float = 0.95,
    buffer_cap: int = 48, recall_topk: int = 4,
):
    """返回 dict(text, gen_ids, n_evict_rounds, n_reflection, receiver_hits, receiver_total)。

    arm=rescue_buf：可恢复缓冲——淘汰高签名页时 offload 到 CPU（不丢），reflection 触发
    时召回 top-recall_topk 个 CPU 缓冲页回缓存。buffer_cap 限制 CPU 缓冲页数。
    do_sample/temperature/top_p：采样解码（避免贪心退化）；seed 固定保证臂间可复现。
    """
    assert arm in ARMS, arm
    device = next(model.parameters()).device
    rng = random.Random(seed)
    if do_sample:
        torch.manual_seed(seed)
    input_ids = input_ids.to(device)
    P = input_ids.shape[1]
    store: dict[int, dict] = {}                  # page_idx -> {"pos":[...], "kv":[(k,v)..]}（CPU）
    is_buf = arm == "rescue_buf"
    evict_arm = "evict" if is_buf else arm       # 缓冲臂的淘汰用纯 evict 选页

    out = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    cache = out.past_key_values
    logits = out.logits[:, -1]
    slot_pos = torch.arange(P, device=device)           # 每个缓存槽的真实位置

    eos = tokenizer.eos_token_id
    gen, recent_rows = [], []
    n_evict, n_refl, rcv_hit, rcv_tot = 0, 0, 0, 0
    refl_re = _REFLECTION_RE

    def ingest(out, slot_pos, gen_token_pos):
        """前向紧后、压缩之前调用：此刻 out.attentions 的 kv_len 与 slot_pos 长度严格一致。
        更新在线签名（当前 query 的页注意力行 + 即将生成 token 的熵）。"""
        if arm == "full" or scorer is None:
            return
        num_pages = int(slot_pos.max().item()) // page_size + 1
        pages_row = _attn_row_to_pages(out.attentions, slot_pos, num_pages,
                                       page_size, device)
        scorer.update(pages_row, _entropy(out.logits[:, -1]), token_pos=gen_token_pos)
        recent_rows.append((slot_pos.max().item() // page_size, pages_row))
        if len(recent_rows) > window:
            recent_rows.pop(0)

    # prefill 的 query 在 P-1，预测位置 P 的 token；此刻 slot_pos=arange(P) 与 kv_len 一致
    ingest(out, slot_pos, gen_token_pos=P)

    for step in range(max_new):
        true_pos = P + step
        nxt = _sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            break

        # 自写步进：position_ids = 真实位置（RoPE），attention_mask 全 1（可见全部存活槽）
        cur = torch.tensor([[nxt]], device=device)
        cache_len = slot_pos.numel()
        attn_mask = torch.ones(1, cache_len + 1, device=device, dtype=torch.long)
        pos_ids = torch.tensor([[true_pos]], device=device)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                    output_attentions=True, attention_mask=attn_mask,
                    position_ids=pos_ids, cache_position=torch.tensor([cache_len], device=device))
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos = torch.cat([slot_pos, torch.tensor([true_pos], device=device)])

        # 签名更新（此刻 slot_pos 与 out.attentions 长度一致；务必在淘汰压缩之前）
        ingest(out, slot_pos, gen_token_pos=true_pos + 1)

        # reflection 在线检测（看最近一小段文本）→ 早期注意力质量诊断 + 缓冲召回
        if (step + 1) % 16 == 0:
            tail = tokenizer.decode(gen[-48:])
            if refl_re.search(tail):
                n_refl += 1
                mass = _early_attn_mass(out.attentions, slot_pos, true_pos + 1,
                                        page_size, exclude_recent_pages)
                if mass is not None:
                    rcv_hit += mass        # 累加早期注意力质量占比
                    rcv_tot += 1
                if is_buf and store:       # 纠错触发：召回 top 签名缓冲页回缓存
                    num_pages = int(slot_pos.max().item()) // page_size + 1
                    sig = scorer.scores(num_pages)
                    cand = sorted(store, key=lambda p: float(sig[p]) if p < num_pages
                                  else -1e9, reverse=True)[:recall_topk]
                    slot_pos = _recall_pages(cache, slot_pos, store, cand, device)

        # 周期性淘汰（在 ingest 之后）
        if arm != "full" and (step + 1) % evict_every == 0:
            num_pages = (true_pos + 1) // page_size + 1
            keep_pages = _select_pages(
                scorer, recent_rows, slot_pos, num_pages, page_size, keep_frac,
                protect_recent, exempt_frac, evict_arm, backend, rng)
            if is_buf:                     # 把将被丢弃的高签名页 offload 到 CPU 缓冲
                sig = scorer.scores(num_pages)
                alive = set((slot_pos // page_size).tolist())
                dropped = [p for p in alive if p not in keep_pages and p not in store]
                dropped.sort(key=lambda p: float(sig[p]), reverse=True)
                for p in dropped[: max(0, buffer_cap - len(store))]:
                    _offload_page(cache, slot_pos, store, p, page_size)
            slot_pos = _compact(cache, slot_pos, keep_pages, page_size, device)
            n_evict += 1

    text = tokenizer.decode(gen, skip_special_tokens=True)
    return {"text": text, "gen_ids": gen, "n_evict_rounds": n_evict,
            "n_reflection": n_refl, "receiver_hits": rcv_hit,
            "receiver_total": rcv_tot, "final_cache_len": int(slot_pos.numel())}


def _entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return float(-(torch.where(p > 0, p * logp, torch.zeros_like(p))).sum())


@torch.no_grad()
def score_trace_teacherforced(
    model, tokenizer, full_ids, prompt_len, refl_steps, scorer, arm,
    span_len: int = 32, page_size: int = 16, keep_frac: float = 0.2,
    protect_recent: int = 24, exempt_frac: float = 0.05, evict_every: int = 32,
    window: int = 8, backend: str = "window", seed: int = 0,
):
    """Teacher-forced 因果探针:喂**真** token 走淘汰循环,测各 token 的 NLL。

    不自由生成 -> 无连贯性崩溃混淆;臂间唯一差异是缓存内容,故 NLL 差异纯由淘汰导致。
    correction span = 每个 reflection 步起 span_len 个 token(gen 坐标),模型在此使用回查内容。
    返回 dict(nll_all, nll_corr, nll_noncorr, n_corr, n_all)。
    """
    device = next(model.parameters()).device
    rng = random.Random(seed)
    full_ids = full_ids.to(device)
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    P = prompt_len
    gen_ids = full_ids[0, P:]
    Lgen = gen_ids.numel()
    corr_mask = torch.zeros(Lgen, dtype=torch.bool)
    for s in refl_steps:
        corr_mask[s: min(s + span_len, Lgen)] = True

    out = model(input_ids=full_ids[:, :P], use_cache=True, output_attentions=True)
    cache = out.past_key_values
    logits = out.logits[:, -1]
    slot_pos = torch.arange(P, device=device)
    recent_rows = []

    def ingest(out, slot_pos, gen_token_pos):
        if arm == "full" or scorer is None:
            return
        num_pages = int(slot_pos.max().item()) // page_size + 1
        pages_row = _attn_row_to_pages(out.attentions, slot_pos, num_pages, page_size, device)
        scorer.update(pages_row, _entropy(out.logits[:, -1]), token_pos=gen_token_pos)
        recent_rows.append((slot_pos.max().item() // page_size, pages_row))
        if len(recent_rows) > window:
            recent_rows.pop(0)

    ingest(out, slot_pos, gen_token_pos=P)
    nll_corr, nll_non, n_corr, n_non = 0.0, 0.0, 0, 0
    for t in range(Lgen):
        real = int(gen_ids[t])
        nll = -float(torch.log_softmax(logits.float(), -1)[0, real])
        if corr_mask[t]:
            nll_corr += nll; n_corr += 1
        else:
            nll_non += nll; n_non += 1
        true_pos = P + t
        cur = full_ids[:, P + t: P + t + 1]
        cache_len = slot_pos.numel()
        out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                    output_attentions=True,
                    attention_mask=torch.ones(1, cache_len + 1, device=device, dtype=torch.long),
                    position_ids=torch.tensor([[true_pos]], device=device),
                    cache_position=torch.tensor([cache_len], device=device))
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos = torch.cat([slot_pos, torch.tensor([true_pos], device=device)])
        ingest(out, slot_pos, gen_token_pos=true_pos + 1)
        if arm != "full" and (t + 1) % evict_every == 0:
            num_pages = (true_pos + 1) // page_size + 1
            keep_pages = _select_pages(scorer, recent_rows, slot_pos, num_pages,
                                       page_size, keep_frac, protect_recent,
                                       exempt_frac, arm, backend, rng)
            slot_pos = _compact(cache, slot_pos, keep_pages, page_size, device)
    return {
        "nll_corr": nll_corr / max(n_corr, 1), "n_corr": n_corr,
        "nll_noncorr": nll_non / max(n_non, 1), "n_noncorr": n_non,
        "nll_all": (nll_corr + nll_non) / max(n_corr + n_non, 1),
    }


def _early_attn_mass(attn_layers, slot_pos, q_true_pos, page_size, excl):
    """reflection 时刻：当前 token 注意力落在**早期存活页**上的质量占比。

    跨臂可比的诚实指标：淘汰若杀掉早期纠错锚点，模型回看时这些槽已不在缓存，
    本可投向它们的注意力质量随之塌缩。占比越高 = 模型越能成功回看早期内容。
    返回 float 占比；早期区不存在则 None（不计入）。"""
    acc = None
    for a in attn_layers:
        row = a[0, :, -1, :].mean(0).float()
        acc = row if acc is None else acc + row
    acc = acc / len(attn_layers)
    cur_page = (q_true_pos - 1) // page_size
    cutoff = cur_page - excl
    if cutoff <= 0:
        return None
    early = (slot_pos // page_size) < cutoff
    return float(acc[early].sum() / (acc.sum() + 1e-9))


def _select_pages(scorer, recent_rows, slot_pos, num_pages, page_size, keep_frac,
                  protect_recent, exempt_frac, arm, backend, rng) -> set[int]:
    """返回应保留的页集合。预算 = keep_frac×页数；额外豁免 exempt_frac×页数。"""
    alive_pages = sorted(set((slot_pos // page_size).tolist()))
    budget = max(1, int(num_pages * keep_frac))
    cur_page = (slot_pos.max().item()) // page_size
    protected = {p for p in alive_pages if p > cur_page - protect_recent}

    # 后端打分（越高越保留）
    if backend == "h2o":
        cum = scorer.cum_sum[:num_pages] / scorer.cum_cnt[:num_pages].clamp(min=1)
        score = cum
    else:  # window：各行页数随时间增长，右侧补零到 num_pages 再叠加
        score = torch.zeros(num_pages, device=slot_pos.device)
        for _, row in recent_rows:
            n = min(row.numel(), num_pages)
            score[:n] += row[:n].to(score.device)
    cand = [p for p in alive_pages if p not in protected]
    cand.sort(key=lambda p: float(score[p]), reverse=True)
    keep = set(protected) | set(cand[: max(0, budget - len(protected))])

    # 额外豁免名额（在被淘汰的页里挑）
    n_exempt = max(0, int(num_pages * exempt_frac))
    dropped = [p for p in alive_pages if p not in keep]
    if n_exempt and dropped and arm in ("rescue", "random", "gsig"):
        sig = scorer.scores(num_pages)
        if arm == "rescue":
            dropped.sort(key=lambda p: float(sig[p]), reverse=True)
        elif arm == "gsig":
            dropped.sort(key=lambda p: float(sig[p]))
        else:
            rng.shuffle(dropped)
        keep |= set(dropped[:n_exempt])
    return keep


def _compact(cache, slot_pos, keep_pages, page_size, device) -> torch.Tensor:
    """按页保留压缩每层 KV 缓存；返回新的 slot_pos。"""
    slot_page = slot_pos // page_size
    keep_mask = torch.tensor([int(p) in keep_pages for p in slot_page.tolist()],
                             device=device)
    idx = keep_mask.nonzero(as_tuple=True)[0]
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
    return slot_pos.index_select(0, idx)


def _offload_page(cache, slot_pos, store, page, page_size):
    """把某页的每层 KV（post-RoPE）拷到 CPU 缓冲 store；不改缓存（随后由 _compact 丢弃）。"""
    idx = ((slot_pos // page_size) == page).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return
    kv = [(layer.keys.index_select(2, idx).to("cpu"),
           layer.values.index_select(2, idx).to("cpu")) for layer in cache.layers]
    store[int(page)] = {"pos": slot_pos.index_select(0, idx).to("cpu"), "kv": kv}


@torch.no_grad()
def _recall_pages(cache, slot_pos, store, pages, device):
    """把 CPU 缓冲里的若干页 KV 拼回缓存末尾，扩展 slot_pos（注意力对槽序无关，
    每个 key 自带原位 RoPE 相位）。从 store 移除已召回页。返回新 slot_pos。"""
    add_pos = []
    for p in pages:
        ent = store.pop(int(p), None)
        if ent is None:
            continue
        for layer, (k, v) in zip(cache.layers, ent["kv"]):
            layer.keys = torch.cat([layer.keys, k.to(device)], dim=2).contiguous()
            layer.values = torch.cat([layer.values, v.to(device)], dim=2).contiguous()
        add_pos.append(ent["pos"].to(device))
    if add_pos:
        slot_pos = torch.cat([slot_pos] + add_pos)
    return slot_pos


def _smoke(model_path: str, max_new: int = 64):
    """小冒烟：小模型短生成，验证淘汰循环不崩、缓存确实变小、各 arm 都能跑。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager").eval()
    ids = tok("Compute 12*13 step by step.", return_tensors="pt").input_ids
    import json, tempfile, os
    sc = {"feature_names": ["entropy", "cum_attn", "position", "concentration"],
          "w_raw": [1.0, 0.0, 0.0, 0.0], "b_raw": 0.0, "impute_mean": [2, 0, 0, 0]}
    fd, p = tempfile.mkstemp(suffix=".json"); os.write(fd, json.dumps(sc).encode()); os.close(fd)
    for arm in ARMS:
        sig = OnlineSignature(p, page_size=16)
        r = generate_with_evict(model, tok, ids, sig, arm, max_new=max_new,
                                page_size=16, keep_frac=0.3, evict_every=16)
        print(f"{arm:7s}: gen={len(r['gen_ids'])} evict={r['n_evict_rounds']} "
              f"cache_len={r['final_cache_len']} refl={r['n_reflection']}")
        if arm == "full":
            assert r["n_evict_rounds"] == 0
        else:
            assert r["final_cache_len"] <= ids.shape[1] + max_new
    os.unlink(p)
    print("OK: 五臂淘汰循环冒烟通过")


if __name__ == "__main__":
    import sys
    _smoke(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64)
