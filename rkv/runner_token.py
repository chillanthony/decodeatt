"""Step 3:**token 级**淘汰解码循环(对标 R-KV/SnapKV)+ token 级锚点保护。

相比 runner_evict 的页级版,这里淘汰单位是单 token(R-KV 的本事——精准删复读式 reflection
token——只在 token 级才有效)。复用同一套 HF DynamicCache 压缩机制(index_select 槽维)。

后端:
  snapkv —— 重要性(观察窗注意力 maxpool)top-B + recent window + sink。
  rkv    —— 贪心选「重要性高且与已选槽不冗余(token key 余弦)」+ recent + sink。
锚点保护(anchor=高熵 forking token):在预算外额外钉 top-k 高熵 token;
  对照 mode:none / anchor / random / lowent。
"""
from __future__ import annotations

import torch

from rkv.runner_evict import _entropy, _sample


@torch.no_grad()
def _select_tokens(imp, ent, key_rep, slot_pos, budget, recent, sink,
                   backend, anchor_mode, anchor_k, rng, cum=None, con=None, sig=None):
    """返回应保留的**槽下标** LongTensor。imp/ent/key_rep/slot_pos 均按当前缓存槽对齐。
    anchor_mode="sig" 时用完整 4 维签名(熵,cum_attn,位置,concentration)打分,
    sig=(w[4], b);cum/con 为每槽 cum_attn/concentration。"""
    n = slot_pos.numel()
    dev = slot_pos.device
    if n <= budget:
        return torch.arange(n, device=dev)
    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    keep[:sink] = True                       # attention sink
    keep[n - recent:] = True                 # recent window
    free = (~keep).nonzero(as_tuple=True)[0]  # 可被淘汰/挑选的中段槽
    n_pick = max(0, budget - int(keep.sum()))

    order = free[torch.argsort(imp[free], descending=True)]
    if backend == "rkv" and key_rep is not None and n_pick > 0:
        # 向量化 R-KV:在重要性前列 pool 内,redundancy = 与**更重要** token 的最大余弦,
        # score' = importance - λ·redundancy,一次性取 top（贪心的批量近似,无 Python 循环）。
        lam = 0.5
        pool = order[: min(order.numel(), n_pick * 3 + 128)]   # 限制 pool 省算力/显存
        reps = torch.nn.functional.normalize(key_rep[pool], dim=1)   # [K,D]，已按重要性降序
        K = pool.numel()
        cos = reps @ reps.T                                   # [K,K]
        higher = torch.tril(torch.ones(K, K, device=dev, dtype=torch.bool), -1)  # j<i=更重要
        red = cos.masked_fill(~higher, -1e9).max(1).values.clamp(min=0)  # 第0个无更高者→0
        score2 = imp[pool] - lam * red
        picked = pool[torch.argsort(score2, descending=True)][:n_pick]
    else:                                    # snapkv:纯重要性 top
        picked = order[:n_pick]
    keep[picked] = True

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
                score = feat @ sig[0] + sig[1]
                extra = drop[torch.argsort(score[drop], descending=True)][:anchor_k]
            elif anchor_mode == "lowent":
                extra = drop[torch.argsort(ent[drop])][:anchor_k]
            else:                            # random
                extra = drop[torch.randperm(drop.numel(), device=dev)][:anchor_k]
            keep[extra] = True
    return keep.nonzero(as_tuple=True)[0]


def _key_reps(cache, device):
    """每槽代表 key 向量 [n, D]（层/KV头平均）。"""
    acc = None
    for layer in cache.layers:
        k = layer.keys[0].mean(0).float()    # [n, D]
        acc = k if acc is None else acc + k
    return acc / len(cache.layers)


@torch.no_grad()
def generate_token_evict(
    model, tokenizer, input_ids, budget=1024, recent=64, sink=8,
    backend="rkv", anchor_mode="none", anchor_frac=0.05, evict_every=128,
    obs_window=16, obs_decay=0.9, max_new=4096,
    do_sample=True, temperature=0.6, top_p=0.95, seed=0, sig=None,
):
    """token 级淘汰生成(忠实 R-KV 架构 + 支持长生成)。

    速度/忠实兼顾:平时 output_attentions=False(模型若以 sdpa 加载则走快算子),
    **只在每次压缩前的 obs_window 步**用全层注意力算重要性(SnapKV/R-KV 的观察窗思想)。
    缓冲预分配(slot_pos/imp/ent),消掉逐步 torch.cat 的 O(n²),解锁 16k 长生成。
    返回 dict(text, gen_ids, n_evict, final_cache_len)。"""
    device = next(model.parameters()).device
    if do_sample:
        torch.manual_seed(seed)
    input_ids = input_ids.to(device)
    P = input_ids.shape[1]
    anchor_k = int(anchor_frac * budget)
    track = budget < 10 ** 8                          # full 臂超大预算→不淘汰,全程快路径

    use_sig = anchor_mode == "sig"                    # 完整 4 维签名锚点
    N = P + max_new
    slot_pos = torch.zeros(N, dtype=torch.long, device=device); slot_pos[:P] = torch.arange(P, device=device)
    imp = torch.zeros(N, device=device)
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

    for step in range(max_new):
        true_pos = P + step
        e = _entropy(logits)
        nxt = _sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            break
        # 只在"下次压缩前 obs_window 步内"开注意力,算全层重要性
        to_evict = evict_every - ((step + 1) % evict_every or evict_every)
        want_attn = track and to_evict < obs_window
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    use_cache=True, output_attentions=want_attn,
                    attention_mask=torch.ones(1, n + 1, device=device, dtype=torch.long),
                    position_ids=torch.tensor([[true_pos]], device=device),
                    cache_position=torch.tensor([n], device=device))
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos[n] = true_pos; ent[n] = e; imp[n] = 0.0
        if use_sig:
            att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_attn:                                 # 观察窗:全层注意力均值 maxpool 进 imp
            row = torch.zeros(n, device=device)
            for a in out.attentions:
                row += a[0, :, -1, :].mean(0).float()
            row /= len(out.attentions)
            imp[:n] = torch.maximum(imp[:n] * obs_decay, row)
            if use_sig:                               # cum_attn/concentration 统计
                att_sum[:n] += row; att_cnt[:n] += 1.0
                att_max[:n] = torch.maximum(att_max[:n], row)

        if track and (step + 1) % evict_every == 0 and n > budget + anchor_k:
            key_rep = _key_reps(cache, device) if backend == "rkv" else None
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1)
                con = att_max[:n] / (cum + 1e-9)
            idx = _select_tokens(imp[:n], ent[:n], key_rep, slot_pos[:n], budget, recent,
                                 sink, backend, anchor_mode, anchor_k, None,
                                 cum=cum, con=con, sig=sig_t)
            for layer in cache.layers:
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
            k = idx.numel()
            slot_pos[:k] = slot_pos[idx]; imp[:k] = imp[idx]; ent[:k] = ent[idx]
            if use_sig:
                att_sum[:k] = att_sum[idx]; att_cnt[:k] = att_cnt[idx]; att_max[:k] = att_max[idx]
            n = k
            n_evict += 1

    return {"text": tokenizer.decode(gen, skip_special_tokens=True), "gen_ids": gen,
            "n_evict": n_evict, "final_cache_len": n}


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
    imp = torch.zeros(N, device=device); ent = torch.zeros(N, device=device)
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
        imp[n] = 0.0; att_sum[n] = 0.0; att_cnt[n] = 0.0; att_max[n] = 0.0
        n += 1
        if want_attn:
            row = torch.zeros(n, device=device)
            for a in out.attentions: row += a[0, :, -1, :].mean(0).float()
            row /= len(out.attentions)
            imp[:n] = torch.maximum(imp[:n] * obs_decay, row)
            if use_sig:
                att_sum[:n] += row; att_cnt[:n] += 1.0; att_max[:n] = torch.maximum(att_max[:n], row)
        if (t + 1) % evict_every == 0 and n > budget + anchor_k:
            key_rep = _key_reps(cache, device) if backend == "rkv" else None
            cum = con = None
            if use_sig:
                cum = att_sum[:n] / att_cnt[:n].clamp(min=1); con = att_max[:n] / (cum + 1e-9)
            idx = _select_tokens(imp[:n], ent[:n], key_rep, slot_pos[:n], budget, recent, sink,
                                 backend, anchor_mode, anchor_k, None, cum=cum, con=con, sig=sig_t)
            for layer in cache.layers:
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
            k = idx.numel()
            slot_pos[:k] = slot_pos[idx]; imp[:k] = imp[idx]; ent[:k] = ent[idx]
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
    for be in ["snapkv", "rkv"]:
        for am in ["none", "anchor"]:
            r = generate_token_evict(model, tok, ids, budget=128, recent=32, sink=4,
                                     backend=be, anchor_mode=am, evict_every=32, max_new=max_new)
            print(f"backend={be:6s} anchor={am:6s} gen={len(r['gen_ids'])} "
                  f"evict={r['n_evict']} cache={r['final_cache_len']} (budget128)")
    print("TOKEN SMOKE OK")


if __name__ == "__main__":
    import sys
    _smoke(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
