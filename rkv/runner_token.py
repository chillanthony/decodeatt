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
                   backend, anchor_mode, anchor_k, rng):
    """返回应保留的**槽下标** LongTensor。imp/ent/key_rep/slot_pos 均按当前缓存槽对齐。"""
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
        # 贪心:重要性 - λ·与已选最大余弦
        reps = torch.nn.functional.normalize(key_rep, dim=1)
        sel, selreps = [], []
        lam = 0.5
        cand = order[: min(order.numel(), n_pick * 4 + 64)]  # 只在重要性前列里贪心
        cand = cand.tolist()
        while cand and len(sel) < n_pick:
            best, bestv = None, -1e9
            for s in cand[:96]:
                red = 0.0 if not selreps else float((reps[s:s+1] @ torch.stack(selreps).T).max())
                v = float(imp[s]) - lam * red
                if v > bestv:
                    bestv, best = v, s
            sel.append(best); selreps.append(reps[best]); cand.remove(best)
        picked = torch.tensor(sel, device=dev) if sel else order[:0]
    else:                                    # snapkv:纯重要性 top
        picked = order[:n_pick]
    keep[picked] = True

    # 锚点保护:预算外额外钉 anchor_k 个槽
    if anchor_mode != "none" and anchor_k > 0:
        drop = (~keep).nonzero(as_tuple=True)[0]
        if drop.numel():
            if anchor_mode == "anchor":
                extra = drop[torch.argsort(ent[drop], descending=True)][:anchor_k]
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
    model, tokenizer, input_ids, budget=1024, recent=256, sink=8,
    backend="rkv", anchor_mode="none", anchor_frac=0.05, evict_every=64,
    obs_decay=0.9, max_new=4096, do_sample=True, temperature=0.6, top_p=0.95, seed=0,
):
    """token 级淘汰生成。返回 dict(text, gen_ids, n_evict, final_cache_len)。
    anchor_mode: none/anchor/random/lowent;anchor_k = anchor_frac×budget。"""
    device = next(model.parameters()).device
    if do_sample:
        torch.manual_seed(seed)
    input_ids = input_ids.to(device)
    P = input_ids.shape[1]
    anchor_k = int(anchor_frac * budget)

    out = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    cache = out.past_key_values
    logits = out.logits[:, -1]
    slot_pos = torch.arange(P, device=device)
    imp = torch.zeros(P, device=device)               # 每槽重要性（观察窗 maxpool+衰减）
    ent = torch.zeros(P, device=device)               # 每槽熵（prompt 记 0）
    eos = tokenizer.eos_token_id
    gen, n_evict = [], 0

    for step in range(max_new):
        true_pos = P + step
        e = _entropy(logits)                          # 即将生成 token 的熵
        nxt = _sample(logits, do_sample, temperature, top_p)
        gen.append(nxt)
        if nxt == eos:
            break
        cache_len = slot_pos.numel()
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    use_cache=True, output_attentions=True,
                    attention_mask=torch.ones(1, cache_len + 1, device=device, dtype=torch.long),
                    position_ids=torch.tensor([[true_pos]], device=device),
                    cache_position=torch.tensor([cache_len], device=device))
        cache = out.past_key_values
        logits = out.logits[:, -1]
        slot_pos = torch.cat([slot_pos, torch.tensor([true_pos], device=device)])
        imp = torch.cat([imp, torch.zeros(1, device=device)])
        ent = torch.cat([ent, torch.tensor([e], device=device)])
        # 观察窗：当前 query 对各槽注意力（层/头平均）maxpool 进 imp（带衰减）
        row = torch.zeros(slot_pos.numel(), device=device)
        for a in out.attentions:
            row += a[0, :, -1, :].mean(0).float()
        row /= len(out.attentions)
        imp = torch.maximum(imp * obs_decay, row)

        if (step + 1) % evict_every == 0 and slot_pos.numel() > budget + anchor_k:
            key_rep = _key_reps(cache, device) if backend == "rkv" else None
            idx = _select_tokens(imp, ent, key_rep, slot_pos, budget, recent, sink,
                                 backend, anchor_mode, anchor_k, None)
            for layer in cache.layers:
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
            slot_pos, imp, ent = slot_pos[idx], imp[idx], ent[idx]
            n_evict += 1

    return {"text": tokenizer.decode(gen, skip_special_tokens=True), "gen_ids": gen,
            "n_evict": n_evict, "final_cache_len": int(slot_pos.numel())}


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
