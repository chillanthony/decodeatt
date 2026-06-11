"""行级注意力探针：只为**选定的 query 行**计算注意力，解锁 16k-32k 长序列。

probe_sequence（eager + 全量 [L,L]）在 L>4k 时显存爆炸。本模块改为：
- 前向用 sdpa/flash（不物化注意力矩阵），hook 住每层 q_proj/k_proj 的输出；
- q 只留选定行，k 留全长；前向后逐层手动做 RoPE + GQA 展开 + softmax(qK^T/√d)，
  只算 [n_rows, L] 而非 [L, L]，立即聚合到 page。
- 显存 ~ O(n_rows × L)，32k 序列 + ~1000 行在 A100-40G 上轻松。

与 probe_sequence 的数值等价性由 _selftest 验证（同模型同输入同行，逐元素对比）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [1, h, n, D]; cos/sin: [1, n, D]（与 transformers apply_rotary_pos_emb 等价）。"""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * c + torch.cat((-x2, x1), dim=-1) * s


@torch.no_grad()
def probe_rows(
    model,
    input_ids: torch.Tensor,
    rows: list[int],
    page_size: int = 16,
    layers=None,
    head_chunk: int = 7,
) -> torch.Tensor:
    """返回 [n_rows, num_pages]：第 i 行 = 查询位置 rows[i] 对各 key-page 的注意力
    （选定层/全头平均，行和=1）。模型可用 sdpa/flash 加载（不要求 eager）。"""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    L = input_ids.shape[1]
    num_pages = (L + page_size - 1) // page_size
    rows = sorted(rows)
    rows_t = torch.tensor(rows, device=device)
    n = len(rows)

    cfg = model.config
    H = cfg.num_attention_heads
    KV = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    groups = H // KV

    base = getattr(model, "model", model)
    attn_mods = [m for nm, m in model.named_modules() if nm.endswith(".self_attn")]
    sel = set(layers) if layers is not None else None
    qk: dict[int, dict] = {}
    handles = []

    def mk(idx, kind):
        def hook(module, inputs, output):
            if sel is not None and idx not in sel:
                return
            d = qk.setdefault(idx, {})
            d[kind] = (output[:, rows_t, :] if kind == "q" else output).detach()
        return hook

    for i, m in enumerate(attn_mods):
        handles.append(m.q_proj.register_forward_hook(mk(i, "q")))
        handles.append(m.k_proj.register_forward_hook(mk(i, "k")))
    try:
        base(input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    pos = torch.arange(L, device=device).unsqueeze(0)
    cos, sin = base.rotary_emb(torch.zeros(1, 1, D, device=device), pos)  # [1, L, D]
    cos, sin = cos.float(), sin.float()
    causal = pos[0].unsqueeze(0) <= rows_t.unsqueeze(1)                   # [n, L]

    acc = torch.zeros(n, num_pages, dtype=torch.float32, device=device)
    pad = num_pages * page_size - L
    count = 0
    for idx in sorted(qk):
        q = qk[idx]["q"].view(1, n, H, D).transpose(1, 2).float()         # [1,H,n,D]
        k = qk[idx]["k"].view(1, L, KV, D).transpose(1, 2).float()        # [1,KV,L,D]
        q = _rope(q, cos[:, rows_t], sin[:, rows_t])
        k = _rope(k, cos, sin)
        k = k.repeat_interleave(groups, dim=1)[0]                         # [H,L,D]
        q = q[0]                                                          # [H,n,D]
        row_attn = torch.zeros(n, L, dtype=torch.float32, device=device)
        for h0 in range(0, H, head_chunk):
            logits = torch.einsum("hnd,hld->hnl", q[h0:h0 + head_chunk],
                                  k[h0:h0 + head_chunk]) / (D ** 0.5)
            logits.masked_fill_(~causal.unsqueeze(0), float("-inf"))
            row_attn += logits.softmax(-1).sum(0)
        a = row_attn / H
        if pad > 0:
            a = F.pad(a, (0, pad))
        acc += a.view(n, num_pages, page_size).sum(-1)
        count += 1
        qk[idx] = None                                                    # 释放
    return (acc / max(count, 1)).cpu()


@torch.no_grad()
def receiver_lookup_row(
    row_attn: torch.Tensor,
    pos: int,
    page_size: int = 16,
    receiver_topk: int = 8,
    exclude_recent_pages: int = 4,
) -> set[int]:
    """对某行（查询位置 pos 的 [num_pages] 注意力向量）取 Top-k 早期 page = R_t。
    与 attn_probe.receiver_lookup 同语义，但输入是单行而非全矩阵。"""
    row = row_attn.clone()
    cur_page = pos // page_size
    lo = max(0, cur_page - exclude_recent_pages)
    row[lo:] = -1.0
    valid = int((row >= 0).sum().item())
    k = min(receiver_topk, valid)
    if k <= 0:
        return set()
    return set(torch.topk(row, k).indices.tolist())


def _selftest(model_path: str, max_len: int = 512):
    """数值等价性自测：probe_rows vs probe_sequence（eager 全量）在同行上的逐元素对比。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rkv.attn_probe import probe_sequence

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    ).eval()
    text = "Solve: what is the remainder when 7^100 is divided by 13? " * 40
    ids = tok(text, return_tensors="pt").input_ids[:, :max_len]
    L = ids.shape[1]
    rows = [L // 4, L // 2, L - 2]

    full = probe_sequence(model, ids, page_size=16)          # [L, num_pages]
    part = probe_rows(model, ids, rows, page_size=16)        # [n, num_pages]
    ref = full[torch.tensor(rows)].cpu()
    diff = (part - ref).abs().max().item()
    print(f"L={L} rows={rows} max|diff|={diff:.2e} "
          f"row_sums={part.sum(-1).tolist()}")
    assert diff < 5e-3, f"FAIL: 探针不等价 diff={diff}"
    print("OK: probe_rows 与 probe_sequence 数值一致")


if __name__ == "__main__":
    import sys

    _selftest(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512)
