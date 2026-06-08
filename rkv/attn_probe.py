"""注意力探针：取每个解码步对各 early page 的注意力；receiver-head 反查 R_t。

核心：
- `probe_sequence`：一次前向（output_attentions，eager），对每层每头的注意力**在 key 维按 page
  聚合**，再对选定层/全头取平均，得到 [L_query, num_pages] 的"步 -> page 注意力"矩阵。
- `receiver_lookup`：对某 reflection 步取该行 Top-k 的**早期** page（屏蔽最近窗口）= R_t。

GQA 说明：eager 路径返回的 attentions 形状是 [B, num_attention_heads, Lq, Lk]（KV 头已
按组重复展开到 Q 头），所以直接对 head 维取平均即可，无需手工处理 GQA。

内存：output_attentions 会同时持有所有层的 [H,L,L]。L 越长越贵（bf16 下 L=2048≈6.6G，
L=4096≈26G）。长 trace 用 `max_len` 截断、或用 `layers` 只取少数 receiver 层来控量。
模型须以 attn_implementation="eager" 加载（sdpa/flash 不返回注意力）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def probe_sequence(
    model,
    input_ids,
    page_size: int = 16,
    layers=None,
    max_len: int | None = None,
) -> torch.Tensor:
    """返回 [L_query, num_pages]：第 q 行 = 查询步 q 对各 key-page 的注意力（选定层/全头平均）。

    内存安全实现：用 forward hook 在**每层算完即把注意力聚合到 page**，并把模块输出的
    attn_weights 置 None，阻止 transformers 累积所有层的 [H,L,L]（否则长序列必 OOM）。
    显存峰值只占单层注意力。
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if max_len is not None and input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    L = input_ids.shape[1]
    num_pages = (L + page_size - 1) // page_size

    attn_mods = [m for n, m in model.named_modules() if n.endswith(".self_attn")]
    sel = set(layers) if layers is not None else None
    acc = torch.zeros(L, num_pages, dtype=torch.float32, device=device)
    count = [0]
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            if not (isinstance(output, tuple) and len(output) >= 2 and output[1] is not None):
                return output
            if sel is not None and idx not in sel:
                return (output[0], None) + tuple(output[2:])  # 不选的层也释放
            aw = output[1]                       # [B, H, Lq, Lk]
            a = aw[0].float().mean(0)            # 对 head 平均 -> [Lq, Lk]
            pad = num_pages * page_size - a.shape[-1]
            if pad > 0:
                a = F.pad(a, (0, pad))
            acc.add_(a.view(a.shape[0], num_pages, page_size).sum(-1))  # 聚合到 page
            count[0] += 1
            return (output[0], None) + tuple(output[2:])   # 置 None，阻止累积
        return hook

    for i, m in enumerate(attn_mods):
        handles.append(m.register_forward_hook(make_hook(i)))
    try:
        model(input_ids, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return acc / max(count[0], 1)


@torch.no_grad()
def receiver_lookup(
    page_attn: torch.Tensor,
    step: int,
    page_size: int = 16,
    receiver_topk: int = 8,
    exclude_recent_pages: int = 4,
) -> set[int]:
    """对查询步 `step` 取 Top-k 的早期 page（屏蔽当前及最近 exclude_recent_pages 个 page）= R_t。"""
    row = page_attn[step].clone()
    cur_page = step // page_size
    lo = max(0, cur_page - exclude_recent_pages)
    row[lo:] = -1.0                          # 只看"早期回指"，排除局部最近窗口
    valid = int((row >= 0).sum().item())
    k = min(receiver_topk, valid)
    if k <= 0:
        return set()
    idx = torch.topk(row, k).indices.tolist()
    return set(idx)
