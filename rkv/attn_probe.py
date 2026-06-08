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
    """返回 [L_query, num_pages]：第 q 行 = 查询步 q 对各 key-page 的注意力（选定层/全头平均）。"""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if max_len is not None and input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    L = input_ids.shape[1]

    out = model(input_ids, output_attentions=True, use_cache=False)
    atts = out.attentions  # tuple[num_layers]，每个 [B, H, L, L]
    if layers is None:
        layers = list(range(len(atts)))

    num_pages = (L + page_size - 1) // page_size
    acc = torch.zeros(L, L, dtype=torch.float32, device=atts[0].device)
    for li in layers:
        acc += atts[li][0].float().mean(0)  # 对 head 取平均 -> [L_query, L_key]
    acc /= len(layers)

    pad = num_pages * page_size - L
    if pad:
        acc = F.pad(acc, (0, pad))          # 在 key 维补齐到整 page
    page_attn = acc.view(L, num_pages, page_size).sum(-1)  # [L_query, num_pages]
    return page_attn


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
