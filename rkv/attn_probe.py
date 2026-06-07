"""廉价注意力抓取：单 query 行 × 全 key，按 page 聚合；receiver-head 反查。

不开 output_attentions（O(L^2) 会炸）；只在需要的步取当前 query 行（O(L)）。
"""
# TODO(step3): def probe_step(model, ids, step, layers, heads) -> per_page_weight
# TODO(step3): def receiver_lookup(per_page_weight, receiver_topk) -> set[int]  # R_t
