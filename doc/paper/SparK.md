# SparK 精读笔记

## 1. 研究背景与动机
- 论文：SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning。
- 问题：长上下文推理中 KV cache 随序列长度线性增长，attention 计算和显存都成为瓶颈。
- 现有 KV 压缩多沿 temporal axis 做 token eviction/merge，较少处理 channel axis 的细粒度冗余。
- 作者观察：不同 token、不同 query 对 key channel 的依赖高度不均匀，固定 channel mask 或结构化剪枝会损害注意力表达。
- 核心目标：在不丢弃 token 的情况下，对 key cache 的 channel 做 query-aware unstructured pruning，并在 attention 计算时恢复被剪通道。

## 2. 方法详解
- SparK 是 training-free、plug-and-play、model-agnostic 的 channel-level KV pruning 方法。
- 它把每个 head、每个 token 的 channel 选择建模为保留 T 个最重要 channel 的优化问题。
- 重要性代理分数为 query/key 对应 channel 范数乘积，用于近似该 channel 对 QK dot-product 的贡献。
- prefill 阶段：用最后 observation window 的 mean query 近似 query 分布，计算每个 token-channel 的 saliency。
- 然后对每个 token 保留 top-T channel，得到 unstructured binary mask；剪掉的不是整 token，而是 key cache 中部分 channel entries。
- 为了 decode 时恢复，SparK 存储剪枝通道的统计信息，如均值、标准差或 pruned entries 的均值。
- decode 阶段：根据 recovery function 和缓存分布统计近似恢复被剪 channel，再进行标准 attention score 计算。
- 方法强调 recoverable pruning：相比直接置零/删除，恢复低信息通道能保持 attention 结构，降低高剪枝率下的质量崩塌。

## 3. 实验设计评估
- benchmark：LongBench 与 RULER。
- 模型：LLaMA-3/3.1-8B/70B-Instruct，Qwen3-8B/32B。
- baseline：Full KV、StreamingLLM、PyramidKV、SnapKV、ExpectedAttention，以及 channel pruning baseline THINK。
- 默认设置：主要剪 key cache，使用 degenerate distribution 作为 recovery 策略，并与 token eviction 方法组合测试。
- LongBench：SparK 可与 SnapKV/PyramidKV 叠加，在相同 token budget 下进一步降低 cache 存储。
- 高剪枝率：80% key-channel pruning 相当于约 40% 总 KV cache memory reduction，性能损失通常小于 5%。
- RULER：在 20% cache budget 和 16K 输入下，SparK(0.8) 仍接近原 eviction baseline，而 THINK(0.8) 经常崩溃到极低分。
- 吞吐：论文称恢复机制带来的 decode overhead 很小，长输入下吞吐接近 THINK，并能支持 full-cache OOM 的更长序列。

## 4. 局限性与未来方向
- recovery 会增加额外计算，论文附录明确指出 TTFT 在低延迟场景可能变差。
- 短输入下 KV cache 本身不大，动态 channel scoring 与 recovery 的收益可能抵不过开销。
- Value pruning 仍偏 heuristic，论文主要强调 key cache channel pruning。
- 方法依赖 saliency proxy 和 observation window；如果后续 query 分布显著变化，prefill 中估计的 channel mask 可能不再最优。
- 【⚠️ 存疑】论文宣称 plug-and-play，但高效实现需要处理不规则 channel mask 和 recovery，真实系统收益依赖 kernel 支持。

## 5. 评价
- 贡献点：把 KV 压缩从“删 token”扩展到“删 key channel”，且用 recovery 避免高剪枝率下注意力结构破坏。
- 方法优势：与 token eviction、quantization 正交，可叠加；适合在已有 KV eviction 方法上继续压缩 memory。
- 对 KVCache 实验的启发：如果 token-level eviction 已接近上限，channel-level redundancy 是下一层可压缩空间。
- 代价：不规则 channel sparsity 的工程复杂度高，且恢复机制可能降低首 token 延迟。
- 总体判断：SparK 是 2026 AAAI 中与 KV 压缩高度相关的 channel-axis 工作，适合作为 token eviction 之外的强相关对照。
