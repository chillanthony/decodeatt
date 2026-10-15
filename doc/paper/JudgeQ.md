# JudgeQ 精读笔记

## 1. 研究背景与动机
- 论文：Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction。
- 问题：长输入下 KV cache 线性增长，prefill 阶段就需要决定保留哪些 key/value。
- 现有 eviction 方法常用最后一个 local window 的 query 计算 KV 重要性，因为问题通常在输入末尾。
- 缺陷：last window 容易过度关注局部信息；如果问题不在末尾，或全局证据分散，重要 token 会被误删。
- 作者提出的核心想法：用可训练 soft tokens 模拟未来 decoded tokens 的 attention，从而更接近“真实生成 token 选择 KV”的上界。

## 2. 方法详解
- Judge Q 在模型词表中加入 n 个 learnable soft tokens，只训练这些 soft token 对应的 embedding 参数，其余模型冻结。
- 训练时构造两类输入：Prompt + Soft，以及 Prompt + Response。
- 目标是让 soft tokens 对 prompt 的 attention map 与真实 response tokens 对 prompt 的 attention map 对齐。
- 损失函数是两个 attention map 的 MSE，使 soft tokens 学会关注后续生成真正需要的 prompt 区域。
- 推理时，在 prefill 末尾追加训练好的 soft tokens，计算它们对原始输入 token 的 attention score。
- 根据 soft-token attention score 保留 top-k KV，丢弃其他 KV；随后移除 soft tokens，使用裁剪后的 KV cache 正常 decode。
- 方法本质：用一组“judge queries”替代 last-window queries，提供更全局、更接近未来生成需求的 eviction 打分。
- 它主要处理 prefill-stage KV eviction，不是 decode 过程中持续流式淘汰。

## 3. 实验设计评估
- 训练数据：ShareGPT 50K samples，其中 45K common domain，5K computer；默认 soft token 数 n=32。
- 模型：Llama-3.1-8B-Instruct 与 Mistral-7B-Instruct-v0.3。
- benchmark：LongBench、RULER、Needle-in-a-Haystack。
- baseline：StreamingLLM、H2O、SnapKV、PyramidKV，并在相同 local window 和 budget 下比较。
- LongBench：同预算下平均约提升 1 分，低 budget 场景提升更明显，部分任务甚至超过 Full KV。
- RULER：在 8192/32768 长度和多个 budget 下，通常比 baseline 高 3 分以上，最高接近 10 分。
- Needle-in-a-Haystack：显著优于 baseline，说明 soft queries 对检索型长上下文更稳。
- 机制验证：提出 critical key-value hit rate，Judge Q 相比 SnapKV 高约 8 个点，更接近真实 decoded tokens 的选择。

## 4. 局限性与未来方向
- 需要为目标模型训练 soft tokens，虽成本低，但仍不是完全 training-free。
- 训练数据分布会影响 soft tokens 的全局关注模式；跨任务、跨语言、跨领域迁移能力需要进一步验证。
- 方法只在 prefill 阶段做 eviction，作者未来也计划扩展到 streaming KV eviction during decoding。
- 如果 response attention 本身不是最优重要性标签，则 MSE 对齐可能继承 decoded tokens attention 的偏差。
- 【⚠️ 存疑】论文主要评估 7B/8B 级模型，soft tokens 在更大模型和更长上下文下是否仍稳定，需要实测。

## 5. 评价
- 贡献点：把 KV eviction 的 query 来源从启发式 last window 改成可训练 soft query，针对“如何判断重要 KV”这一核心问题。
- 优势：改动小，只训练 embedding 中新增 soft token；推理时可接入现有 top-k eviction pipeline。
- 对 KV 淘汰研究的启发：重要性打分的 query 选择非常关键，last-window attention 不是可靠全局代理。
- 代价：它提升的是 eviction 决策质量，不直接降低 attention 计算复杂度；仍需和具体 KV budget/稀疏实现结合。
- 总体判断：Judge Q 是偏“importance estimator”的 KV eviction 方法，适合与 SnapKV/PyramidKV 类方法对比或组合。
