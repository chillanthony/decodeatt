# ICLR 2026 推理解码与缓存优化论文调研与分析

检索日期：2026-08-12。检索范围：从原 66 篇中移除“算子与架构级计算削减”和“外围或误归类”两类共 5 篇，保留 7 类 61 篇；均已核对 ICLR 2026 官方展示页的摘要与实验描述。

## 分类概览

| 顺序 | 分类 | 论文数 |
|---:|---|---:|
| 1 | 扩散与非自回归解码 | 17 |
| 2 | 投机与多 Token 并行解码 | 14 |
| 3 | KV Cache 压缩与管理 | 10 |
| 4 | 上下文与推理轨迹压缩 | 6 |
| 5 | 服务系统与模型路由 | 6 |
| 6 | 采样与受控解码 | 4 |
| 7 | 内部状态复用与通信 | 4 |
|  | **合计** | **61** |

## 论文列表

| # | 分类 | 论文 | 年份 / 会议 | 数据 / 实验设定 | 核心范式 | 与研究主线的关系 |
|---:|---|---|---|---|---|---|
| 1 | 扩散与非自回归解码 | [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://iclr.cc/virtual/2026/poster/10009203) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 2 | 扩散与非自回归解码 | [AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](https://iclr.cc/virtual/2026/poster/10011958) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 3 | 扩散与非自回归解码 | [Attention Is All You Need for KV Cache in Diffusion LLMs](https://iclr.cc/virtual/2026/poster/10006427) | 2026 / ICLR | GSM8K、MATH、HumanEval；质量与效率指标 | 自适应 KV 复用与并行去噪 | 相邻效率子线 |
| 4 | 扩散与非自回归解码 | [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](https://iclr.cc/virtual/2026/poster/10011297) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 5 | 扩散与非自回归解码 | [d$^2$Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](https://iclr.cc/virtual/2026/poster/10009386) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 自适应 KV 复用与并行去噪 | 相邻效率子线 |
| 6 | 扩散与非自回归解码 | [Diffusion Language Model Knows the Answer Before It Decodes](https://iclr.cc/virtual/2026/poster/10008174) | 2026 / ICLR Oral | GSM8K；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 7 | 扩散与非自回归解码 | [Diffusion Language Models are Provably Optimal Parallel Samplers](https://iclr.cc/virtual/2026/poster/10011442) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 8 | 扩散与非自回归解码 | [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://iclr.cc/virtual/2026/poster/10007008) | 2026 / ICLR | GSM8K、MATH；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 9 | 扩散与非自回归解码 | [Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models](https://iclr.cc/virtual/2026/poster/10010900) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 10 | 扩散与非自回归解码 | [Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](https://iclr.cc/virtual/2026/poster/10009392) | 2026 / ICLR | GSM8K、MATH、HumanEval；质量与效率指标 | 自适应 KV 复用与并行去噪 | 相邻效率子线 |
| 11 | 扩散与非自回归解码 | [ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping](https://iclr.cc/virtual/2026/poster/10009817) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 12 | 扩散与非自回归解码 | [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://iclr.cc/virtual/2026/poster/10011837) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 13 | 扩散与非自回归解码 | [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://iclr.cc/virtual/2026/poster/10011631) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 自适应 KV 复用与并行去噪 | 相邻效率子线 |
| 14 | 扩散与非自回归解码 | [Hierarchy Decoding: A Training-free Parallel Decoding Strategy  for Diffusion Large Language Models](https://iclr.cc/virtual/2026/poster/10008769) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 15 | 扩散与非自回归解码 | [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](https://iclr.cc/virtual/2026/poster/10008632) | 2026 / ICLR | MATH；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 16 | 扩散与非自回归解码 | [Logit‑KL Flow Matching: Non‑Autoregressive Text Generation via Sampling‑Hybrid Inference](https://iclr.cc/virtual/2026/poster/10007040) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 17 | 扩散与非自回归解码 | [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://iclr.cc/virtual/2026/poster/10010055) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 动态解掩码、重掩码或并行生成 | 相邻效率子线 |
| 18 | 投机与多 Token 并行解码 | [Bridging Draft Policy Misalignment: Group Tree Optimization  for Speculative Decoding](https://iclr.cc/virtual/2026/poster/10008369) | 2026 / ICLR | GSM8K、MATH、HumanEval、MT-Bench；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 19 | 投机与多 Token 并行解码 | [Draft-based Approximate Inference for LLMs](https://iclr.cc/virtual/2026/poster/10011888) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 20 | 投机与多 Token 并行解码 | [FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](https://iclr.cc/virtual/2026/poster/10006409) | 2026 / ICLR | MATH；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 21 | 投机与多 Token 并行解码 | [Flatter Tokens are More Valuable for Speculative Draft Model Training](https://iclr.cc/virtual/2026/poster/10006708) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 22 | 投机与多 Token 并行解码 | [Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models](https://iclr.cc/virtual/2026/poster/10007953) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 一次前向预测或验证多个 token | 相邻效率子线 |
| 23 | 投机与多 Token 并行解码 | [Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10010314) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 24 | 投机与多 Token 并行解码 | [Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs](https://iclr.cc/virtual/2026/poster/10010205) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 25 | 投机与多 Token 并行解码 | [Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](https://iclr.cc/virtual/2026/poster/10010023) | 2026 / ICLR Oral | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 26 | 投机与多 Token 并行解码 | [Parallel Token Prediction for  Language Models](https://iclr.cc/virtual/2026/poster/10011054) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 一次前向预测或验证多个 token | 相邻效率子线 |
| 27 | 投机与多 Token 并行解码 | [PARD: Accelerating LLM Inference with Low‑Cost PARallel Draft Model Adaptation](https://iclr.cc/virtual/2026/poster/10008956) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 28 | 投机与多 Token 并行解码 | [RepSpec: Structural Re-parameterized Draft Model Training for Speculative Decoding](https://iclr.cc/virtual/2026/poster/10008568) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 29 | 投机与多 Token 并行解码 | [SpecBranch: Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism](https://iclr.cc/virtual/2026/poster/10010904) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 30 | 投机与多 Token 并行解码 | [Speculative Speculative Decoding](https://iclr.cc/virtual/2026/poster/10008711) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 31 | 投机与多 Token 并行解码 | [Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match](https://iclr.cc/virtual/2026/poster/10010178) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | draft–verify、draft 学习或并行分支 | 相邻效率子线 |
| 32 | KV Cache 压缩与管理 | [Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs](https://iclr.cc/virtual/2026/poster/10007269) | 2026 / ICLR | GSM8K、MATH-500、MATH、AIME24；质量与效率指标 | 重要性预测驱动的 KV 保留与淘汰 | 长上下文效率主线 |
| 33 | KV Cache 压缩与管理 | [DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference](https://iclr.cc/virtual/2026/poster/10007483) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 重要性预测驱动的 KV 保留与淘汰 | 长上下文效率主线 |
| 34 | KV Cache 压缩与管理 | [FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference](https://iclr.cc/virtual/2026/poster/10006722) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | GPU–CPU 分层 KV 检索与调度 | 长上下文效率主线 |
| 35 | KV Cache 压缩与管理 | [IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs](https://iclr.cc/virtual/2026/poster/10006569) | 2026 / ICLR | LongBench；质量与效率指标 | GPU–CPU 分层 KV 检索与调度 | 长上下文效率主线 |
| 36 | KV Cache 压缩与管理 | [LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation](https://iclr.cc/virtual/2026/poster/10009483) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 重要性预测驱动的 KV 保留与淘汰 | 长上下文效率主线 |
| 37 | KV Cache 压缩与管理 | [LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences](https://iclr.cc/virtual/2026/poster/10011378) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | GPU–CPU 分层 KV 检索与调度 | 长上下文效率主线 |
| 38 | KV Cache 压缩与管理 | [PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs](https://iclr.cc/virtual/2026/poster/10009118) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 渐进式混合精度 KV 量化 | 长上下文效率主线 |
| 39 | KV Cache 压缩与管理 | [QuoKA: Query-Oriented KV Selection for Efficient LLM Prefill](https://iclr.cc/virtual/2026/poster/10008892) | 2026 / ICLR | MATH、LongBench、RULER；质量与效率指标 | 稀疏 KV 选择与预算管理 | 长上下文效率主线 |
| 40 | KV Cache 压缩与管理 | [Reconstructing KV Caches with Cross-Layer Fusion for Enhanced Transformers](https://iclr.cc/virtual/2026/poster/10011526) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 跨层 KV 共享与重构 | 长上下文效率主线 |
| 41 | KV Cache 压缩与管理 | [ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](https://iclr.cc/virtual/2026/poster/10009980) | 2026 / ICLR Oral | MATH；质量与效率指标 | 重要性预测驱动的 KV 保留与淘汰 | 长上下文效率主线 |
| 42 | 上下文与推理轨迹压缩 | [Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors](https://iclr.cc/virtual/2026/poster/10011199) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 语义压缩、记忆重写或隐式推理表示 | 相邻效率子线 |
| 43 | 上下文与推理轨迹压缩 | [Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning](https://iclr.cc/virtual/2026/poster/10008228) | 2026 / ICLR | MATH；质量与效率指标 | 语义压缩、记忆重写或隐式推理表示 | 相邻效率子线 |
| 44 | 上下文与推理轨迹压缩 | [COMI: Coarse-to-fine Context Compression via Marginal Information Gain](https://iclr.cc/virtual/2026/poster/10009801) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 语义压缩、记忆重写或隐式推理表示 | 相邻效率子线 |
| 45 | 上下文与推理轨迹压缩 | [KaVa: Latent Reasoning via Compressed KV-Cache Distillation](https://iclr.cc/virtual/2026/poster/10008323) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 语义压缩、记忆重写或隐式推理表示 | 相邻效率子线 |
| 46 | 上下文与推理轨迹压缩 | [Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy](https://iclr.cc/virtual/2026/poster/10008526) | 2026 / ICLR | MATH；质量与效率指标 | 语义压缩、记忆重写或隐式推理表示 | 相邻效率子线 |
| 47 | 上下文与推理轨迹压缩 | [RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers](https://iclr.cc/virtual/2026/poster/10007054) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 语义压缩、记忆重写或隐式推理表示 | 相邻效率子线 |
| 48 | 服务系统与模型路由 | [AdaCache: Adaptive Caching and Context Augmentation for Efficient LLM Serving](https://iclr.cc/virtual/2026/poster/10010915) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 缓存亲和调度、模型路由或异构执行 | 相邻效率子线 |
| 49 | 服务系统与模型路由 | [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://iclr.cc/virtual/2026/poster/10006705) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 缓存亲和调度、模型路由或异构执行 | 相邻效率子线 |
| 50 | 服务系统与模型路由 | [DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving](https://iclr.cc/virtual/2026/poster/10006475) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 缓存亲和调度、模型路由或异构执行 | 相邻效率子线 |
| 51 | 服务系统与模型路由 | [Near-Optimal Online Deployment and Routing for Streaming LLMs](https://iclr.cc/virtual/2026/poster/10010213) | 2026 / ICLR | MATH；质量与效率指标 | 缓存亲和调度、模型路由或异构执行 | 相邻效率子线 |
| 52 | 服务系统与模型路由 | [Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters](https://iclr.cc/virtual/2026/poster/10008093) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 缓存亲和调度、模型路由或异构执行 | 相邻效率子线 |
| 53 | 服务系统与模型路由 | [Universal Model Routing for Efficient LLM Inference](https://iclr.cc/virtual/2026/poster/10007775) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 缓存亲和调度、模型路由或异构执行 | 相邻效率子线 |
| 54 | 采样与受控解码 | [$p\textrm{-less}$ Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding](https://iclr.cc/virtual/2026/poster/10010257) | 2026 / ICLR Oral | MATH；质量与效率指标 | 自适应采样或带约束的推理时控制 | 相邻效率子线 |
| 55 | 采样与受控解码 | [DP-Fusion: Token-Level Differentially Private Inference for Large Language Models](https://iclr.cc/virtual/2026/poster/10009055) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 自适应采样或带约束的推理时控制 | 相邻效率子线 |
| 56 | 采样与受控解码 | [Robust Multi-Objective Controlled Decoding of Large Language Models](https://iclr.cc/virtual/2026/poster/10008004) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 自适应采样或带约束的推理时控制 | 相邻效率子线 |
| 57 | 采样与受控解码 | [THE END OF MANUAL DECODING: TOWARDS TRULY END-TO-END LANGUAGE MODELS](https://iclr.cc/virtual/2026/poster/10008510) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 自适应采样或带约束的推理时控制 | 相邻效率子线 |
| 58 | 内部状态复用与通信 | [Beyond Speedup - Utilizing KV Cache for Sampling and Reasoning](https://iclr.cc/virtual/2026/poster/10010484) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 复用 KV/activation 作为表示或通信介质 | 相邻效率子线 |
| 59 | 内部状态复用与通信 | [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://iclr.cc/virtual/2026/poster/10010020) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 复用 KV/activation 作为表示或通信介质 | 相邻效率子线 |
| 60 | 内部状态复用与通信 | [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://iclr.cc/virtual/2026/poster/10010626) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 复用 KV/activation 作为表示或通信介质 | 相邻效率子线 |
| 61 | 内部状态复用与通信 | [LatentQA: Teaching LLMs to Decode Activations Into Natural Language](https://iclr.cc/virtual/2026/poster/10007461) | 2026 / ICLR | 多模型/多任务；质量与效率指标 | 复用 KV/activation 作为表示或通信介质 | 相邻效率子线 |

## 逐篇要点


### 扩散与非自回归解码（17 篇）

1. **Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
2. **AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
3. **Attention Is All You Need for KV Cache in Diffusion LLMs**：归入“扩散与非自回归解码”；核心范式是自适应 KV 复用与并行去噪，摘要实验覆盖GSM8K、MATH、HumanEval。
4. **Constrained Decoding of Diffusion LLMs with Context-Free Grammars**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
5. **d$^2$Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching**：归入“扩散与非自回归解码”；核心范式是自适应 KV 复用与并行去噪，摘要实验覆盖多模型或多任务实验。
6. **Diffusion Language Model Knows the Answer Before It Decodes**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖GSM8K。
7. **Diffusion Language Models are Provably Optimal Parallel Samplers**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
8. **Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖GSM8K、MATH。
9. **Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
10. **Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM**：归入“扩散与非自回归解码”；核心范式是自适应 KV 复用与并行去噪，摘要实验覆盖GSM8K、MATH、HumanEval。
11. **ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
12. **Fast-dLLM v2: Efficient Block-Diffusion LLM**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
13. **Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding**：归入“扩散与非自回归解码”；核心范式是自适应 KV 复用与并行去噪，摘要实验覆盖多模型或多任务实验。
14. **Hierarchy Decoding: A Training-free Parallel Decoding Strategy  for Diffusion Large Language Models**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
15. **Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖MATH。
16. **Logit‑KL Flow Matching: Non‑Autoregressive Text Generation via Sampling‑Hybrid Inference**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。
17. **ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding**：归入“扩散与非自回归解码”；核心范式是动态解掩码、重掩码或并行生成，摘要实验覆盖多模型或多任务实验。

### 投机与多 Token 并行解码（14 篇）

18. **Bridging Draft Policy Misalignment: Group Tree Optimization  for Speculative Decoding**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖GSM8K、MATH、HumanEval、MT-Bench。
19. **Draft-based Approximate Inference for LLMs**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
20. **FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖MATH。
21. **Flatter Tokens are More Valuable for Speculative Draft Model Training**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
22. **Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models**：归入“投机与多 Token 并行解码”；核心范式是一次前向预测或验证多个 token，摘要实验覆盖多模型或多任务实验。
23. **Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
24. **Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
25. **Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
26. **Parallel Token Prediction for  Language Models**：归入“投机与多 Token 并行解码”；核心范式是一次前向预测或验证多个 token，摘要实验覆盖多模型或多任务实验。
27. **PARD: Accelerating LLM Inference with Low‑Cost PARallel Draft Model Adaptation**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
28. **RepSpec: Structural Re-parameterized Draft Model Training for Speculative Decoding**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
29. **SpecBranch: Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
30. **Speculative Speculative Decoding**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。
31. **Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match**：归入“投机与多 Token 并行解码”；核心范式是draft–verify、draft 学习或并行分支，摘要实验覆盖多模型或多任务实验。

### KV Cache 压缩与管理（10 篇）

32. **Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs**：归入“KV Cache 压缩与管理”；核心范式是重要性预测驱动的 KV 保留与淘汰，摘要实验覆盖GSM8K、MATH-500、MATH、AIME24。
33. **DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference**：归入“KV Cache 压缩与管理”；核心范式是重要性预测驱动的 KV 保留与淘汰，摘要实验覆盖多模型或多任务实验。
34. **FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference**：归入“KV Cache 压缩与管理”；核心范式是GPU–CPU 分层 KV 检索与调度，摘要实验覆盖多模型或多任务实验。
35. **IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs**：归入“KV Cache 压缩与管理”；核心范式是GPU–CPU 分层 KV 检索与调度，摘要实验覆盖LongBench。
36. **LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation**：归入“KV Cache 压缩与管理”；核心范式是重要性预测驱动的 KV 保留与淘汰，摘要实验覆盖多模型或多任务实验。
37. **LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences**：归入“KV Cache 压缩与管理”；核心范式是GPU–CPU 分层 KV 检索与调度，摘要实验覆盖多模型或多任务实验。
38. **PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs**：归入“KV Cache 压缩与管理”；核心范式是渐进式混合精度 KV 量化，摘要实验覆盖多模型或多任务实验。
39. **QuoKA: Query-Oriented KV Selection for Efficient LLM Prefill**：归入“KV Cache 压缩与管理”；核心范式是稀疏 KV 选择与预算管理，摘要实验覆盖MATH、LongBench、RULER。
40. **Reconstructing KV Caches with Cross-Layer Fusion for Enhanced Transformers**：归入“KV Cache 压缩与管理”；核心范式是跨层 KV 共享与重构，摘要实验覆盖多模型或多任务实验。
41. **ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models**：归入“KV Cache 压缩与管理”；核心范式是重要性预测驱动的 KV 保留与淘汰，摘要实验覆盖MATH。

### 上下文与推理轨迹压缩（6 篇）

42. **Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors**：归入“上下文与推理轨迹压缩”；核心范式是语义压缩、记忆重写或隐式推理表示，摘要实验覆盖多模型或多任务实验。
43. **Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning**：归入“上下文与推理轨迹压缩”；核心范式是语义压缩、记忆重写或隐式推理表示，摘要实验覆盖MATH。
44. **COMI: Coarse-to-fine Context Compression via Marginal Information Gain**：归入“上下文与推理轨迹压缩”；核心范式是语义压缩、记忆重写或隐式推理表示，摘要实验覆盖多模型或多任务实验。
45. **KaVa: Latent Reasoning via Compressed KV-Cache Distillation**：归入“上下文与推理轨迹压缩”；核心范式是语义压缩、记忆重写或隐式推理表示，摘要实验覆盖多模型或多任务实验。
46. **Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy**：归入“上下文与推理轨迹压缩”；核心范式是语义压缩、记忆重写或隐式推理表示，摘要实验覆盖MATH。
47. **RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers**：归入“上下文与推理轨迹压缩”；核心范式是语义压缩、记忆重写或隐式推理表示，摘要实验覆盖多模型或多任务实验。

### 服务系统与模型路由（6 篇）

48. **AdaCache: Adaptive Caching and Context Augmentation for Efficient LLM Serving**：归入“服务系统与模型路由”；核心范式是缓存亲和调度、模型路由或异构执行，摘要实验覆盖多模型或多任务实验。
49. **Cascadia: An Efficient Cascade Serving System for Large Language Models**：归入“服务系统与模型路由”；核心范式是缓存亲和调度、模型路由或异构执行，摘要实验覆盖多模型或多任务实验。
50. **DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving**：归入“服务系统与模型路由”；核心范式是缓存亲和调度、模型路由或异构执行，摘要实验覆盖多模型或多任务实验。
51. **Near-Optimal Online Deployment and Routing for Streaming LLMs**：归入“服务系统与模型路由”；核心范式是缓存亲和调度、模型路由或异构执行，摘要实验覆盖MATH。
52. **Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters**：归入“服务系统与模型路由”；核心范式是缓存亲和调度、模型路由或异构执行，摘要实验覆盖多模型或多任务实验。
53. **Universal Model Routing for Efficient LLM Inference**：归入“服务系统与模型路由”；核心范式是缓存亲和调度、模型路由或异构执行，摘要实验覆盖多模型或多任务实验。

### 采样与受控解码（4 篇）

54. **$p\textrm{-less}$ Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding**：归入“采样与受控解码”；核心范式是自适应采样或带约束的推理时控制，摘要实验覆盖MATH。
55. **DP-Fusion: Token-Level Differentially Private Inference for Large Language Models**：归入“采样与受控解码”；核心范式是自适应采样或带约束的推理时控制，摘要实验覆盖多模型或多任务实验。
56. **Robust Multi-Objective Controlled Decoding of Large Language Models**：归入“采样与受控解码”；核心范式是自适应采样或带约束的推理时控制，摘要实验覆盖多模型或多任务实验。
57. **THE END OF MANUAL DECODING: TOWARDS TRULY END-TO-END LANGUAGE MODELS**：归入“采样与受控解码”；核心范式是自适应采样或带约束的推理时控制，摘要实验覆盖多模型或多任务实验。

### 内部状态复用与通信（4 篇）

58. **Beyond Speedup - Utilizing KV Cache for Sampling and Reasoning**：归入“内部状态复用与通信”；核心范式是复用 KV/activation 作为表示或通信介质，摘要实验覆盖多模型或多任务实验。
59. **Cache-to-Cache: Direct Semantic Communication Between Large Language Models**：归入“内部状态复用与通信”；核心范式是复用 KV/activation 作为表示或通信介质，摘要实验覆盖多模型或多任务实验。
60. **KVComm: Enabling Efficient LLM Communication through Selective KV Sharing**：归入“内部状态复用与通信”；核心范式是复用 KV/activation 作为表示或通信介质，摘要实验覆盖多模型或多任务实验。
61. **LatentQA: Teaching LLMs to Decode Activations Into Natural Language**：归入“内部状态复用与通信”；核心范式是复用 KV/activation 作为表示或通信介质，摘要实验覆盖多模型或多任务实验。

## 检索结论

- **已较充分研究**：固定预算下按 token 重要性压缩 KV、draft–verify 投机解码，以及 dLLM 的置信度式并行解码均较拥挤。
- **仍明显不足**：长输入与长输出同时出现时，token 未来效用会漂移；永久淘汰、可逆检索与量化之间尚缺统一的阶段感知策略和 worst-case 保障。
- **数据 / 评测机会**：需要统一覆盖长文档、多轮对话、长 CoT 与代码代理的评测，并同时报告 TTFT、TPOT、吞吐、峰值显存、CPU–GPU 流量和最坏样本精度损失。

---

# ICLR 2026 推理解码与缓存优化分析

## 范式分类表格

| 范式 | 数量 | 内化了什么 | 代表论文 | 仍未解决 |
|---|---:|---|---|---|
| 扩散与非自回归解码 | 17 | 将逐 token 生成改成并行去噪、动态解掩码与缓存复用 | Fast-dLLM、SlowFast、Learn2PD、ReFusion | 真实端到端速度、复杂 kernel、质量稳定性 |
| 投机与多 Token 并行解码 | 14 | 用 draft、树或多 token 头提前提出候选，再由目标模型验证 | HSD、GTO、LTD、SpecBranch | draft 成本、并发气泡、分布漂移与适配成本 |
| KV Cache 压缩与管理 | 10 | 只保留、读取或高精度存储未来有用的 KV | TRIM-KV、DefensiveKV、LookaheadKV、ThinKV | 未来效用漂移、永久误删、跨任务泛化与系统实速 |
| 上下文与推理轨迹压缩 | 6 | 将原始 token/CoT 改写成语义锚点、latent state 或持久记忆 | SAC、COMI、KaVa、RMAAT | 压缩后可验证的信息保真和开放域泛化 |
| 服务系统与模型路由 | 6 | 在请求、模型和设备层面复用缓存与分配算力 | DualMap、Cascadia、UniRoute、Prima.cpp | 动态负载、尾延迟、公平成本核算 |
| 采样与受控解码 | 4 | 在 token 分布层面自动选择采样或约束策略 | $p\textrm{-less}$、RMOD、AutoDeco | 与系统速度和长上下文能力的联系较弱 |
| 内部状态复用与通信 | 4 | 把 KV 或 activation 当作表示、监督或模型间消息 | Cache-to-Cache、KVComm、LatentQA | 异构模型对齐、语义可解释性和协议鲁棒性 |

## ICLR 2026 Oral 论文

| 论文简称 | 归类 | 一句话总结 |
|---|---|---|
| **Prophet** | 扩散与非自回归解码 | 利用扩散语言模型在完整解码前便已确定答案的现象，动态提前结束 refinement，在基本保持质量的同时减少解码步骤。 |
| **HSD** | 投机与多 Token 并行解码 | 通过无损的分层联合验证提高 draft token 接受长度，在维持目标分布不变的前提下加速投机解码。 |
| **ThinKV** | KV Cache 压缩与管理 | 根据长 CoT 中不同 thought 的重要性联合执行 KV 量化与淘汰，以极小缓存维持接近无损的推理准确率并提高吞吐。 |
| **$p$-less Sampling** | 采样与受控解码 | 根据信息论信号逐 token 自动确定截断阈值，消除 top-$p$ 等超参数并改善高温采样的质量与效率。 |

## 方向选择

排序口径是：以长上下文扩张为主线，在 64 张 A800、相对短周期的约束下，优先考虑问题仍开放且能用现有开源模型完成验证的方向。

### 算力筛选

**暂不选择“扩散与非自回归解码”作为主线。** 2026 年该方向已经形成解码策略、缓存、并行生成、蒸馏和系统优化等密集路线；若要在核心模型或架构层形成强贡献，通常需要训练具有竞争力的基础 dLLM，64 张 A800 难以在短周期内覆盖预训练和大规模对照。只做 training-free 加速虽然可行，但赛道已经较拥挤。[2026 dLLM 推理加速综述](https://arxiv.org/abs/2607.12829)

### 推荐排序

| 排名 | 方向 | 2026 最新动态与简要理由 |
|---:|---|---|
| 1 | **KV Cache 压缩与管理** | 最新的 QEvict 已将不可逆 eviction 推进到“量化保存、必要时恢复”的多级管理，说明 attention drift、恢复机制和长输入+长输出联合管理仍是开放问题；与长上下文最直接，64 卡足够进行多模型、多长度和 kernel 实验。[QEvict](https://arxiv.org/abs/2608.05326) |
| 2 | **上下文与推理轨迹压缩** | TaC 表明 reasoning model 本身可以充当 context compressor，研究重点开始从独立压缩器转向任务感知、预算可控和信息保真的压缩；训练规模适中，实验反馈快，创新空间仍大。[Thinking as Compression](https://arxiv.org/abs/2605.28713) |
| 3 | **投机与多 Token 并行解码** | 最新工作开始解决长上下文 verification 的 KV 读取瓶颈和在线适配 draft，而不再只追求 acceptance length；现成框架和模型多、64 卡完全够用，但竞争激烈，必须抓住 long-context verification 这一新问题。[Dustin](https://arxiv.org/abs/2606.24957)、[Test-Time Speculation](https://arxiv.org/abs/2605.09329) |
| 4 | **内部状态复用与通信** | 2026 年研究已扩展到跨架构 latent alignment、KV 通道完整性和安全攻击，并出现“latent 信息更多但未必提升任务性能”的负结果；问题非常开放、算力要求不高，但评测协议尚不成熟且与长上下文扩张的联系较间接。[Latent Communication](https://arxiv.org/abs/2607.14103)、[KV-Cache Integrity](https://arxiv.org/abs/2606.28958) |
| 5 | **服务系统与模型路由** | 最新工作关注未知输出长度下的 KV 预留、异构 serving group 路由，以及多轮对话中的跨模型 KV 共享；64 卡适合做真实集群实验，但需要可控网络、CPU/SSD 层级和 workload trace，工程周期通常长于算法方向。[Robust KV Management](https://arxiv.org/abs/2607.16892)、[SwiftCache](https://arxiv.org/abs/2606.16135) |
| 6 | **采样与受控解码** | $p$-less 等工作显示自适应 temperature、截断阈值和多目标控制仍可研究，且实验成本最低；但该方向与“扩大有效长上下文”的联系最弱，也较难把 64 卡资源转化为明显研究壁垒。[$p$-less Sampling](https://iclr.cc/virtual/2026/poster/10010257) |

### KV Cache论文

1. **ThinKV**：把长 CoT 划分为不同 thought 类型，并联合使用量化与 eviction；它把“重要性随推理阶段变化”推到了方法中心，也是集合中最直接的 Oral 标杆。
2. **LookaheadKV**：用 learned lookahead tokens 和 LoRA 预测未来响应会关注的 prompt KV，绕开显式 draft generation；代表“从历史注意力转向未来效用预测”。
3. **Cache What Lasts / TRIM-KV**：在 token 创建时预测长期保留价值，并让分数随时间衰减；代表轻量 learned retention gate。
4. **DefensiveKV**：不再只优化平均重要性，而是控制极端生成步骤中的最坏风险；代表 KV eviction 的鲁棒性分支。
5. **FreeKV**：不永久删除 KV，而是通过 CPU–GPU 分层存储、预测式 recall 和双缓冲隐藏检索延迟；是 eviction 的关键对照范式。
6. **LouisKV**：利用关键 KV 的时间局部性，只在语义边界触发 retrieval，并区分长输入与长输出；是 long-input + long-output 设定的重要系统基线。
7. **PM-KVQ**：通过渐进式混合精度避免长 CoT 中量化误差累积；说明“低价值 KV 降精度”是永久删除之外的可逆替代。
8. **QuoKA**：从代表性 query 出发选择 key，主要优化 chunked prefill；它明确了 prefill 稀疏化与 decode eviction 的不同目标。
9. **IceCache**：通过语义 token 聚类与 PagedAttention 组织 GPU–CPU 分层 KV，在有限 GPU token 预算下提高缓存命中率与传输效率。
10. **FusedKV**：针对跨层 KV 共享的信息损失，用底层与中层表示融合重构高层 KV，在降低长序列缓存占用的同时缩小与逐层缓存的性能差距。

## CCF-A 投稿标准分析

### 1. Motivation 的层级要求

需要先揭示一个现有方法无法解释的新瓶颈，例如 ThinKV 的 thought-dependent sparsity、DefensiveKV 的极端步骤脆弱性或 LouisKV 的输入/输出 KV 分布差异；单纯换一个 token 打分函数不够。若能在几十张 A800 上给出跨模型、跨长度、跨并发度的工业规模证据，会显著强化故事。

### 2. 方法的差异化锐度

本线可用的差异化轴包括：历史注意力到未来效用、点估计到不确定性、永久删除到多级可逆状态、静态预算到阶段感知预算，以及算法指标到端到端系统指标。最清楚的一句话差异应类似：“不同于 ThinKV 的量化+淘汰和 FreeKV 的全量卸载，我们按未来效用置信度在四级存储状态间动态迁移。”

### 3. 实验体量

建议最低覆盖 2–3 个模型家族、3 种输入/输出长度象限、至少 5 个强 baseline 和 4 类真实任务；硬门槛是同时报告准确率、TTFT、TPOT、吞吐、峰值显存与传输量，并在真实 kernel/serving 实现上测端到端速度，而不是只报告 FLOPs。

### 4. 消融与可解释性

需要分别消融效用预测、阶段识别、状态迁移和预算控制，给出压缩率—质量—延迟曲线、跨层/头预算热图、误删案例与超参敏感性。ThinKV 的 thought 类型分析和 DefensiveKV 的 worst-case 分析可作为模板。

### 5. 写作和故事线

贡献应围绕一个矛盾展开：“永久 eviction 快但不可恢复，retrieval 稳但传输慢，量化可逆但节省有限。”Related Work 必须逐篇对照 TRIM-KV、LookaheadKV、DefensiveKV、ThinKV、FreeKV/LouisKV，而不是仅按 eviction/retrieval/quantization 罗列。

### 6. 会议偏好差异

| 会议 | 偏好 | 相对命中率 | 本方向案例/定位 |
|---|---|---:|---|
| ICLR | 新现象、学习机制、表征与完整实证 | 高 | ThinKV、LookaheadKV、TRIM-KV |
| ICML | 清晰优化目标、学习算法和理论/统计保证 | 中高 | 未来效用学习、不确定性校准 |
| NeurIPS | 方法新颖性与大规模、广泛实验 | 中高 | 跨模型统一 KV policy |
| MLSys | kernel、服务系统与真实端到端收益 | 高 | FreeKV、LouisKV 式系统协同 |
| ACL/EMNLP | 长文档、对话、代码等语言任务收益 | 中 | 任务感知缓存与真实长上下文评测 |
