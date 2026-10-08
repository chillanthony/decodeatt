# 长推理 Decoding 阶段稀疏注意力 论文阅读列表

**方向定位**:推理模型(DeepSeek-R1、QwQ、o1 等)在 decoding 阶段生成数万 token 的长链思维(long CoT),自回归 decoding 阶段的注意力计算成为新的瓶颈,与传统 prefilling 优化形成对比。本列表聚焦:
1. 推理模型专用的注意力/KV 优化
2. 长 CoT 的注意力模式分析
3. Decoding/generation 阶段的稀疏注意力(而非 prefilling)
4. KV 选择/淘汰用于 decoding 加速
5. 头功能分类(检索头、流式头、归纳头等)
6. Test-time compute scaling 与注意力的相互作用

---

| #   | 论文                                     | 年份   | 发表                                 | 引用数        | 关键词                                                                                           |
| --- | -------------------------------------- | ---- | ---------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| 1   | SeerAttention-R                        | 2025 | arxiv (Microsoft)                  | 近期发表,引用待累积 | 推理模型稀疏注意力、自蒸馏门控、长生成、TileLang、FlashAttention-3 加速                                              |
| 2   | RaaS                                   | 2025 | arxiv                              | 近期发表,引用待累积 | 推理感知稀疏、milestone token、O(L) 时间内存、CoT decoding                                                 |
| 3   | R-KV                                   | 2025 | NeurIPS 2025                       | 近期发表,引用待累积 | 推理模型 KV 缓存、冗余感知、reflection 压缩、6.6× 吞吐                                                         |
| 4   | ThinKV                                 | 2026 | ICLR 2026 Oral (NVIDIA)            | 近期发表,引用待累积 | thought-adaptive KV、混合量化-淘汰、思维类型分类、PagedAttention                                             |
| 5   | LazyEviction                           | 2025 | arxiv                              | 近期发表,引用待累积 | 滞后 KV 淘汰、Token Importance Recurrence、长推理                                                      |
| 6   | LessIsMore                             | 2025 | arxiv (Princeton+CMU)              | 近期发表,引用待累积 | 训练免训稀疏、跨头共享 token 选择、推理任务                                                                     |
| 7   | DELTA                                  | 2025 | arxiv                              | 近期发表,引用待累积 | 层感知 token 注意力、Δ-selection、长上下文推理                                                              |
| 8   | RLKV                                   | 2025 | arxiv                              | 近期发表,引用待累积 | RL 引导 KV 压缩、推理关键头识别、头功能分解                                                                     |
| 9   | Lil (Less is Less)                     | 2025 | arxiv                              | 近期发表,引用待累积 | 长 decode 信息损失分析、early-stopping、稀疏负效应                                                          |
| 10  | RetroAttention                         | 2025 | arxiv                              | 近期发表,引用待累积 | 回溯注意力修正、累积误差、长生成精度                                                                            |
| 11  | SpecAttn                               | 2025 | arxiv                              | 近期发表,引用待累积 | 稀疏-投机解码协同、verification 阶段免费 oracle、AIME                                                       |
| 12  | NSA                                    | 2025 | ACL 2025 (Best Paper)              | ~150       | 原生稀疏注意力、训练时稀疏、decoding 加速 11.6×                                                               |
| 13  | MoBA                                   | 2025 | arxiv (Kimi/MoonshotAI)            | ~200       | 块级稀疏注意力、MoE式注意力、长上下文、可切换全/稀疏                                                                  |
| 14  | DeepSeek Sparse Attention (DSA)        | 2025 | arxiv (DeepSeek)                   | 近期发表,引用待累积 | Lightning indexer、token selector、MLA 集成、生产部署                                                  |
| 15  | MiniMax-M1                             | 2025 | arxiv (MiniMax)                    | 近期发表,引用待累积 | Lightning Attention、混合注意力、MoE、长 CoT 推理                                                        |
| 16  | Quest                                  | 2024 | ICML 2024                          | ~300       | query-aware KV 选择、page 级稀疏、decoding 加速                                                        |
| 17  | DuoAttention                           | 2025 | ICLR 2025                          | ~150       | 检索头 vs 流式头、KV 缓存差异化、长上下文推理                                                                    |
| 18  | Ada-KV                                 | 2025 | NeurIPS 2025                       | ~80        | KV 缓存淘汰、自适应预算分配、逐头优化、即插即用                                                                     |
| 19  | Gated Attention (Qwen)                 | 2025 | NeurIPS 2025 (Best Paper)          | ~500       | 注意力门控、稀疏性、非线性、无注意力汇聚                                                                          |
| 20  | Twilight                               | 2025 | NeurIPS 2025 (Spotlight)           | ~50        | 自适应稀疏注意力、Top-p 剪枝、层次化、可组合优化                                                                   |
| 21  | Retrieval Head                         | 2025 | ICLR 2025                          | ~200       | 检索头机制、长上下文事实性、头功能解释                                                                           |
| 22  | AttentionPredictor                     | 2025 | NeurIPS 2025                       | 近期发表,引用待累积 | 时空注意力预测、卷积建模、KV 压缩 13×、跨 token 预取                                                             |
| 23  | RocketKV                               | 2025 | ICML 2025                          | 近期发表,引用待累积 | 两阶段 KV 压缩、SnapKV+top-K 混合、decode 加速 3.7×                                                      |
| 24  | DMS (Inference-Time Hyper-Scaling)     | 2025 | NeurIPS 2025 (NVIDIA+Edinburgh)    | 近期发表,引用待累积 | TTS+KV 压缩、淘汰感知训练、推理预算放大                                                                       |
| 25  | HeadKV (Not All Heads Matter)          | 2025 | ICLR 2025                          | ~100       | 头级 KV 压缩、检索+推理打分、头功能选择                                                                        |
| 26  | Expected Attention                     | 2025 | arxiv (NVIDIA)                     | 近期发表,引用待累积 | 期望注意力闭式估计、未来 query 分布、训练免训                                                                    |
| 27  | Hold Onto That Thought                 | 2025 | NeurIPS 2025 ER Workshop           | 近期发表,引用待累积 | KV 压缩在推理任务实证研究、SnapKV vs H2O、副作用揭示                                                            |
| 28  | Hardware-Efficient Attention (GTA/GLA) | 2025 | arxiv (Princeton+Tri Dao)          | 近期发表,引用待累积 | Grouped-Tied/Latent Attention、decoding 算术强度、MLA 改进                                            |
| 29  | NOSA                                   | 2025 | arxiv 2510.13602                   | 近期发表,引用待累积 | KV cache offloading、CPU-GPU 传输约束、局部性分解、decoding 吞吐 5.04×                                      |
| 30  | Thought Anchors                        | 2025 | arxiv 2506.19143 (ICLR 2026 在审)    | 近期发表,引用待累积 | thought anchor 识别、receiver heads、black-box 重采样 + white-box 注意力聚合 + 因果 ablation                |
| 31  | SEAL                                   | 2025 | COLM 2025 (arxiv 2504.07986)       | 近期发表,引用待累积 | execution/reflection/transition 三类 thought 分类、latent steering 校准                              |
| 32  | SSA (Sparse Sparse Attention)          | 2025 | arxiv 2511.20102                   | 近期发表,引用待累积 | 稀疏注意力误差线性上界、特征空间对齐、dropped attention mass                                                     |
| 33  | ASEntmax                               | 2025 | ICLR 2026 (arxiv 2506.16640)       | 近期发表,引用待累积 | α-entmax + 可学温度、长度外推 1000×、function-sparse                                                    |
| 34  | Crystal-KV                             | 2026 | arxiv 2601.16986                   | 近期发表,引用待累积 | Answer-First Principle、think 阶段答案相关 KV 保留、SlipKV vs CrystalKV                                 |
| 35  | ForesightKV                            | 2026 | arxiv 2602.03203                   | 近期发表,引用待累积 | 推理模型 KV 淘汰、long-term contribution 学习、golden eviction SFT + MDP/GRPO、low-entropy 纠错关键 token 保护 |
| 36  | DefensiveKV (Taming Fragility)         | 2026 | ICLR 2026 (arxiv 2510.13334)       | 近期发表,引用待累积 | KV 淘汰脆弱性、重要性骤变、max-aggregation 最坏情况风险、prior-risk 校正、18 数据集近无损                                 |
| 37  | G-KV                                   | 2025 | arxiv 2512.00504                   | 近期发表,引用待累积 | decoding-time KV 淘汰、global attention 打分、推理任务、training-free、AIME/LiveCodeBench                 |
| 38  | LookaheadKV                            | 2026 | ICLR 2026 (arxiv 2603.10899)       | 近期发表,引用待累积 | learned lookahead tokens、LoRA、未来注意力预测、proactive 保留、LongBench、长上下文                               |
| 39  | CaliDrop                               | 2025 | arxiv 2507.19906                   | 近期发表,引用待累积 | KV 淘汰后不删、offload + query 相似度投机校准、可叠加任意淘汰后端                                                    |
| 40  | TriAttention                           | 2026 | arxiv 2604.04921 (MIT/NVIDIA/浙大)   | 近期发表,引用待累积 | pre-RoPE 三角距离打分、Recursive State Query benchmark、回溯压力测试                                        |
| 41  | SparK                                  | 2025 | arxiv 2508.15212 (AMD)             | 近期发表,引用待累积 | channel 维剪枝、动态恢复、training-free、与 KV 压缩正交                                                      |
| 42  | GraphKV                                | 2025 | EMNLP 2025 main (arxiv 2509.00388) | 近期发表,引用待累积 | 图传播动态重要性更新、plug-and-play 叠加 SnapKV/PyramidKV                                                  |
| 43  | TRIM-KV (Cache What Lasts)             | 2026 | ICLR 2026 (arxiv 2512.03324)       | 近期发表,引用待累积 | token 创建时轻量 retention gate、长期重要性、分数时间衰减、蒸馏微调                                             |
| 44  | SideQuest                              | 2026 | arxiv 2602.22603 (NVIDIA)          | 近期发表,引用待累积 | model-driven KV 管理、agentic 长程推理、辅助任务并行压缩                                                      |
| 45  | Judge Q                                | 2025 | arxiv 2509.10798                   | 近期发表,引用待累积 | 可学习 query 优化 KV 淘汰信息保留、trained 路线                                                             |
| 46  | KV Cache as Reasoning Primitive        | 2025 | OpenReview id=vs2qwVfU2C           | 近期发表,引用待累积 | content-aware retention、premise 保留、长上下文逻辑一致性                                                  |
| 47  | ARKV                                   | 2026 | arxiv 2603.08727                   | 近期发表,引用待累积 | 受限预算下自适应资源高效 KV 管理                                                                            |
| 48  | SnapKV                                 | 2024 | NeurIPS 2024                       | 基础工作       | observation window 注意力聚合、prompt KV 选择、固定预算压缩                                                |
| 49  | StreamingLLM                           | 2023 | ICLR 2024                          | 基础工作       | attention sink、最近窗口、无限长度流式推理                                                                  |
| 50  | H2O                                    | 2023 | NeurIPS 2023                       | 基础工作       | heavy hitter token、累计注意力、动态 KV 淘汰                                                               |
| 51  | ShadowKV                               | 2024 | arxiv 2410.21465                   | 系统工作       | 低秩 key cache、value offloading、稀疏 KV 动态选择与重构                                                   |
| 52  | InfLLM-V2                              | 2025 | arxiv 2509.24663                   | 近期发表,引用待累积 | dense-sparse switchable attention、短长上下文自适应、可训练稀疏注意力                                      |
| 53  | Protection Is (Nearly) All You Need    | 2026 | arxiv 2605.18053                   | 近期发表,引用待累积 | 全局预算 KV 淘汰、结构边界保护、保护优先于打分                                                              |
| 54  | EpiKV                                  | 2026 | arxiv 2606.26472                   | 近期发表,引用待累积 | 跨层隐藏状态变化、attention-free token 打分、FlashAttention 兼容淘汰                                      |
| 55  | Beyond the 80/20 Rule                  | 2025 | NeurIPS 2025                       | 近期发表,引用待累积 | 高熵少数 token、RLVR 梯度筛选、reasoning forking token                                                    |
| 56  | SparDA                                 | 2026 | arxiv 2606.04511                   | 近期发表,引用待累积 | Forecast projection、跨层 KV 预取、稀疏注意力 offloading                                                  |
| 57  | Lethe                                  | 2025 | arxiv 2511.06029                   | 近期发表,引用待累积 | 层级预算分配、时序多轮剪枝、推理阶段动态 KV 管理                                                             |
| 58  | IndexCache                             | 2026 | arxiv 2603.12201                   | 近期发表,引用待累积 | 跨层 top-k index 复用、training-free/training-aware、稀疏选择加速                                         |
| 59  | FlashMemory-DeepSeek-V4                | 2026 | arxiv 2606.09079                   | 近期发表,引用待累积 | Lookahead Sparse Attention、Neural Memory Indexer、CSA KV 按需召回、CPU offload、1M context                     |
| 60  | FreeKV                                 | 2026 | ICLR 2026                          | 近期发表,引用待累积 | speculative retrieval、细粒度校正、CPU–GPU 混合布局、双缓冲流式召回、延迟隐藏                                      |
| 61  | LouisKV                                | 2026 | ICLR 2026                          | 近期发表,引用待累积 | 关键 KV 时间局部性、语义边界检索、输入/输出解耦细粒度管理、Triton/CUDA kernel                              |
| 62  | PM-KVQ                                 | 2026 | ICLR 2026                          | 近期发表,引用待累积 | 长 CoT KV 量化、渐进式混合精度、block-wise 内存分配、位置插值校准、累积误差                               |
| 63  | QuoKA                                  | 2026 | ICLR 2026                          | 近期发表,引用待累积 | chunked prefill、代表性 query 选择、query-oriented key 筛选、training-free、硬件无关                          |
| 64  | IceCache                               | 2026 | ICLR 2026                          | 近期发表,引用待累积 | 语义 token 聚类、PagedAttention、CPU–GPU KV offloading、分层动态索引、传输带宽利用                              |
| 65  | FusedKV                                | 2026 | ICLR 2026                          | 近期发表,引用待累积 | 跨层 KV 共享、底层/中层表示融合、post-RoPE key、FusedKV-Lite、可学重构                                   |

---

## 5 分钟泛读摘要

---

## [1] SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

**论文基本信息**
- 简称:SeerAttention-R
- 年份:2025 | 发表:arxiv 2506.08889(Microsoft,2025年6月) | 引用:近期发表,引用待累积
- 关键词:推理模型稀疏注意力、自蒸馏门控、长生成、TileLang、FlashAttention-3 加速
- 开源代码:https://github.com/microsoft/SeerAttention

**核心问题**
推理模型(DeepSeek-R1、QwQ 等)需要在 decoding 阶段生成数万 token 的长链思维,注意力计算成为主要瓶颈。原版 SeerAttention 设计针对 prefilling,使用 query pooling,无法适配自回归 decoding 场景。

**主要方法/贡献**
- **去除 query pooling:** 仅对 key 进行 pooling,保留逐 token 的 query,使门控机制兼容自回归生成。
- **轻量自蒸馏门控插件:** 仅训练门控参数(不修改原模型权重),通过自蒸馏让门控学会识别重要的注意力块;仅需 **0.4B token** 训练。
- **GQA 对齐共享稀疏:** 同一 GQA 组共享稀疏结构,与 GQA 内核高效融合。
- **TileLang 高效内核:** 针对大块稀疏(block size 64/128)设计的 GPU 内核,最大化 tensor core 利用率。

**关键结果**
- AIME 推理基准上以 4K token 注意力预算实现**近无损**精度。
- 在 H100 上 90% 稀疏度时相比 FlashAttention-3 实现 **9× 加速**(接近理论上限)。
- 适用于 DeepSeek-R1、QwQ 等多种推理模型。

**创新点**
- **首个面向长推理 decoding 阶段**的稀疏注意力适配方案,填补了 NSA、MInference 等 prefilling 工作的空白。
- 大块稀疏(64/128)的设计与 TileLang 内核协同,把稀疏注意力的实际加速比真正逼近理论极限。

**阅读价值判断**
- 当推理模型成为主流时,decoding 阶段的注意力优化是新的关键瓶颈。本工作是该方向的代表性早期成果与**事实上的锚点工作**,是后续 RaaS、R-KV、ThinKV 等研究的对照基线。必读。

---

## [2] RaaS: Reasoning-Aware Attention Sparsity for Efficient LLM Reasoning

**论文基本信息**
- 简称:RaaS
- 年份:2025 | 发表:arxiv 2502.11147(2025年2月) | 引用:近期发表,引用待累积
- 关键词:推理感知稀疏、milestone token、O(L) 时间内存、CoT decoding
- 开源代码:公开

**核心问题**
现有稀疏注意力(Quest、H2O 等)在长 CoT 推理任务上面临"准确率-时间-内存"的不可能三角:激进稀疏导致精度暴跌,保守稀疏内存仍随长度线性增长。需要专门面向推理任务的注意力分析。

**主要方法/贡献**
- **Milestone token 模式发现:** 系统刻画了 CoT decoding 中的注意力模式——某些 token 作为推理"中间引理"在被使用前持续高注意力,使用后注意力骤降。
- **基于 page 的时间戳打分:** 跟踪每个 KV page 最后一次被高注意力访问的时间戳,过期则淘汰。
- **同时实现 O(L) 时间和 O(L) 内存:** 比 Quest(O(L) 时间但 O(L) 内存)与 H2O(常数内存但精度差)在推理任务上同时占优。

**关键结果**
- 在 AIME、MATH 等推理基准上,以远低于 Quest 的内存达到接近无损精度。
- 长链思维(8K+ output)场景下,内存峰值显著降低。

**创新点**
- 首篇系统形式化"推理任务 decoding 注意力模式"的工作,提出 milestone token 概念,为后续 R-KV、ThinKV 等推理特化方法提供了经验基础。

**阅读价值判断**
- 与 SeerAttention-R 共同构成"推理 decoding 稀疏"的两个起点,前者偏系统-内核,后者偏模式-算法。本工作的注意力模式分析对自己的方向 1 有直接启发。必读。

---

## [3] R-KV: Redundancy-aware KV Cache Compression for Reasoning Models

**论文基本信息**
- 简称:R-KV
- 年份:2025 | 发表:NeurIPS 2025(arxiv 2505.24133) | 引用:近期发表,引用待累积
- 关键词:推理模型 KV 缓存、冗余感知、reflection 压缩、长 CoT
- 开源代码:https://github.com/Zefan-Cai/R-KV

**核心问题**
推理模型的 long CoT 中存在大量冗余:模型反复 reflection、回溯、自我验证,导致 KV 缓存中存在冗余的语义重复 token。仅基于注意力分数的淘汰会保留这些重复内容,效率较差。

**主要方法/贡献**
- **联合重要性 + 冗余打分:** 实时基于 key 向量相似度检测语义重复 token,与注意力重要性联合决策。
- **滚动压缩:** 在 decoding 中持续压缩,而非一次性 prefill 后压缩。
- **保留多样化 milestone:** 优先保留语义独立的关键 token,丢弃 reflection 中的语义复制。

**关键结果**
- **保留 10% KV 即可达 100% 完整性能;16% KV 反超基线 105%**(冗余去除带来积极效果)。
- 90% 内存节省,**6.6× 吞吐**提升于 R1-Distill-Llama-8B/Qwen-14B。

**创新点**
- 首次将"语义冗余检测"显式纳入推理模型 KV 压缩;发现 reflection 是推理 KV 冗余的主要来源。
- "压缩反而提升精度"的反直觉结果,指向推理模型的 KV 中存在干扰性冗余。

**阅读价值判断**
- NeurIPS 2025。是推理模型 KV 压缩三大代表作(SeerAttention-R、RaaS、R-KV)之一,代码完整。强烈建议精读其冗余分析部分。

---

## [4] ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models

**论文基本信息**
- 简称:ThinKV
- 年份:2026 | 发表:ICLR 2026 Oral(arxiv 2510.01290,NVIDIA) | 引用:近期发表,引用待累积
- 关键词:thought-adaptive KV、混合量化-淘汰、思维类型分类、PagedAttention
- 开源代码:公开

**核心问题**
推理模型的 CoT 包含异质性的"思维类型"——分析、规划、计算、验证、反思等步骤的注意力稀疏度差异巨大,统一压缩策略无法适配这种异质性。

**主要方法/贡献**
- **思维类型自动分类:** 通过注意力稀疏特征自动将 CoT 分解为不同 thought 类型,识别其重要性等级。
- **混合量化-淘汰:** 高重要性思维使用高精度量化,低重要性思维渐进式淘汰。
- **PagedAttention 内核扩展:** 支持淘汰槽位的复用,避免内存碎片。

**关键结果**
- **保留不到 5% 原始 KV** 仍接近无损,在 R1-Distill、GPT-OSS、AceReason 上验证。
- 端到端 decoding 加速明显。

**创新点**
- 首个将 CoT 在"思维类型"粒度上做异质压缩的工作,比"token 级 milestone"(RaaS)粒度更高,比"全局头分类"(DuoAttention)更动态。
- 引入了"思维语义结构 → 注意力策略"的新映射维度。

**阅读价值判断**
- NVIDIA 出品,工程完整度高。对 CoT 内部结构感兴趣的研究者值得精读;其"thought-level"分析对自己提出的"步骤定位"洞察直接相关。

---

## [5] LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Long Reasoning

**论文基本信息**
- 简称:LazyEviction
- 年份:2025 | 发表:arxiv 2506.15969(2025年6月) | 引用:近期发表,引用待累积
- 关键词:滞后 KV 淘汰、Token Importance Recurrence、长推理
- 开源代码:公开

**核心问题**
推理模型的注意力模式中,**token 重要性会复发**——某 token 在被冷落数千步后会再次被高度关注(典型场景:模型回头检索早期推理步骤)。即时淘汰会错杀这些"复发型"token。

**主要方法/贡献**
- **Token Importance Recurrence(TIR)现象发现:** 系统量化推理模型中 token 重要性的"复发"行为,定义最大复发间隔(Maximum Recurrence Interval, MRI)。
- **滞后淘汰算法:** 设置观察窗口,token 在窗口内未被复用才淘汰,容忍其复发可能。
- **MRI 自适应窗口:** 根据每个 token 的历史复发间隔自适应调整观察窗口大小。

**关键结果**
- 长推理任务上 **50%–70% KV 降低**且精度无损。
- 显著优于即时淘汰策略(H2O 等)。

**创新点**
- 首次系统刻画推理模型注意力的"复发性"现象,与 RaaS 的 milestone 形成互补:milestone 关注"何时重要",TIR 关注"重要性如何重新出现"。

**阅读价值判断**
- 是与自己方向 1 中"回头检索"洞察最契合的工作之一,必读。其 MRI 量化方法可直接借鉴。

---

## [6] LessIsMore: Training-Free Sparse Attention with Global Locality for Efficient Reasoning

**论文基本信息**
- 简称:LessIsMore
- 年份:2025 | 发表:arxiv 2508.07101(Princeton+CMU,2025年8月) | 引用:近期发表,引用待累积
- 关键词:训练免训稀疏、跨头共享 token 选择、推理任务
- 开源代码:https://github.com/DerrickYLJ/LessIsMore

**核心问题**
现有稀疏注意力在每个头独立选择 token,带来索引开销大、跨头不一致等问题。但实证显示**推理模型中关键 token 在多头之间高度共享**,且在 decoding 步骤间稳定。

**主要方法/贡献**
- **跨头共享 token 选择:** 一次计算选出全局重要 token,所有头共享(节省冗余索引计算)。
- **decoding 步骤间稳定性利用:** 选择的 token 集合在相邻步骤间高度相似,可缓存复用。
- **训练免训:** 直接应用于 R1-Distill 等推理模型。

**关键结果**
- 同等精度下注意力 token 量减少 **2×**,端到端速度提升 **1.13×**。
- 在 AIME、MATH 上一致优于 Quest、H2O 等基线。

**创新点**
- 首次定量证明推理模型中"关键 token 跨头一致性"现象,简化了稀疏注意力的算法复杂度。
- "Less is More"标题表达了"减少头独立性带来的去冗余收益"。

**阅读价值判断**
- 训练免训路线下的代表作之一。建议结合 LessIsMore 与 SeerAttention-R 对比"训练 vs 免训"的性能极限。

---

## [7] DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning

**论文基本信息**
- 简称:DELTA
- 年份:2025 | 发表:arxiv 2510.09883(2025年10月) | 引用:近期发表,引用待累积
- 关键词:层感知 token 注意力、Δ-selection、长上下文推理
- 开源代码:公开

**核心问题**
不同 transformer 层的注意力稀疏化收益差异巨大——前几层涉及精确语义匹配,激进稀疏会损失关键信息;深层注意力较为均匀,稀疏化收益高。固定层稀疏率无法利用这种异质性。

**主要方法/贡献**
- **层级分区:** 将层分为"完整注意力层 + Δ-selection 层 + 后续稀疏层"三类。
- **Δ-selection 层动态选择:** 在 Δ 层计算 high-recall token 子集,后续稀疏层复用此选择。
- **跨层选择复用:** 大幅降低重复 top-K 选择的开销。

**关键结果**
- 在 AIME、GPQA-Diamond 上匹配/超过完整注意力,token 量减少 **5×**,端到端 **1.5×** 加速。

**创新点**
- 系统化"层级稀疏策略",而非全模型统一稀疏率。
- 把 token 选择视为一次性"投资",在多层间复用,减少索引计算开销。

**阅读价值判断**
- 与 LessIsMore 的"跨头共享"互补——前者跨头共享,后者跨层共享,共同构成"减少冗余选择"的两条优化路径。值得关注。

---

## [8] RLKV: Which Heads Matter for Reasoning? RL-Guided KV Cache Compression

**论文基本信息**
- 简称:RLKV
- 年份:2025 | 发表:arxiv 2510.08525(2025年10月) | 引用:近期发表,引用待累积
- 关键词:RL 引导 KV 压缩、推理关键头识别、头功能分解
- 开源代码:公开

**核心问题**
不同注意力头对推理质量的贡献差异巨大,但传统的注意力分数或检索任务打分(DuoAttention)与"推理是否正确"并非直接对齐。需要直接以"推理结果质量"为优化目标识别推理关键头。

**主要方法/贡献**
- **RL 引导头重要性学习:** 以推理任务最终生成质量为奖励信号,通过 RL 优化每个头的 KV 缓存预算分配。
- **关键头全 KV、其余头激进压缩:** 找到的关键头保留完整 KV,其他头执行强压缩。
- **直接面向推理结果优化:** 不依赖代理目标(如 NIAH)。

**关键结果**
- **20%–50% 缓存降低**且推理精度无损,加速最高 1.21×。
- 发现的"推理关键头"与 DuoAttention 的"检索头"部分重叠但有显著差异。

**创新点**
- 首次以 RL+生成质量直接优化 KV 预算,优于基于注意力代理打分的方法。
- 揭示了"检索头 ≠ 推理头",为头功能分类提供了新的细分维度。

**阅读价值判断**
- 对自己方向 A(头功能图谱 × KV 差异化)有直接借鉴价值。其"推理头"概念是对 DuoAttention 二分的细化,可作为细粒度头分类的一类。强烈推荐。

---

## [9] Lil: Less is Less When Applying Post-Training Sparse-Attention in Long-Decode

**论文基本信息**
- 简称:Lil
- 年份:2025/2026 | 发表:arxiv 2601.03043 | 引用:近期发表,引用待累积
- 关键词:长 decode 信息损失分析、early-stopping、稀疏负效应
- 开源代码:公开

**核心问题**
反直觉地发现:稀疏注意力在长 decoding 中**可能反而增加端到端成本**——稀疏带来的信息损失会导致模型生成更长、更冗余的输出,抵消稀疏化的算力节省。

**主要方法/贡献**
- **"信息损失-生成长度"分析:** 量化稀疏率与生成长度膨胀的关系。
- **信息损失/收益交叉点检测:** 在 decoding 过程中实时监测,在收益由正转负时早停。
- **生成长度感知的稀疏调度:** 根据当前生成进度动态调整稀疏率。

**关键结果**
- 在保留 90% token 削减的前提下精度损失 <2%,但若不加 early-stopping,激进稀疏反而**总耗时上升 30%**。

**创新点**
- 首篇系统揭示"稀疏注意力的副作用"——生成长度膨胀,这一现象在 prefilling 优化中不存在。
- 是对"激进稀疏越好"思维定势的重要纠偏。

**阅读价值判断**
- **必读**警示性论文。任何做推理 decoding 稀疏的工作都需考虑此现象,自己在做实验时也需注意端到端耗时,不能仅看注意力 FLOPs。Hold Onto That Thought 也佐证了类似现象。

---

## [10] Retrospective Sparse Attention (RetroAttention)

**论文基本信息**
- 简称:RetroAttention
- 年份:2025 | 发表:arxiv 2508.09001(2025年8月) | 引用:近期发表,引用待累积
- 关键词:回溯注意力修正、累积误差、长生成精度
- 开源代码:公开

**核心问题**
长 decoding 中,稀疏注意力造成的误差**逐步累积**——早期 token 的稀疏选择即使局部最优,也会因后续 KV 加入而变得次优。需要回溯修正机制。

**主要方法/贡献**
- **轻量输出缓存:** 缓存历史 attention output,而非仅 KV。
- **后置修正:** 当新 KV 进入时,回溯重算受影响的历史输出。
- **修正粒度自适应:** 根据 KV 影响范围动态决定回溯深度。

**关键结果**
- 长生成基准上有效 KV 暴露提升 **1.6×**,精度提升 **+21.9%**。

**创新点**
- 首次提出"稀疏选择是动态最优问题",而非每步独立的瞬时决策。
- 回溯机制为长 CoT 中的"自我验证-回退"步骤提供了原生支持。

**阅读价值判断**
- 对长 CoT 中"思考-验证-回退"映射到注意力的探索,具有方法层面的启发性。其修正机制可与方向 1 的"回头检索"分析结合。

---

## [11] SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding

**论文基本信息**
- 简称:SpecAttn
- 年份:2025 | 发表:arxiv 2510.27641 | 引用:近期发表,引用待累积
- 关键词:稀疏-投机解码协同、verification 阶段免费 oracle、AIME
- 开源代码:公开

**核心问题**
投机解码中 verification 阶段实际上**免费**完成了完整注意力计算——这部分信息可作为 oracle 指导 drafting 阶段的稀疏 token 选择,但传统方法未利用此协同。

**主要方法/贡献**
- **Verification 即 oracle:** 利用 verification 阶段的完整注意力分数指导后续 drafting 的稀疏选择。
- **drafting 阶段稀疏化:** 仅基于 oracle 选出的关键 token 进行 draft 计算,大幅加速。
- **AIME25、CodeElo 等推理基准评测**

**关键结果**
- 相对自回归基线 **2.81× 加速**,相对 SOTA 稀疏-投机方法 **1.29× 加速**。

**创新点**
- 首次系统将稀疏注意力与投机解码协同设计,把 verification 计算"二次利用"。
- 是对推理 decoding 加速的"算法栈级"优化。

**阅读价值判断**
- 投机解码方向研究者必读;若自己方向 1 后续要扩展到投机解码场景,可作为对照。

---

## [12] NSA: Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**论文基本信息**
- 简称:NSA
- 年份:2025 | 发表:ACL 2025(Best Paper,DeepSeek) | 引用:~150
- 关键词:原生稀疏注意力、训练时稀疏、可训练注意力结构、decoding 加速 11.6×
- 开源代码:公开

**核心问题**
现有稀疏注意力方法大多是后处理式的(post-hoc),即在全注意力预训练模型上施加稀疏化,无法保证稀疏结构与模型能力的深度匹配。从头训练稀疏注意力模型又面临硬件不友好、梯度无法回传等挑战。

**主要方法/贡献**
- **三类原生稀疏模式:** 设计三种互补的稀疏注意力组件:(1)**Token Compression**(局部内容压缩为粗粒度 token);(2)**Token Selection**(全局动态选择最重要 token);(3)**Sliding Window**(保留局部精度),三类模式联合计算注意力。
- **端到端可训练:** 整个稀疏注意力结构可在预训练中直接优化,稀疏模式随任务自然形成。
- **硬件高效内核:** 针对 GPU tensor core 设计专用计算路径,保证稀疏计算的实际效率。

**关键结果**
- 在长上下文基准(64k+ token)上,NSA 与全注意力性能相当甚至更优。
- 训练加速:前向 9×、反向 6×;**decoding 加速 11.6×** at 64k 上下文。

**创新点**
- 首次提出将稀疏注意力作为**训练时的一等公民**(而非推理时的后处理),实现稀疏模式与任务分布的深度对齐。
- DeepSeek 工业落地验证,可信度高。

**阅读价值判断**
- DeepSeek 重要技术报告,**ACL 2025 最佳论文**。其 decoding 加速章节明确指向 long generation 场景,与本方向直接相关。是理解"稀疏注意力训练范式转变"的必读工作。

---

## [13] MoBA: Mixture of Block Attention for Long-Context LLMs

**论文基本信息**
- 简称:MoBA
- 年份:2025 | 发表:arxiv 2502.13189(Kimi/MoonshotAI 技术报告) | 引用:~200
- 关键词:块级稀疏注意力、MoE 式注意力、长上下文、可切换全/稀疏注意力
- 开源代码:https://github.com/MoonshotAI/MoBA

**核心问题**
扩展 LLM 上下文长度时,自注意力的二次复杂度成为算力瓶颈。现有稀疏注意力方法大多依赖**预定义的注意力偏置模式**(sink、window 等),这类强先验在不同任务上可能不适配。

**主要方法/贡献**
- **MoE 化注意力机制:** 将 KV 序列切分为多个块,借鉴 MoE 路由思想,为每个 query token 动态选择 Top-K 个最相关的 KV 块进行注意力计算。
- **全/稀疏可切换:** 同一组参数可在全注意力与块级稀疏注意力之间无缝切换,训练时混合使用,推理时按需选择。
- **无强先验设计:** 让模型自行学习路由策略,更具任务通用性。

**关键结果**
- 在 1M token 上下文下相比全注意力实现 **6.5×** 加速;在 10M token 时 **16×**。
- 已部署在 Kimi 线上服务长上下文请求中,工业级验证。

**创新点**
- 首次将 MoE 路由范式系统应用到注意力**块选择**层面,且支持全/稀疏的训练-推理切换。
- "无预定义偏置"的设计哲学是对 StreamingLLM、A-shape 等强先验路线的反思。

**阅读价值判断**
- Kimi 线上系统的核心长上下文模块,与 NSA 并列 2025 年训练时稀疏注意力的两大代表作。其 decoding 加速场景明确,与本方向直接相关。

---

## [14] DeepSeek Sparse Attention (DSA, V3.2-Exp)

**论文基本信息**
- 简称:DSA / DeepSeek-V3.2
- 年份:2025 | 发表:arxiv 2512.02556(DeepSeek,V3.2-Exp 2025年9月) | 引用:近期发表,引用待累积
- 关键词:Lightning indexer、token selector、MLA 集成、生产部署
- 开源代码:DeepSeek-V3.2-Exp 公开

**核心问题**
DeepSeek-V3 的 MLA 已实现 KV 压缩,但完整注意力的 O(L²) 复杂度仍是 128K+ 上下文的瓶颈。如何将稀疏注意力与 MLA 协同,既保留 MLA 的 KV 压缩,又获得稀疏的算力节省?

**主要方法/贡献**
- **Lightning Indexer(FP8 实现):** 极轻量的 indexer 估计每个 query 与 KV 的相关性,FP8 计算确保高吞吐。
- **Token Selector:** 基于 indexer 输出选择 top-K KV token 计算注意力,O(L²)→O(Lk)。
- **MLA + DSA 集成:** 在 MLA 的低维潜变量空间上做稀疏选择,双重压缩协同。

**关键结果**
- 128K 上下文下质量损失可忽略。
- DeepSeek-V3.2-Exp 实际部署验证,生产级。

**创新点**
- 首个工业级"低秩(MLA)+稀疏(DSA)"双重压缩的开源大模型实现,验证了两种压缩机制的正交性。
- Lightning indexer 的 FP8 设计为稀疏选择的运行时开销提供了实用解。

**阅读价值判断**
- DeepSeek 系列工业落地的注意力优化集大成者。其"低秩+稀疏"乘性收益对自己方向 B(MHA2MLA × 稀疏)有直接对照价值。**强烈推荐**。

---

## [15] MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

**论文基本信息**
- 简称:MiniMax-M1
- 年份:2025 | 发表:arxiv 2506.13585(MiniMax,2025年6月) | 引用:近期发表,引用待累积
- 关键词:Lightning Attention、混合注意力、MoE、长 CoT 推理、CISPO 强化学习
- 开源代码:https://github.com/MiniMax-AI/MiniMax-M1

**核心问题**
推理模型需要大量 test-time 思考 token,但 softmax 注意力在长生成下计算成本剧增,限制了推理深度。如何在 1M token 上下文 + 长链思维场景下显著降低 FLOPs?

**主要方法/贡献**
- **Lightning Attention 混合架构:** 每 7 个 transnormer 块(lightning linear attention)后插入 1 个 softmax attention 块,**7:1 比例**混合堆叠。
- **456B 总参数 / 45.9B 激活的 Hybrid-MoE:** 首个开源大规模混合注意力推理模型,原生支持 1M token 上下文。
- **CISPO 强化学习算法:** 通过裁剪 importance sampling 权重提升长 RL 训练稳定性。

**关键结果**
- 生成 100K token 时仅消耗 DeepSeek-R1 约 **25% 的 FLOPs**。
- 上下文窗口 1M token,是 DeepSeek-R1 的 **8×**。

**创新点**
- 首个开源的大规模 hybrid-attention 推理模型,证明 lightning linear attention 在百亿级 MoE 推理任务上的可行性。
- 7:1 极端线性化比例,与 Kimi Linear/Qwen3-Next 的 3:1 形成对比。

**阅读价值判断**
- 对推理模型架构设计、长 CoT test-time scaling 感兴趣的研究者必读。是 hybrid attention + 推理模型方向的关键样本。

---

## [16] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

**论文基本信息**
- 简称:Quest
- 年份:2024 | 发表:ICML 2024(MIT-Han Lab) | 引用:~300
- 关键词:query-aware KV 选择、page 级稀疏、decoding 加速
- 开源代码:https://github.com/mit-han-lab/Quest

**核心问题**
长上下文 decoding 中,完整注意力对每个 query 计算所有 KV 的 dot product,但实际上每步只有少量 KV 有显著贡献。如何在 decoding 时高效识别"关键 KV"?

**主要方法/贡献**
- **每个 KV page 的 min/max key:** 保存 page 内 key 的极值统计。
- **Query-aware top-K page 选择:** 计算 query 与 page min/max 的内积上界,选 top-K page 进行实际注意力。
- **Decoding 阶段稀疏:** 直接面向 decoding,而非 prefilling。

**关键结果**
- 自注意力加速 **7.03×**,端到端 **2.23×**。
- 在 long-bench 上精度无损。

**创新点**
- 是 decoding 阶段 KV 选择的奠基工作,后续 RaaS、R-KV、SeerAttention-R 等均以 Quest 为基线。
- Page-level 抽象兼顾了选择粒度与 GPU 友好性。

**阅读价值判断**
- decoding 稀疏注意力的**事实标准基线**,几乎所有后续工作都以其为对照。必读。

---

## [17] DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

**论文基本信息**
- 简称:DuoAttention
- 年份:2024/2025 | 发表:ICLR 2025(arxiv 2410.10819,MIT-Han Lab) | 引用:~150
- 关键词:检索头、流式头、KV 缓存差异化、长上下文推理
- 开源代码:https://github.com/mit-han-lab/duo-attention

**核心问题**
长上下文 LLM 推理中 KV 缓存内存压力极大。但深入分析发现**只有部分注意力头真正需要全长 KV 历史**("检索头"),其余头其实只关注局部窗口("流式头"),过去的方法对所有头一视同仁地保留全 KV,造成大量浪费。

**主要方法/贡献**
- **检索头 vs 流式头分类:** 提出基于轻量优化的算法,使用合成数据自动识别每个注意力头的类别。
- **差异化 KV 缓存:** 检索头维持完整 KV cache,流式头使用固定长度的循环缓存。
- **量化协同:** 与 8-bit 权重 + 4-bit KV 量化结合后,单张 A100-80G 即可承载 **3.3M token** 的 Llama-3-8B 推理。

**关键结果**
- MHA 模型 KV 内存降低 **2.55×**,GQA 模型降低 **1.67×**。
- 解码延迟降低 **2.18×**(MHA)/ **1.50×**(GQA)。

**创新点**
- 首次系统提出注意力头的**功能性二分**(检索 vs 流式),并给出可自动识别的算法。
- "异构 KV 缓存"设计为后续头粒度优化方法(Ada-KV、HeadKV、RLKV)提供了重要先例。

**阅读价值判断**
- ICLR 2025(Song Han 团队)。是当前 KV 优化方向最具影响力的工作之一,对长上下文推理系统设计、注意力头可解释性研究均有参考价值。**头功能分类方向的奠基性工作**,必读。

---

## [18] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation

**论文基本信息**
- 简称:Ada-KV
- 年份:2025 | 发表:NeurIPS 2025 | 引用:~80
- 关键词:KV 缓存淘汰、自适应预算分配、逐头优化、即插即用
- 开源代码:https://github.com/FFY0/AdaKV

**核心问题**
现有 KV 缓存淘汰方法(H2O、SnapKV 等)通常采用**全局统一预算**,对所有注意力头分配相同的保留 token 数。但实际上不同头的注意力分布形态差异巨大,统一预算会同时造成"重要头过度淘汰"与"冗余头预算浪费"。

**主要方法/贡献**
- **首个 head-wise 自适应预算分配:** 在固定总缓存预算约束下,根据每个头的注意力集中度动态分配 token 保留数。
- **理论损失界:** 给出淘汰前后注意力输出差异的理论上界。
- **即插即用:** 可叠加于 H2O、SnapKV、Pyramidal-KV 等现有淘汰算法之上。

**关键结果**
- 在 Ruler 和 LongBench 共 **29 个数据集**上一致优于均匀预算基线。
- 与多种现有方法组合后均带来精度提升。

**创新点**
- 第一次将"逐头自适应预算"系统化为 KV 缓存淘汰的标准设计原则。
- 理论 + 实证的双重论证,为后续 KV 优化方法树立了规范基线。

**阅读价值判断**
- NeurIPS 2025。对长上下文 LLM 推理优化、KV 缓存系统设计有直接工程价值;与 DuoAttention、HeadKV 共同覆盖头粒度 KV 优化的三条主线。

---

## [19] Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

**论文基本信息**
- 简称:Gated Attention(Qwen)
- 年份:2025 | 发表:NeurIPS 2025(Best Paper,Oral) | 引用:~500
- 关键词:注意力门控、稀疏性、非线性、无注意力汇聚
- 开源代码:https://github.com/qiuzh20/gated_attention

**核心问题**
标准 SDPA 存在"注意力汇聚"现象,少数 token 持续吸引大量注意力,导致训练不稳定、长上下文外推性能差。

**主要方法/贡献**
- **Head-specific Sigmoid Gate:** SDPA 输出后施加逐头 sigmoid 门控。
- **大规模系统实验:** 30+ 变体,15B MoE 与 1.7B 稠密模型,3.5T token 训练对比。
- **多维效益:** 非线性 + 自然稀疏 + 消除 attention sink + 长上下文外推 + 训练稳定性。

**关键结果**
- 稳定提升所有规模与任务性能;长上下文外推显著改善;融入 Qwen3-Next。

**创新点**
- 首次行业尺度系统比较门控注意力变体,揭示"简单逐头 sigmoid 门控"的多维收益。

**阅读价值判断**
- NeurIPS 2025 最佳论文。其"门控诱导稀疏 + 消除 sink"对方向 1 中"reasoning sink"分析直接相关——可对照分析推理模型是否仍存在 sink 现象。必读。

---

## [20] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

**论文基本信息**
- 简称:Twilight
- 年份:2025 | 发表:NeurIPS 2025(Spotlight) | 引用:~50
- 关键词:自适应稀疏注意力、Top-p 剪枝、层次化预算决策、可组合优化
- 开源代码:https://github.com/tsinghua-ideal/Twilight

**核心问题**
现有稀疏方法采用**固定 Top-k 预算**,无法适应注意力分布的高度动态性。

**主要方法/贡献**
- **Top-p 稀疏注意力:** 借鉴 nucleus sampling,动态积累注意力权重直至阈值 p。
- **层次化剪枝:** 两级决策结构,块级粗预算 + 块内细 Top-p 选择。
- **可组合优化器:** 可叠加于 MInference、FlexPrefill 等之上。

**关键结果**
- 中长上下文场景下自适应剪枝高达 **98% tokens**,几乎无精度损失。
- A100 上相比全注意力 **1.4× 推理加速**;长上下文 decoding 上注意力 **15.4× / 端到端 3.9×** 加速。

**创新点**
- "Top-p 稀疏"跨领域迁移自 nucleus sampling,创意新颖。
- 可组合性使其成为稀疏注意力的通用加速模块。

**阅读价值判断**
- NeurIPS 2025 Spotlight。其 Top-p 思路可与方向 1 中"α-entmax 函数稀疏"联动,具有方法层面的启发性。

---

## [21] Retrieval Head Mechanistically Explains Long-Context Factuality

**论文基本信息**
- 简称:Retrieval Head
- 年份:2025 | 发表:ICLR 2025(arxiv 2404.15574) | 引用:~200
- 关键词:检索头机制、长上下文事实性、头功能解释
- 开源代码:公开

**核心问题**
长上下文 LLM 中,模型如何在数十万 token 中精确检索特定信息?哪些头在承担此功能?

**主要方法/贡献**
- **检索头识别:** 系统识别承担"信息复制/检索"功能的头(<5% 的头)。
- **机制性消融:** 移除 20 个检索头使 NIAH 准确率从 **94.7% → 63.6%**。
- **模型/任务通用性:** 检索头在不同模型中位置稳定,跨任务共享。

**关键结果**
- <5% 的头主导长上下文检索能力。

**创新点**
- 首次从机制可解释性角度定量证明"检索头"的存在与作用。
- 是 DuoAttention、HeadKV、RLKV 等头功能分类工作的理论基础。

**阅读价值判断**
- 理解头功能分类的**理论基石**。任何研究头粒度 KV 优化的工作都需引用此文。必读。

---

## [22] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression

**论文基本信息**
- 简称:AttentionPredictor
- 年份:2025 | 发表:NeurIPS 2025(arxiv 2502.04077) | 引用:近期发表,引用待累积
- 关键词:时空注意力预测、卷积建模、KV 压缩 13×、跨 token 预取
- 开源代码:公开

**核心问题**
现有 KV 压缩仅根据当前注意力分数决策,忽略了**注意力分数的时空规律性**——分数在序列时间和空间上都有可预测模式。

**主要方法/贡献**
- **轻量卷积模型预测下一步注意力:** 捕捉时空规律,预测哪些 KV 即将被关注。
- **跨 token 预取:** 预测下一步重要 KV 提前加载,隐藏 decoding 中的稀疏选择延迟。
- **统一压缩 + 预取框架**

**关键结果**
- KV 压缩 **13×**,decoding 加速 **5.6×**。

**创新点**
- 首次将"注意力时空规律建模"显式纳入 KV 压缩。
- 预取机制有效隐藏稀疏选择的运行时开销。

**阅读价值判断**
- 是少数显式建模"注意力时序"的工作,与 LazyEviction 的 TIR 形成方法互补。值得精读。

---

## [23] RocketKV: Two-Stage KV Cache Compression for Efficient Decoding

**论文基本信息**
- 简称:RocketKV
- 年份:2025 | 发表:ICML 2025(NVIDIA) | 引用:近期发表,引用待累积
- 关键词:两阶段 KV 压缩、SnapKV+top-K 混合、decode 加速 3.7×
- 开源代码:公开

**核心问题**
单一阶段的 KV 压缩在极端高压缩比下精度急剧下降。

**主要方法/贡献**
- **两阶段:** Stage 1 SnapKV 风格粗淘汰(prefill 末);Stage 2 decoding 时 top-K 混合稀疏。
- **极高压缩比:** 高达 **400× 压缩**仍保持精度。
- **A100 部署:** decode 端到端加速 **3.7×**,内存降低 32.6%。

**关键结果**
- 极致压缩比下精度损失可控,工程友好。

**创新点**
- 阶段化压缩思路,粗筛 + 精筛分工。

**阅读价值判断**
- ICML 2025。工程实用性强,作为 KV 压缩极限的参考方法。

---

## [24] Inference-Time Hyper-Scaling with KV Cache Compression (DMS)

**论文基本信息**
- 简称:DMS
- 年份:2025 | 发表:NeurIPS 2025(NVIDIA + Edinburgh) | 引用:近期发表,引用待累积
- 关键词:TTS+KV 压缩、淘汰感知训练、推理预算放大
- 开源代码:公开

**核心问题**
推理 test-time scaling(TTS)需要更多 token 思考,但单 token 成本(注意力 FLOPs)的下降可在同等算力预算下生成更多 token,从而提升推理质量——形成"压缩 → 更多 token → 更高精度"的正反馈。

**主要方法/贡献**
- **TTS + KV 压缩协同:** 把节省的算力反哺为更多生成 token。
- **淘汰感知训练:** 仅 1K 训练步即可训练出 8× 压缩的 KV 模型。
- **推理预算放大:** 同等 FLOPs 预算下精度全面提升。

**关键结果**
- **8× KV 压缩**,推理预算下精度上升。

**创新点**
- 首次系统打通"KV 压缩"与"test-time scaling"两条优化路径,提出"压缩为了思考更多"的新视角。

**阅读价值判断**
- 是连接 TTS 与稀疏注意力的关键工作。对自己方向 1 有重要启发——稀疏不应只看精度损失,应看"同等算力下整体推理表现"。强烈推荐。

---

## [25] HeadKV: Not All Heads Matter — Head-Level KV Cache Compression

**论文基本信息**
- 简称:HeadKV
- 年份:2025 | 发表:ICLR 2025(arxiv 2410.19258) | 引用:~100
- 关键词:头级 KV 压缩、检索+推理打分、头功能选择
- 开源代码:公开

**核心问题**
DuoAttention 的检索/流式二分仍然粗糙,实际上不同头在"检索 vs 推理"上的能力分布连续。

**主要方法/贡献**
- **检索 + 推理双打分:** 同时为每个头评估其在检索能力(NIAH 类)与推理能力(reasoning 类任务)上的贡献。
- **头级 KV 预算分配:** 综合双分数为每个头分配 KV 预算。
- **跨任务一致性验证:** 验证打分稳定性。

**关键结果**
- 在长上下文与推理混合任务上,优于 DuoAttention、Ada-KV 等单维度方法。

**创新点**
- 把"头功能"从二分细化到双维度连续打分,为头分类提供新维度。

**阅读价值判断**
- 是方向 A(头功能图谱 × KV 差异化)的直接前置工作,**与 RLKV 共同构成"头功能再细分"的两条路径**。必读。

---

## [26] Expected Attention: Closed-Form KV Importance Estimation

**论文基本信息**
- 简称:Expected Attention
- 年份:2025 | 发表:arxiv 2510.00636(NVIDIA,ICLR 2026 在审) | 引用:近期发表,引用待累积
- 关键词:期望注意力闭式估计、未来 query 分布、训练免训
- 开源代码:NVIDIA kvpress

**核心问题**
现有 KV 重要性评分依赖当前已观察的注意力分数,无法预测未来 query 的需求。

**主要方法/贡献**
- **闭式期望注意力:** 假设未来 query 服从某分布,推导出每个 KV 在未来注意力中的期望分数闭式解。
- **训练免训:** 直接应用于已有模型,无需训练。
- **集成于 NVIDIA kvpress 库**

**关键结果**
- 在多项长上下文 benchmark 上优于基于历史注意力的 KV 选择方法。

**创新点**
- 把 KV 重要性估计从"事后观察"提升到"事前期望",理论上更优。

**阅读价值判断**
- ICLR 2026 在审。理论上优雅,对自己方向 1 中"理论指导的稀疏 mask"提供了一个新视角(期望注意力 → 稀疏 mask)。值得深读。

---

## [27] Hold Onto That Thought: Assessing KV Cache Compression on Reasoning

**论文基本信息**
- 简称:Hold Onto That Thought
- 年份:2025 | 发表:NeurIPS 2025 ER Workshop(arxiv 2512.12008) | 引用:近期发表,引用待累积
- 关键词:KV 压缩在推理任务实证研究、SnapKV vs H2O、副作用揭示
- 开源代码:公开

**核心问题**
现有 KV 压缩方法在推理模型上的系统对照——多数方法在通用 benchmark 上设计,推理任务有特殊性。

**主要方法/贡献**
- **跨方法系统评测:** 对比 H2O、SnapKV、StreamingLLM、Quest 等在 R1 蒸馏模型上的表现。
- **decoding-enabled SnapKV 变体:** 揭示 SnapKV 在 decoding 场景的潜力。
- **副作用揭示:** 低预算淘汰反而**导致更长推理链**(与 Lil 论文呼应),需端到端评估。

**关键结果**
- H2O 与 decoding-SnapKV 在 R1 distillations 上整体最优;但激进压缩有"延长生成"副作用。

**创新点**
- 首篇推理任务专用 KV 压缩 benchmark 研究。

**阅读价值判断**
- 实证基准提供方,对自己实验设计有直接参考价值。配合 Lil 一起阅读。

---

## [28] Hardware-Efficient Attention for Fast Decoding (GTA / GLA)

**论文基本信息**
- 简称:GTA / GLA
- 年份:2025 | 发表:arxiv 2505.21487(Princeton + Tri Dao) | 引用:近期发表,引用待累积
- 关键词:Grouped-Tied/Latent Attention、decoding 算术强度、MLA 改进
- 开源代码:公开

**核心问题**
Decoding 阶段的瓶颈不在于 FLOPs 而在于**内存带宽**(KV 加载),需要从架构层面提升算术强度。

**主要方法/贡献**
- **Grouped-Tied Attention(GTA):** GQA 改进,**算术强度 ~2×**,KV 减半。
- **Grouped Latent Attention(GLA):** 类 MLA 的低秩潜变量结构,**内核相比 FlashMLA 快 2×**(投机解码场景)。
- **专为 decoding 优化的硬件感知设计**

**关键结果**
- GLA 在投机解码场景显著优于 FlashMLA。

**创新点**
- 从"内存带宽 vs 算力"维度优化 decoding 注意力,与稀疏化路线正交。

**阅读价值判断**
- 架构层 decoding 优化的代表作。对希望从架构改造而非稀疏化路线优化推理速度的研究者必读。

---

## [29] NOSA: Native and Offloadable Sparse Attention

**论文基本信息**
- 简称:NOSA
- 年份:2025 | 发表:arxiv 2510.13602(2025年10月) | 引用:近期发表,引用待累积
- 关键词:KV cache offloading、CPU-GPU 传输约束、locality 分解、decoding 吞吐
- 开源代码:公开

**核心问题**
Decoding 吞吐受限于 GPU 显存，主要瓶颈是 KV cache。已有方案存在两难：训练免训的 KV offloading（如 InfLLMv2、ShadowKV）因训练-推理稀疏模式不匹配而在长生成任务上精度下降；而可训练的稀疏注意力（如 NSA）由于 KV 访问无约束，可能触发大量 CPU→GPU 传输，反而抵消 offloading 的吞吐收益。

**主要方法/贡献**
- **局部性分解:** 将 token 选择分解为 **query-aware**（逐步动态选）和 **query-agnostic**（静态局部窗口）两部分，后者可提前预载入 GPU，显著降低 CPU-GPU 传输量。
- **显式传输量约束:** 训练时将 CPU-GPU KV 传输体积作为硬约束纳入设计，使训练与 offloading 推理的稀疏模式对齐。
- **NOSI 推理系统:** 配套实现的 KV offloading 推理引擎，充分利用 NOSA 的局部性约束实现高效流水线。

**关键结果**
- Decoding 吞吐相比 FullAttn **5.04×**、相比 InfLLMv2 **1.92×**、相比 ShadowKV **1.83×**。
- 在通用、长输入、长生成三类任务上均优于 offloading 基线，精度无明显损失。
- 在 1B/3B/8B 模型规模上一致验证。

**创新点**
- 首个将"可训练稀疏注意力"与"KV offloading"统一设计的工作，解决了两者长期以来的不兼容问题。
- "训练时约束传输量"的设计思路为 memory-constrained decoding 提供了新范式。

**阅读价值判断**
- 是回答"NSA 训练稀疏对 offloading 场景是否有帮助"的直接实证工作，与 NSA 高度互补。其局部性分解方法对推理模型长 CoT 场景的 KV 管理有直接借鉴价值。推荐阅读。

---

## [30] Thought Anchors: Which LLM Reasoning Steps Matter?

**论文基本信息**
- 简称:Thought Anchors
- 年份:2025 | 发表:arxiv 2506.19143(2025年6月,ICLR 2026 在审) | 引用:近期发表,引用待累积
- 关键词:thought anchor 识别、receiver heads、black-box 重采样 + white-box 注意力聚合 + 因果 ablation
- 开源代码:公开

**核心问题**
推理模型生成的长 CoT 中,哪些 step 对最终答案起决定性作用?现有可解释性工作多在 token 粒度,缺少 step/sentence 粒度的 anchor 识别方法。

**主要方法/贡献**
- **三种互补识别方法:** black-box 重采样(扰动 step 看输出变化)、white-box 注意力聚合(找 receiver heads 广播关注的句子)、因果 ablation 验证。
- **Anchor 类型识别:** 发现 planning sentences 与 uncertainty management sentences 是主要 anchor。
- **Receiver heads 现象:** 一类头持续广播性地关注 anchor 句,与普通 head 行为差异显著。

**关键结果**
- 在 R1-Distill 等推理模型上验证 anchor 句删除导致显著精度下降,而非 anchor 句删除影响小。

**创新点**
- 首次系统提出 sentence-level "thought anchor" 概念,以多方法交叉验证而非单一 attention score。
- Receiver heads 是新发现的头功能类别。

**阅读价值判断**
- 与方向 1 中"step-locator / milestone"类直接重叠,是本方向 taxonomy 设计**最强外部参照**。其 receiver head 概念可纳入头功能分类体系。必读。

---

## [31] SEAL: Steerable Reasoning Calibration

**论文基本信息**
- 简称:SEAL
- 年份:2025 | 发表:COLM 2025(arxiv 2504.07986,2025年4月) | 引用:近期发表,引用待累积
- 关键词:execution/reflection/transition 三类 thought、latent steering、reasoning 校准
- 开源代码:公开

**核心问题**
推理模型 CoT 内部存在异质 thought 类型,过多的 reflection 或 transition 反而稀释推理质量。需要显式刻画 thought 类型并加以校准。

**主要方法/贡献**
- **三类 thought 划分:** execution(实际计算/推导)、reflection(自我验证/质疑)、transition(承接/总结)。
- **Latent steering:** 在 latent 空间识别每类 thought 的方向向量,推理时按需 steer 减少冗余 reflection。
- **零训练应用:** 推理时直接施加,无需重新训练。

**关键结果**
- 在 AIME/MATH 上以更短输出达到同等或更高精度,验证 thought 异质性可被显式调控。

**创新点**
- 首次将 CoT 显式分为 3 类 thought 并给出 steering 实现。
- 是 ThinKV 之外第二条"thought-type"路线,但走 steering 而非 KV 压缩。

**阅读价值判断**
- 与方向 1 的"统一 taxonomy"主张**直接重叠**,是必须正面比较与区隔的工作。区隔点应放在"attention-pattern 维度"而非"thought-content 维度"。必读。

---

## [32] SSA: Sparse Sparse Attention

**论文基本信息**
- 简称:SSA
- 年份:2025 | 发表:arxiv 2511.20102(2025年11月) | 引用:近期发表,引用待累积
- 关键词:稀疏注意力误差上界、特征空间对齐、dropped attention mass
- 开源代码:公开

**核心问题**
现有稀疏注意力方法缺乏对"稀疏化引入的近似误差"的形式化刻画,只能通过下游精度间接反映,理论保证薄弱。

**主要方法/贡献**
- **特征空间对齐分析:** 把 sparse 注意力输出与 full 注意力输出在 feature space 中对齐,推导差异上界。
- **线性误差界:** 证明近似误差 ∝ dropped attention mass(被剪掉的注意力权重总和),给出可计算的紧上界。
- **指导稀疏决策:** 用此界指导 top-k 选择阈值,而非凭经验。

**关键结果**
- 在长上下文基准上以同等精度匹配或超越现有 top-k 稀疏方法,且首次给出形式化误差保证。

**创新点**
- **2025-2026 年首个为稀疏注意力给出整体形式化误差上界**的工作。
- 把"剪枝越激进越好/越坏"从经验问题转为可量化问题。

**阅读价值判断**
- 直接威胁方向 1 的"theory-guided error bound"主张。区隔关键:SSA 是**整体 bound**,本方向必须做**per-pattern 不同形式的 bound**(milestone 类 lifecycle bound、recurrence 类 period-aware bound)才能立住差异化。必读。

---

## [33] ASEntmax (Long-Context Generalization with Sparse Attention)

**论文基本信息**
- 简称:ASEntmax
- 年份:2025 | 发表:ICLR 2026(arxiv 2506.16640,2025年6月) | 引用:近期发表,引用待累积
- 关键词:α-entmax、可学温度、长度外推 1000×、function-sparse
- 开源代码:公开

**核心问题**
softmax 注意力随上下文增长权重被过度平滑,长度外推性能急剧下降;固定 α 的 entmax 又难以适配不同长度需求。

**主要方法/贡献**
- **α-entmax 替代 softmax:** 用 α-entmax(α∈(1,2])诱导**函数级稀疏**(零权重内生),而非事后剪枝。
- **可学温度自适应长度:** 温度参数随序列长度自动调节,稀疏度随之自适应。
- **训练时引入,稳定收敛:** 提供数值稳定的实现与梯度路径。

**关键结果**
- 在长度外推任务上达到 **1000× 训练长度**仍保持精度,显著优于 softmax 与固定 α 的 entmax。

**创新点**
- 把"长度自适应稀疏"通过单一 α 参数 + 温度学习实现,极简而有效。
- 是 entmax 路线在长上下文的最强代表作。

**阅读价值判断**
- 与方向 1 中"α-entmax 函数稀疏"主张**直接重叠**且已是 ICLR 2026 工作。区隔关键:本方向把 entmax 作为**pattern-specific 操作器之一**(如 reflection 类的语义去重)注入到 mask 体系内,而非通用替代 softmax。必读。

---

## [34] Crystal-KV: Answer-First Principle for Reasoning KV Compression

**论文基本信息**
- 简称:Crystal-KV
- 年份:2026 | 发表:arxiv 2601.16986(2026年1月) | 引用:近期发表,引用待累积
- 关键词:Answer-First Principle、answer-anchor、SlipKV vs CrystalKV
- 开源代码:公开

**核心问题**
推理模型在 think 阶段产生大量 KV,但只有一部分对最终 answer 真正有贡献。现有压缩方法基于 attention score 而非 answer-relevance,会保留无关 think 内容。

**主要方法/贡献**
- **Answer-First Principle:** 以"是否对最终 answer token 的生成有贡献"为标尺评估 think 阶段每个 KV 的价值。
- **CrystalKV vs SlipKV 区分:** 把 think 阶段 KV 分为 crystal(对 answer 关键)与 slip(对 answer 不关键)两类,差异化保留。
- **后向贡献评分:** 利用 answer 阶段对 think 阶段的反向注意力分布作为贡献信号。

**关键结果**
- 在 R1-Distill 系列上以更激进的 think 阶段压缩比保持答案精度。

**创新点**
- 提出"answer-first"作为 reasoning KV 压缩的指导原则,与 milestone(RaaS)、recurrence(LazyEviction)、reflection(R-KV)并列为第四条原则。
- 把"think 阶段 KV 价值"用 answer 阶段反向定义,概念清晰。

**阅读价值判断**
- 与方向 1 中"answer-anchor"类**几乎完全重叠**,必须正面比较。区隔点:本方向把 answer-anchor 作为 6 类 pattern 之一,而非整套方法的核心原则。必读。

---

## [35] ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution

**论文基本信息**
- 简称:ForesightKV
- 年份:2026 | 发表:arxiv 2602.03203 | 引用:近期发表,引用待累积
- 关键词:推理模型 KV 淘汰、long-term contribution 学习、golden eviction SFT、MDP/GRPO、low-entropy 纠错关键 token 保护
- 开源代码:公开

**核心问题**
长 CoT 推理中 KV 缓存随输出线性增长。基于规则的淘汰(H2O/SnapKV/Quest)无法捕捉复杂注意力模式与语义依赖;而一次性(one-pass)训练的打分器无法跟踪生成不同阶段的动态重要性。需要一个能预测"长期贡献"的淘汰器。

**主要方法/贡献**
- **两阶段训练框架(LLM 权重冻结,仅训练打分器):**
  - **监督学习(Golden Eviction):** 用"未来注意力分数"构造最优淘汰标签——把注意力矩阵分块、计算每个 KV 的未来最大注意力,训练一个 MLP scorer 以 pairwise ranking loss 保持重要性相对序。
  - **强化学习(MDP/GRPO):** 把淘汰建模为 MDP,reward 专门盯住"low-entropy 但淘汰后 loss 暴增"的 token(数字、符号、实体),用 MSE reward 惩罚这类灾难性 loss spike,GRPO 细化策略。
- **纠错关键 token 保护:** 显式保护 low-entropy(底部 80%)且 loss 增量超阈值的 token,论文指出这些 token 的误差会沿长序列累积、扭曲后续推理,**对自我纠正与事实一致性至关重要**。

**关键结果**
- 半预算(2K budget)下保留 92% AIME 性能;在三个模型上优于 R-KV、SnapKV、H2O。
- 32K 生成长度下 **9.79× 吞吐**;数学训练可泛化到 science/coding。
- 模型:R1-Distill-Qwen-7B、Qwen3-4B/1.7B;基准:AIME24/25、GPQA、LiveCodeBench-V3。

**创新点**
- 首个用"未来注意力 golden label + RL"学习推理 KV 长期贡献的淘汰器,把"纠错关键 token 保护"显式做进 reward。

**阅读价值判断**
- **是 RescueKV idea 的最强竞品**——同样以"纠错关键 token / 长期贡献"为保护对象且已在 AIME 上验证。差异点必须落在:RescueKV 免训练(它需 SFT+RL)、周期感知(MRI 而非 future-attention max)、纠错事件驱动、且定位为可叠加其上的薄保护层。必读,且实验需把它当作"叠加后端"验证正交增益。

---

## [36] DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference

**论文基本信息**
- 简称:DefensiveKV(Taming Fragility)
- 年份:2026 | 发表:ICLR 2026(arxiv 2510.13334) | 引用:近期发表,引用待累积
- 关键词:KV 淘汰脆弱性、重要性骤变、max-aggregation 最坏情况风险、prior-risk 校正
- 开源代码:公开

**核心问题**
现有选择性淘汰假设"一个固定子集持续重要"。本文实证揭示该假设**脆弱**:平均重要性保持率约 0.92,但最坏情况会骤降到 0.34,存在频繁离群点。基于 mean aggregation 的方法只优化平均、忽视罕见极端负例。

**主要方法/贡献**
- **防御式聚合(Defensive Aggregation):** 用两步最坏情况风险管理替代均值聚合。
  - **最坏情况风险估计:** 取历史 token 上观测到的最大重要性 `R̃ᵢ = max Iⱼ,ᵢ`。
  - **自适应 prior-risk 校正:** 类 Laplace 平滑,若观测风险低于头级均值则用先验替代 `Rᵢ = max(R̃ᵢ, R̄)`。
  - 两步均线性时间、开销可忽略。

**关键结果**
- DefensiveKV / Layer-DefensiveKV 在 20% 预算下相对 CriticalKV 把质量损失降低 **2.3× / 4.3×**;最坏情况保留重要性从 0.33 提升到 0.61;18 数据集 7 类任务近无损。

**创新点**
- 把 KV 淘汰从"优化平均"转为"管理最坏情况风险",从金融期望/尾部风险类比切入。

**阅读价值判断**
- 与 RescueKV 的"重要性脆弱/复发"动机直接呼应——复发正是被均值掩盖的最坏情况。区隔:它做通用聚合鲁棒化,RescueKV 做纠错语义专用 + 周期感知 + page-quota 豁免。是 RescueKV 必比 baseline 与可叠加对象。必读。

---

## [37] G-KV: Decoding-Time KV Cache Eviction

**论文基本信息**
- 简称:G-KV
- 年份:2025 | 发表:arxiv 2512.00504 | 引用:近期发表,引用待累积
- 关键词:decoding-time KV 淘汰、global attention 打分、推理任务、training-free
- 开源代码:公开

**核心问题**
长上下文/长生成中 KV 缓存成为部署瓶颈,需要在 decoding 时做选择性缓存管理而不损质量。

**主要方法/贡献**
- **global attention 打分:** 在推理时用全局注意力机制计算重要性,淘汰低注意力贡献的 token,仅保留最相关 KV。
- **decoding-time、training-free:** 直接用于已训练模型,无需重训。

**关键结果**
- 多基准(AMC8/AIME 数学推理、LiveCodeBench 编码、对话)上有效压缩 KV 同时保持竞争性能。

**创新点**
- 面向推理负载的免训练 decoding 阶段淘汰。

**阅读价值判断**
- 中等相关:同为 decoding-time、training-free、推理基准,可作 RescueKV 的对照后端之一,但无"自我纠正事件"语义维度。泛读即可。

---

## [38] LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future

**论文基本信息**
- 简称:LookaheadKV
- 年份:2026 | 发表:ICLR 2026(arxiv 2603.10899) | 引用:近期发表,引用待累积
- 关键词:learned lookahead tokens、LoRA、未来注意力预测、proactive 保留、长上下文
- 开源代码:公开

**核心问题**
随上下文增长 KV 缓存成为瓶颈,需要在不显式生成 draft 的前提下预测未来响应会关注的 prompt KV。

**主要方法/贡献**
- **"glimpse into the future":** 通过 learned lookahead tokens 预测哪些 prompt KV 对未来响应重要,主动保留相关 KV。
- 用 LoRA 训练前瞻预测,避免为获取未来 query 而显式进行 draft generation。

**关键结果**
- LongBench 等长上下文基准上保持精度同时降低内存(Llama 系列)。

**创新点**
- 用"前瞻预测"替代"事后观测"做淘汰决策。

**阅读价值判断**
- 中等相关:与 Expected Attention[26]、AttentionPredictor[22] 同属"预测未来注意力"路线,但偏通用长上下文(LongBench)而非推理纠错。可作方法对照,泛读即可。

---

## [39] CaliDrop: KV Cache Compression with Calibration

**论文基本信息**
- 简称:CaliDrop
- 年份:2025 | 发表:arxiv 2507.19906(已转投,OpenReview id=2WzCzpkeTc) | 引用:近期发表,引用待累积
- 关键词:KV 淘汰后不删、offload + query 相似度投机校准、可叠加任意淘汰后端
- 开源代码:公开

**核心问题**
token 淘汰在高压缩比下精度骤降。

**主要方法/贡献**
- **淘汰不删,改为 offload:** 被淘汰 KV 不直接丢弃,而是 offload。
- **投机校准(speculative calibration):** 用当前 query 与被淘汰 KV 做注意力计算,基于"相邻位置 query 高度相似"的观察,保留 query、softmax 分母、注意力输出供后续校准。
- **可叠加任意 token 淘汰后端**

**关键结果**
- 可叠加在任意 token 淘汰方法之上,显著提升其精度。

**创新点**
- 首次把"淘汰后救回"做成可叠加的计算侧补偿机制,与单纯淘汰策略形成正交补救层。

**阅读价值判断**
- **是 RescueKV 机制最近邻竞品**。CaliDrop 与 RescueKV 都做"淘汰后救回 + 叠加任意后端"。**区隔点**:CaliDrop 在**计算侧**用 query 相似度近似补偿,救的是"贡献值";RescueKV 在**存储侧**用"复发周期 × 纠错事件"语义信号决定**保留哪些 page 不删**,救的是"KV 实体"且绑定自我纠正语义。RescueKV 必须在论文里把 CaliDrop 列为最近邻并明确"存储侧豁免 vs 计算侧校准""query 相似度 vs 纠错事件语义"两条分野,否则会被审稿人指为 CaliDrop 的语义变体。必读。

---

## [40] TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

**论文基本信息**
- 简称:TriAttention
- 年份:2026 | 发表:arxiv 2604.04921(MIT + NVIDIA + 浙大,2026年4月) | 引用:近期发表,引用待累积
- 关键词:pre-RoPE 三角距离打分、Recursive State Query benchmark、回溯压力测试
- 开源代码:公开(GitHub)

**核心问题**
主流方法用 post-RoPE query 的注意力分估计 KV 重要性,但 query 随位置旋转、代表性 query 太少 → top-key 选择差、推理不稳。

**主要方法/贡献**
- **pre-RoPE 几何观察:** 观察 pre-RoPE 空间中 Q/K 高度集中于固定非零中心且跨位置稳定 → query 偏好特定距离的 key(三角级数刻画)。
- **距离偏好 + Q/K 范数打分**
- **Recursive State Query benchmark:** 递归任务要求模型跨长链保持中间状态并**回溯(backtrack)**,深度 16 内匹配 Full Attention。

**关键结果**
- AIME25 32K 生成下匹配 Full Attention 精度,**2.5× 吞吐 / 10.7× KV 内存压缩**;基线在同等效率下只有约一半精度。

**创新点**
- 首次系统刻画 pre-RoPE Q/K 几何并据此设计稀疏选择。
- **首个专为回溯/递归状态保持设计的 KV 压缩 benchmark**。

**阅读价值判断**
- **直接占据 RescueKV 自称"没人专门测回溯"的评测缝隙**。TriAttention 测的是"递归算法的中间状态回溯",不是"自我纠正/reflection/答案翻转"——这是 RescueKV 仍可立足的细缝。**应对**:RescueKV 评测必须明确区分"recursion 状态回溯"(TriAttention)与"self-correction / reflection 触发的语义回指"(RescueKV 专攻),并直接把 TriAttention 的 Recursive State Query 纳入对照。必读。

---

## [41] SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning

**论文基本信息**
- 简称:SparK
- 年份:2025 | 发表:arxiv 2508.15212(AMD,2025年8月) | 引用:近期发表,引用待累积
- 关键词:channel 维剪枝、动态恢复、training-free、与 KV 压缩正交
- 开源代码:公开(ROCm blog 有解读)

**核心问题**
channel saliency 随 query/position 剧烈变化;某些 channel 对特定 query 近零信息。

**主要方法/贡献**
- **关键 channel 集选择:** 把剪枝重构为 query-aware 的关键 channel 选择问题。
- **动态恢复机制:** 在注意力打分时用轻量函数近似被剪 channel 的贡献。
- **training-free,与现有 KV 压缩/量化正交可叠加**

**关键结果**
- 相比基方法精度下降 <5%(THINK 在同设置掉 47.6%)。

**创新点**
- 首次在 channel 维做 query-aware 剪枝-恢复;与 token/page 维淘汰天然正交。

**阅读价值判断**
- 撞 RescueKV"training-free + 正交叠加原语 + 动态恢复"三重定位。**区隔点**:SparK 是 **channel 维**剪枝-恢复(正交于 token/page 维淘汰),RescueKV 是 **token/page 维**的语义豁免;二者实际上**可同时叠加且不冲突**。建议 RescueKV 把 SparK 归为"正交维度"而非竞品,甚至作为可叠加后端之一。

---

## [42] GraphKV: Breaking the Static Selection Paradigm with Graph-Based KV Cache Eviction

**论文基本信息**
- 简称:GraphKV
- 年份:2025 | 发表:**EMNLP 2025 main**(arxiv 2509.00388) | 引用:近期发表,引用待累积
- 关键词:图传播动态重要性更新、plug-and-play 叠加 SnapKV/PyramidKV
- 开源代码:公开

**核心问题**
top-k 等静态启发式无法捕捉 token 间随推理演化的隐式依赖。

**主要方法/贡献**
- **token 图建模:** token 为节点(带重要性分),边为相似关系。
- **decay-signal-propagation:** 在图上传播、动态更新 token 重要性。
- **plug-and-play 叠加 SnapKV / PyramidKV**

**关键结果**
- 在长上下文与推理任务上稳定优于静态 top-k 基线。

**创新点**
- 首次把"重要性动态更新"建模为图上信号传播,而非每步独立打分。

**阅读价值判断**
- 已 EMNLP 2025 main 发表,核心卖点之一是"动态更新重要性 + 即插即用叠加现有淘汰器"—— 与 RescueKV"正交叠加层"定位重叠。**区隔点**:GraphKV 是用图传播**重新计算通用重要性**(本质仍是更好的淘汰打分),RescueKV 是绑定**自我纠正事件语义**的少量豁免,不重打分。RescueKV 需强调"不重做重要性估计、只对纠错相关 token 行使豁免名额"。必读。

---

## [43] TRIM-KV (Cache What Lasts): Token Retention for Memory-Bounded KV Cache in LLMs

**论文基本信息**
- 简称:TRIM-KV / Cache What Lasts
- 年份:2026 | 发表:ICLR 2026(arxiv 2512.03324) | 引用:近期发表,引用待累积
- 关键词:token 创建时的内在长期重要性、轻量 retention gate、分数衰减、蒸馏微调
- 开源代码:公开

**核心问题**
saliency shift —— token 重要性在 decoding 过程中变化,某 token 可能很久后才再次关键(即便近期未被关注)。

**主要方法/贡献**
- **轻量 retention gate:** 在 token 创建时预测按层、按头的内在长期重要性,而非依赖近期注意力。
- **时间衰减与固定预算:** 保留分数随时间衰减,预算溢出时淘汰低分 token。
- **低成本训练:** 冻结原模型,用蒸馏损失和 capacity loss 只微调 gate。

**关键结果**
- 在 GSM8K、MATH-500、AIME24、LongProc、LongMemEval、LongBenchV2 和 SCBench 上稳定优于强淘汰/可学检索基线,优势在低内存预算下更明显。
- 部分设置甚至超过 full-cache,表明选择性保留可能通过抑制无信息 token 形成正则化效果。

**创新点**
- 把"长期重要性"前置为 token 内在属性,并用轻量可解释 gate 直接学习,而非事后通过注意力观测。

**阅读价值判断**
- 与 RescueKV"复发感知"动机同源(都针对"久未关注但将再次关键")。**区隔点**:Cache What Lasts 预测的是创建时静态内在分,RescueKV 用**在线复发周期统计 + 实时纠错事件**两路动态信号,且免训练(Cache What Lasts 需学习内在重要性)。值得精读。

---

## [44] SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning

**论文基本信息**
- 简称:SideQuest
- 年份:2026 | 发表:arxiv 2602.22603(NVIDIA;Sanjay Kariyappa, G. Edward Suh) | 引用:近期发表,引用待累积
- 关键词:model-driven KV 管理、agentic 长程推理、辅助任务并行压缩
- 开源代码:公开

**核心问题**
agentic 长程任务(deep research)context 被外部检索 token 主导,内存暴涨。

**主要方法/贡献**
- **模型自驱压缩:** 让 LRM **自身**推理 context 中 token 的有用性来做压缩。
- **并行辅助任务:** 把 KV 压缩设为与主推理并行的辅助任务,避免管理过程 token 污染记忆。
- **小样本训练:** 仅用 215 样本训练。

**关键结果**
- 峰值 token 降 65%。

**创新点**
- 首个把 KV 压缩做成 LRM 自身的辅助推理任务、与主推理并行。

**阅读价值判断**
- 场景偏 agentic 检索而非数学 CoT 纠错;且需训练。可作为"模型自驱压缩"对照路线提一句。与 RescueKV 直接对抗性低。泛读即可。

---

## [45] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction

**论文基本信息**
- 简称:Judge Q
- 年份:2025 | 发表:arxiv 2509.10798(2025年9月) | 引用:近期发表,引用待累积
- 关键词:可学习 query 优化 KV 淘汰信息保留、trained 路线
- 开源代码:公开

**核心问题**
基于固定 query 的注意力打分淘汰会丢失重要信息。

**主要方法/贡献**
- 训练一组可学习 query,优化淘汰时的信息保留。

**关键结果**
- 在长上下文任务上优于固定 query 的淘汰打分。

**创新点**
- 把"query 本身"作为可优化对象引入 KV 淘汰流程。

**阅读价值判断**
- trained 路线,与 RescueKV 免训练定位对立,可归入"需训练的 query 优化"对照组。泛读即可。

---

## [46] KV Cache as a Reasoning Primitive for Long Context Reasoning

**论文基本信息**
- 简称:KV-as-Reasoning-Primitive
- 年份:2025 | 发表:OpenReview id=vs2qwVfU2C(在投) | 引用:近期发表,引用待累积
- 关键词:content-aware retention、premise 保留、长上下文逻辑一致性
- 开源代码:暂无

**核心问题**
长上下文中早期前提被部分遗忘/扭曲 → 跨相关问题答案不一致;KV 策略决定哪些前提可被注意力访问,从而调控逻辑一致性。

**主要方法/贡献**
- **content-aware retention:** PoC cache manager 显式保留前提相关 KV。
- **premise-retrieval 压力测试**

**关键结果**
- 在 premise-dependent 任务上一致性显著优于不区分内容的淘汰器。

**创新点**
- 把 KV 保留视为"推理原语",强调保住关键早期信息以维持一致性。

**阅读价值判断**
- 也把 KV 保留当"推理原语"、也强调保住关键早期信息以维持一致性。**区隔**:它针对**前提一致性**,RescueKV 针对**自我纠正/回溯**;评测维度不同。命名上"reasoning primitive"与 RescueKV"protection primitive"相近,需注意区分。值得关注。

---

## [47] ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget

**论文基本信息**
- 简称:ARKV
- 年份:2026 | 发表:arxiv 2603.08727(2026) | 引用:近期发表,引用待累积
- 关键词:受限预算下自适应资源高效 KV 管理
- 开源代码:暂无

**核心问题**
受限内存预算下如何自适应分配 KV 资源以保持长上下文性能。

**主要方法/贡献**
- 受限预算下的自适应资源高效 KV 管理,偏系统/长上下文。

**关键结果**
- 在 memory-budget 设定下取得 Pareto 优。

**创新点**
- 系统侧预算分配视角。

**阅读价值判断**
- 背景参照,预算约束管理对照。泛读即可。

---

## [59] FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention

**论文基本信息**
- 简称:FM-DS-V4 / FlashMemory
- 年份:2026 | 发表:arxiv 2606.09079(v3,2026年7月) | 引用:近期发表,引用待累积
- 作者/机构:Yan Wang 等;Tencent、HKUST(GZ)、清华大学及独立研究者
- 关键词:Lookahead Sparse Attention、Neural Memory Indexer、DeepSeek-V4 CSA、KV cache offloading、超长上下文
- 论文:https://arxiv.org/abs/2606.09079
- 开源代码:https://github.com/libertywing/FlashMemory-Deepseek-V4
- 检索器权重:https://huggingface.co/libertywing/FlashMemory-Deepseek-V4

**核心问题**
DeepSeek-V4 的 CSA 虽然已将每步注意力计算稀疏化,但完整历史 KV 仍常驻 GPU,内存随 context 线性增长。论文要在不牺牲全局精细检索能力的前提下,只把当前及近期生成真正会用到的 CSA KV 放在 GPU。

**主要方法/贡献**
- **Lookahead Sparse Attention:** 每 64 个 decode step 由 Neural Memory Indexer 预测未来 64 token 会用到的历史 chunk;被选 chunk 从 CPU cold pool 异步召回 GPU,其余 CSA KV 保留在 CPU。
- **两级检索:** 第一级 Memory Indexer 用 Sigmoid 阈值动态召回 chunk,第二级再在居留集内运行 DeepSeek-V4 原生 Lightning Indexer 的 top-k 精选;同时保留全部 128:1 压缩的 HCA、prompt 最后 8K token 和已生成 token。
- **Backbone-free 训练:** 把 indexer 当作双编码检索器,用冻结的 compressed keys 和预计算 hidden states 独立训练;标签由未来窗口内 top-p 选择加跨层多数投票构造。最终只在第 10/12/20 层放置 indexer,训练约需 1 个 H20 GPU 小时。
- **系统实现:** 在 PD 分离服务中将 recall 放到 CUDA graph 之外异步执行,让预取和 decode 重叠,将算法稀疏真正转化为 HBM 节省。

**关键结果**
- LongBench-v2、LongMemEval 和 RULER 共 9 个设置的平均分从 DS-V4-Flash 的 **76.9 升至 77.5(+0.6)**,平均 GPU KV 开销从 **0.93 GB 降至 0.10 GB**;论文按单样本比例统计为仅保留基线的 **13.5%**,即减少 **86.5%**。
- LongBench-v2-L(493K) 上精度 **68.1→70.0(+1.9)**,同时 KV 开销 **1.80→0.18 GB**;LongMemEval-M(500K) 上 **39.3→40.2**,KV **1.82→0.17 GB**。
- 1M context 的8×H20部署中,GPU KV 从 **3.73→0.37 GB(-90%)**,每 decode token 计算从 **118.9→35.4 GFLOP(0.30×)**,聚合吞吐提升 **2.8×**,并发能力提升 **2.7×**。
- 稀疏化并非普遍无损:RULER-256K 从 **90.5 降至 88.2**;更极端的密集全局记忆 MRCR 上从 **76.0 降至 48.0**,表明该双编码器对 dense-memory 任务仍会严重漏召回。

**创新点**
- **效率 + 新方法:** 已有 sparse attention 多聚焦“当前步算哪些 token”,LSA 进一步预测“未来一段会需要哪些 KV”,同时把索引器变成 GPU—CPU 内存调度器。
- 与 SparDA 的跨层一步预取不同,它面向未来 64-token 窗口做 horizon-level 保留,核心优化目标是物理 KV 驻留集,而不只是隐藏传输延迟。

**局限性**
- 分类阈值会在超长干扰池中累积假阳性,因而 context 无关任务上的绝对内存并未达到理想的常数级。
- 训练最长 512K 时只可靠泛化到约 2×长度,超出后因位置分布外偏移而选择退化;且 64-step 间隔与 0.5 阈值未做完整 ablation。
- 项目已暂停主动开发,当前结果应视为有限算力下的强 PoC,而非已完善的通用长上下文方案。

**阅读价值判断**
对本列表的 decoding 稀疏注意力方向很有价值:它把“未来注意力预测”与“物理 KV offload/recall”连成了一个可部署系统,且给出 1M context 的端到端数据。建议精读 Method 2.1–2.5 和 MRCR 失败分析,后者直接揭示了 learned retrieval 在“高 precision 省内存”与“高 recall 保能力”之间的核心张力。

---

## [60] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

**论文基本信息**
- 简称:FreeKV
- 年份:2026 | 发表:ICLR 2026 | 引用:近期发表,引用待累积
- 关键词:speculative retrieval、细粒度校正、CPU–GPU 混合 KV 布局、双缓冲流式召回
- 论文:https://iclr.cc/virtual/2026/poster/10006722

**核心问题**
KV 永久删除容易丢失精度,而把完整 KV 放到 CPU 后按需召回又会让选择和传输落在 decoding 关键路径上。

**主要方法/贡献**
- **投机检索:** 预先选择并召回可能需要的 KV,把检索移出关键路径。
- **细粒度校正:** 对投机结果补做校正,降低预取错误对精度的影响。
- **系统协同:** CPU/GPU 采用混合 KV 布局避免碎片传输,并用双缓冲 streamed recall 重叠传输与计算。

**关键结果**
- 多模型、多场景下达到近无损精度;相对当时先进 KV retrieval 方法最高加速 **13×**。

**阅读价值判断**
- 这是“不删 KV,而做分层召回”的关键系统基线,适合与 CaliDrop、LouisKV、IceCache 对照。精读其预取命中率、校正成本和 PCIe 延迟隐藏条件。

---

## [61] LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences

**论文基本信息**
- 简称:LouisKV
- 年份:2026 | 发表:ICLR 2026 | 引用:近期发表,引用待累积
- 关键词:关键 KV 时间局部性、语义边界检索、输入/输出解耦管理、Triton/CUDA kernel
- 论文:https://iclr.cc/virtual/2026/poster/10011378

**核心问题**
现有 KV retrieval 通常每 token 检索,并以粗粒度 page 管理 KV;在长输出推理中,这会同时带来频繁检索开销和关键 KV 定位不准。

**主要方法/贡献**
- **时间局部性:** 发现 decoding 中关键 KV 在时间上高度局部,只在语义边界触发检索。
- **输入/输出解耦:** 针对 prompt KV 和 generated KV 的不同分布设计差异化、细粒度 retrieval unit。
- **内核优化:** 自定义 Triton/CUDA kernel 加速 KV 聚类和检索。

**关键结果**
- 在长输入短输出、短输入长输出和长输入长输出三类场景下均保持近无损精度,相对先进 KV retrieval 方法最高加速 **4.7×**。

**阅读价值判断**
- 它将“何时检索”与“输入/输出如何分别管理”作为两个核心设计轴,对 long-input + long-output 的真实 decoding 系统很有参考价值。

---

## [62] PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

**论文基本信息**
- 简称:PM-KVQ
- 年份:2026 | 发表:ICLR 2026 | 引用:近期发表,引用待累积
- 关键词:长 CoT、渐进式混合精度 KV 量化、block-wise 内存分配、位置插值校准
- 论文:https://iclr.cc/virtual/2026/poster/10009118
- 开源代码:https://github.com/thu-nics/PM-KVQ

**核心问题**
直接把短上下文 KV 量化方法用到长 CoT 会在每步 decoding 累积误差;短序列校准还无法覆盖 RoPE 下远位置的罕见 key channel 分布。

**主要方法/贡献**
- **渐进量化:** 在每个 block 中逐渐降低 KV bit-width,避免过早丢失精度并充分利用内存。
- **block-wise 预算分配:** 为更敏感的 Transformer block 分配更高 bit-width。
- **位置插值校准:** 用短校准数据近似长上下文位置分布,不增加长校准开销。

**关键结果**
- 在 7B–70B 长 CoT 模型上,同内存预算下比量化基线最高提升 **8%** 推理性能;相对原始 16-bit LLM 达到 **2.73–5.18×** 吞吐。

**阅读价值判断**
- 代表“不删 token,而按时间和 block 降精度”的可逆压缩路线,应与 ThinKV 的量化+淘汰混合策略对照。

---

## [63] QuoKA: Query-Oriented KV Selection for Efficient LLM Prefill

**论文基本信息**
- 简称:QuoKA
- 年份:2026 | 发表:ICLR 2026 | 引用:近期发表,引用待累积
- 关键词:chunked prefill、query-oriented KV selection、代表性 query、training-free、硬件无关
- 论文:https://iclr.cc/virtual/2026/poster/10008892

**核心问题**
chunked prefill 中不同 query 的信息量不同;若所有 query 均匀参与 key 选择,会带来不必要的注意力计算。

**主要方法/贡献**
- 发现与平均 query 余弦相似度低的 query 会与更多 key 交互,对最终 attention logits 贡献更大。
- 先保留少量代表性 query,再选出与它们最对齐的 key,以近似 full attention。
- 方法免训练且不绑定特定硬件。

**关键结果**
- 使每次 attention 评估的 KV 减少 **88%**,TTFT 降低 **3×**;注意力算子在 NVIDIA GPU 上加速 **5×**,在 Intel Xeon CPU 上接近 **7×**,同时保持接近基线精度。

**阅读价值判断**
- 主要优化 **prefill** 而非 decoding eviction,与 Quest 等 decode 期 query-aware 选择的目标不同;可作为输入端稀疏基线,与 decoding 策略正交组合。

---

## [64] IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs

**论文基本信息**
- 简称:IceCache
- 年份:2026 | 发表:ICLR 2026 | 引用:近期发表,引用待累积
- 关键词:语义 token 聚类、PagedAttention、CPU–GPU offloading、分层动态数据结构
- 论文:https://iclr.cc/virtual/2026/poster/10006569
- 开源代码:https://yuzhenmao.github.io/IceCache/

**核心问题**
现有 CPU offloading 方法容易选错 token,且零散 KV 召回不能有效利用传输带宽,在长生成与 CoT 中尤其明显。

**主要方法/贡献**
- 将语义相关 token 聚类到连续内存区域,使选择粒度与传输粒度更一致。
- 用分层、可动态更新的数据结构管理聚类,并与 PagedAttention 集成。
- 通过连续布局提高 CPU–GPU 批量传输的带宽利用率。

**关键结果**
- LongBench 上仅用 256-token GPU 预算保留 full-cache **99%** 的原始精度;相对其他 offloading 方法,仅用 **25%** KV token 预算便取得有竞争力或更好的延迟和精度。

**阅读价值判断**
- 特色在于用语义聚类同时解决“选谁”和“如何连续传”,适合与 FreeKV 的投机预取和 LouisKV 的语义边界触发对比。

---

## [65] FusedKV: Reconstructing KV Caches with Cross-Layer Fusion for Enhanced Transformers

**论文基本信息**
- 简称:FusedKV
- 年份:2026 | 发表:ICLR 2026 | 引用:近期发表,引用待累积
- 关键词:跨层 KV 共享、底层/中层信息融合、post-RoPE key、FusedKV-Lite
- 论文:https://iclr.cc/virtual/2026/poster/10011526

**核心问题**
YOCO、CLA 等跨层 KV 共享虽能减少 cache,但往往不如 GQA 等层内方法;原因是高层 key 和 value 需要的来源信息并不相同。

**主要方法/贡献**
- 实证发现高层 value 主要来自底层,而 key 同时需要底层和中层信息。
- **FusedKV:** 用可学融合从底层与中层重构高层 KV;直接融合 post-RoPE key,避免重做旋转位置编码。
- **FusedKV-Lite:** 高层 value 直接用底层、key 直接用中层,以小幅 perplexity 代价减少 I/O。

**关键结果**
- 在 332M–4B 模型上减少 **50%** KV cache,同时验证 perplexity 低于标准 Transformer decoder;提供 Triton 实现。

**阅读价值判断**
- 它是需要改变模型结构并训练的跨层共享路线,与 training-free token eviction 不是直接竞品,但是研究“不同层到底需要保留什么”的重要结构对照。

---

## 补充论文单句摘要

## [48] SnapKV: LLM Knows What You Are Looking for Before Generation

- 利用生成前 observation window 的注意力聚合预测重要 KV，在固定预算下保留关键上下文并压缩 cache。

## [49] StreamingLLM: Efficient Streaming Language Models with Attention Sinks

- 保留 attention sink 和最近窗口 token，使 LLM 能在有限 KV cache 下稳定处理超长流式输入。

## [50] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

- 基于 heavy hitter token 假设累计注意力分数，动态保留持续被关注的 token 与近期 token，以降低 cache 开销。

## [51] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 通过低秩 key cache、value offloading 和按需稀疏 KV 重构，在控制传输成本的同时提升长上下文推理吞吐。

## [52] InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation

- 复用 dense attention 参数构建可切换的稠密—稀疏结构，使模型按序列长度平滑切换注意力模式。

## [53] Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction

- 发现全局预算淘汰中 prompt 边界等结构 token 的显式保护比打分差异更关键，少量保护即可显著恢复质量。

## [54] EpiKV: Epiphany-Aware KV Cache Eviction Without the Attention Matrix

- 利用跨层隐藏状态变化识别推理转折 token，并在不物化注意力矩阵的情况下执行 training-free KV 淘汰。

## [55] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

- 在 RL 后训练中仅对少量高熵 forking token 更新策略梯度，即可匹配或超过全 token 训练效果。

## [56] SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference

- 增加轻量 Forecast projection 预测下一层所需 KV block，以跨层预取隐藏 CPU—GPU 传输并降低稀疏选择开销。

## [57] Lethe: Layer- and Time-Adaptive KV Cache Pruning for Reasoning-Intensive LLM Serving

- 同时按层分配剪枝预算并在生成过程中多轮动态淘汰，实现空间和时间两个维度的自适应 KV 压缩。

## [58] IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse

- 复用相邻层高度相似的 top-k 稀疏索引，并通过 training-free 或 training-aware 配置减少 indexer 计算。

---

## 附录:其他相关工作

| 名称 | 类别 | 备注 |
|------|------|------|
| The Sparse Frontier | 实证综述 | arxiv 2504.17768,稀疏注意力 6 方法 × 9 任务 |
| Qwen3-Next | 混合架构 | Gated DeltaNet + Gated Attention + 超稀疏 MoE |
| FlexPrefill | prefill 稀疏 | ICLR 2025 Oral,与本方向相关性较低(prefilling) |
| SCBench | benchmark | ICLR 2025,KV-cache-centric 全生命周期评测 |
| SamKV | 多上下文 KV | 首个 multi-context KV 稀疏 |
| SALS | 潜空间稀疏 | 低秩投影 + RoPE-free 稀疏选择 |
| AhaKV | KV 淘汰 | arxiv 2506.03762,自适应整体注意力驱动淘汰,通用 LLM |
| RetentiveKV | 多模态 KV | arxiv 2605.04075,state-space memory + 不确定性感知多模态淘汰 |
| WindowKV | 任务自适应 KV | 连续语义窗口分配,约 12% 缓存达 1.5–2× 加速 |
