# R-KV 及后续推理模型 KV Cache 压缩方法调研

> 整理时间：2026 年 8 月 12 日
> 调研范围：面向长链式推理模型、作用于自回归解码阶段的 KV Cache 淘汰、压缩或预算分配方法。

## 1. R-KV 基本信息

- 论文：[R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration](https://arxiv.org/abs/2505.24133)
- 首次公开：2025 年 5 月
- 发表：NeurIPS 2025
- 目标：解决推理模型生成长 Chain-of-Thought 时，KV Cache 随输出长度线性增长造成的显存与推理吞吐瓶颈。

## 2. R-KV 的主要思想

R-KV 的关键判断是：推理过程中的 token 并非同等重要，而且很多 token 在语义或注意力作用上高度冗余；如果只根据累计注意力分数保留 token，容易留下大量彼此相似的信息，并误删对后续推理关键但暂时注意力较低的状态。

它采用训练无关的“重要性 + 冗余性”联合选择策略：

1. 根据历史注意力估计 KV token 对后续生成的重要程度。
2. 同时衡量候选 token 之间的相似或冗余关系，避免有限预算被重复信息占满。
3. 保留高重要性且具有多样性的 KV 状态，并在解码过程中持续更新缓存。
4. 不修改模型参数，不需要重新训练，可以直接用于已有推理模型。

一句话概括：**R-KV 不只是保留“被关注最多”的 token，还尽量让保留下来的 token 彼此不重复，从而在小缓存中覆盖完整的推理信息。**

## 3. R-KV 的实验设置与结果

### 3.1 实验设置

- 模型以 DeepSeek-R1 蒸馏系列推理模型为主，包括 Qwen 和 Llama 架构版本。
- 任务以长输出数学推理为主，包括 MATH-500、AIME 等数据集。
- 比较对象包括 Full KV Cache，以及 H2O、SnapKV、StreamingLLM、TOVA 等通用 KV Cache 压缩方法。
- 主要考察最终答案准确率、KV Cache 占用、吞吐率，以及不同缓存预算下的性能退化。

### 3.2 主要结果

- 使用约 10% 的原始 KV Cache 时，R-KV 可以保留接近 100% 的 FullKV 推理性能。
- 使用约 16% 的 KV Cache 时，部分设置中的成绩达到 FullKV 的约 105%，说明删除冗余状态有时还能减少错误推理干扰。
- 相比完整缓存，最高报告约 90% 的 KV 显存节省和 $6.6\times$ 的推理吞吐提升。
- 通用 KV 淘汰方法在长推理中可能严重破坏逻辑链，而 R-KV 在相同预算下整体明显更稳健。

## 4. 后续直接优于 R-KV 的方法

下面采用较严格的筛选口径：论文需要面向解码阶段的长推理 KV Cache，并且直接把 R-KV 作为实验基线，在匹配或更小预算下体现出总体优势。月份按 arXiv v1 首次公开时间计算。

截至 2026 年 8 月 12 日，共确认 10 篇较直接的后续工作。

| 时间 | 论文与发表情况 | 一句话介绍及相对 R-KV 的结果 |
|---|---|---|
| 2025 年 6 月 | [LazyEviction](https://arxiv.org/abs/2506.15969)，[ACL 2026 Main](https://2026.aclweb.org/program/accepted_papers/) | 延迟淘汰以观察注意力重要性的周期性回归；MATH-500 在 50% 压缩下，部分模型取得 $75.2$，R-KV 为 $73.2$。 |
| 2025 年 10 月 | [ThinKV](https://arxiv.org/abs/2510.01290)，[ICLR 2026 Oral](https://iclr.cc/virtual/2026/poster/10009980) | 根据不同思考阶段的重要性联合采用量化与淘汰，在不足 5% 原始 KV 容量下接近无损，并报告最高约 $5.8\times$ 于 R-KV 的吞吐率。 |
| 2025 年 10 月 | [RLKV](https://arxiv.org/abs/2510.08525)，[ICML 2026](https://kurt232.github.io/RLKV/) | 用强化学习识别关键推理头，并为其分配更完整的缓存；一个 40% 稀疏度设置下 MATH-500 为 $84.6$，R-KV 为 $77.8$。 |
| 2025 年 11 月 | [G-KV](https://arxiv.org/abs/2512.00504)，arXiv 预印本 | 将局部注意力与历史衰减注意力组合成全局重要性分数，在多数预算、尤其极小缓存预算下优于 R-KV。 |
| 2026 年 1 月 | [Crystal-KV](https://arxiv.org/abs/2601.16986)，arXiv 预印本 | 依据“答案优先”原则区分真正帮助最终答案的 CrystalKV 与临时 SlipKV，相对 R-KV 在代码和数学任务上平均提升约 $12.46$ 和 $7.97$ 个百分点。 |
| 2026 年 2 月 | [ForesightKV](https://arxiv.org/abs/2602.03203)，[ICML 2026](https://openreview.net/forum?id=znV8JHv8b8) | 用未来注意力构造监督信号，再结合排序学习与 GRPO 学习 KV 的长期贡献；Qwen3-4B 在 AIME24 上用 1K 缓存取得 $54.5$，高于 R-KV 使用 2K 时的 $44.8$。 |
| 2026 年 4 月 | [TriAttention](https://arxiv.org/abs/2604.04921)，ICML 2026 | 利用 pre-RoPE 空间中稳定的 Q/K 中心和三角距离偏好预测重要 KV；Qwen3-8B、AIME25、2048 预算下为 $32.9$，R-KV 为 $17.5$。 |
| 2026 年 5 月 | [AMS](https://arxiv.org/abs/2605.23200)，arXiv 预印本 | 通过区域级配额避免全局 Top-$k$ 将连续推理块整体删除；作为 R-KV 插件时，MATH-500 的 512 预算由 $46.4$ 提升至 $53.6$。 |
| 2026 年 6 月 | [VaSE](https://arxiv.org/abs/2606.03928)，arXiv 预印本 | 保护大幅值 Value 状态并引入随机淘汰维持缓存多样性，在 $4\times$ 压缩下相对 R-KV 的平均准确率提高约 $4.4$–$4.9$ 个百分点。 |
| 2026 年 6 月 | [ReasonAlloc](https://arxiv.org/abs/2606.11164)，arXiv 预印本 | 在层级和注意力头级动态分配 R-KV 的缓存预算，MATH-500、512 预算下取得 $82.50$，R-KV 为 $76.48$。 |

## 5. 结果更强但实验口径不同的相关方法

这些论文也直接对比了 R-KV，但分别改变了存储层级、注意力读取方式、生成长度或智能体任务设置，因此不能与纯 KV 淘汰方法完全横向比较。

| 时间 | 论文与发表情况 | 一句话介绍 |
|---|---|---|
| 2026 年 2 月 | [SideQuest](https://arxiv.org/abs/2602.22603)，arXiv 预印本 | 面向多轮智能体，让模型主动判断历史 KV 是否仍有用；相近压缩率下明显优于 R-KV，但不是单轮数学推理设置。 |
| 2026 年 5 月 | [Not All Thoughts Need HBM](https://arxiv.org/abs/2605.09490)，arXiv 预印本 | 将低重要性 KV 移到 CPU 而不是直接删除，准确率明显高于 R-KV，但主要节省 GPU HBM，并未压缩全部存储。 |
| 2026 年 7 月 | [Locks](https://arxiv.org/abs/2607.24555)，arXiv 预印本 | 为每个 KV 页构建独立低秩摘要，只读取高注意力质量的页面；MATH-500 小预算下大幅领先 R-KV，但完整 KV 仍驻留内存，主要减少读取带宽。 |
| 2026 年 8 月 | [ReCo](https://arxiv.org/abs/2608.04771)，arXiv 预印本 | 用过程奖励联合控制 KV 压缩、反思 token 和提前结束，同时减少缓存与生成长度，因此不是纯缓存方法。 |
| 2026 年 8 月 | [CommitKV](https://arxiv.org/abs/2608.07855)，arXiv 预印本 | 根据工具调用前后的“提交状态”淘汰已经完成的智能体事件；2048 预算下平均分为 $38.45$，R-KV 为 $16.21$，但任务是多轮工具智能体。 |

## 6. 如何选择后续基线

如果继续研究 R-KV 这一小方向，建议优先选择以下基线：

- **训练无关的直接基线**：R-KV、LazyEviction、TriAttention、VaSE、ReasonAlloc。
- **训练型方法**：RLKV、ForesightKV。
- **量化与淘汰联合方法**：ThinKV。
- **结构或预算分配插件**：AMS、ReasonAlloc，可与已有 token scorer 组合。
- **系统方向扩展**：Locks、多级 HBM/CPU 存储方法。
- **智能体方向扩展**：SideQuest、CommitKV。

总体来看，R-KV 奠定了“重要性与冗余联合建模”的训练无关基线；后续方法主要沿着五条路线推进：观察更长时间尺度的注意力、学习长期贡献、非均匀分配层/头预算、保护 Value 异常状态，以及从纯淘汰扩展到量化、分层存储和生成过程联合优化。

## 7. 阅读优先级

1. **R-KV**：理解问题定义和冗余感知基线。
2. **LazyEviction**：理解注意力重要性回归及训练无关改进。
3. **TriAttention**：理解 pre-RoPE 几何信息为何比短期注意力窗口稳定。
4. **VaSE**：理解 Value 状态和随机性对长推理稳定性的影响。
5. **ReasonAlloc**：理解层级、头级预算不均匀性。
6. **RLKV / ForesightKV**：进一步考察训练或强化学习能否学习更可靠的长期 KV 价值。

## 8. 注意事项

- “更好”必须在相同模型、数据集、生成长度、缓存预算和预算统计方式下判断，不能只比较论文摘要中的最高数字。
- ThinKV 同时使用量化和淘汰；Locks 保留完整 KV 但减少读取；多级存储方法把 KV 转移到 CPU；ReCo 同时缩短推理链，它们与 R-KV 的资源口径并不完全一致。
- 2026 年的新论文中多数仍是 arXiv 预印本，其结果需要关注后续版本、正式评审和独立复现。
- LongFlow、CASK、EpiKV、LessIsMore、DELTA 等工作虽然相关，但没有在统一口径下显示出相对 R-KV 的稳定总体优势，因此没有放进直接更强列表。

## 9. 正式 CCF-A 论文中引用 R-KV 的代表性工作

下面列出 7 篇已经正式发表于 CCF-A 主会、并在正文或参考文献中明确引用 R-KV 的论文。它们不都属于“直接改进 R-KV”：有些将其作为实验基线，有些借鉴其冗余感知思想，还有些把压缩后的 KV Cache 用于训练或系统优化。

### 9.1 ThinKV：按推理阶段自适应压缩 KV Cache

- 论文：[ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](https://arxiv.org/abs/2510.01290)
- 发表：ICLR 2026 Oral，CCF-A
- 核心思路：ThinKV 根据注意力稀疏模式识别 Chain-of-Thought 中不同类型、不同重要程度的“思考阶段”，再联合使用量化与淘汰。重要阶段保留更高精度和更多 token，较不重要的阶段则逐步降低精度或被淘汰；其 PagedAttention 扩展还可复用被淘汰 token 留下的显存槽位，减少压缩时的数据搬移。
- 与 R-KV 的关系：论文将 R-KV 作为面向长推理解码的代表性 token-level eviction 基线，并指出 R-KV 虽然联合考虑重要性与冗余性，但没有显式利用更高层次的推理阶段结构。

### 9.2 Cache What Lasts：学习 token 的长期保留价值

- 论文：[Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs](https://arxiv.org/abs/2512.03324)
- 发表：ICLR 2026 Poster，CCF-A
- 核心思路：该工作提出 TRIM-KV，在 token 生成时通过轻量 retention gate 预测其长期价值，并让该价值随时间衰减。当缓存超过预算时，模型根据学习到的分数淘汰 token，而不是完全依赖短期注意力或人工规则。训练时只优化小型 gate，原始 LLM 参数保持冻结。
- 与 R-KV 的关系：R-KV 被用作启发式 KV 淘汰基线。TRIM-KV 试图用学习得到的长期价值取代 R-KV 的手工重要性与冗余性评分，在低缓存预算下获得更稳健的保留策略。

### 9.3 Not All Bits Are Equal：推理模型的显存应如何分配

- 论文：[Not All Bits Are Equal: How Model Scale Changes Memory-Optimal Reasoning](https://arxiv.org/abs/2510.10964)
- 发表：ICLR 2026 Poster，CCF-A
- 核心思路：这是一项大规模经验研究，系统比较模型大小、权重量化精度、生成 token 数、并行采样数量和 KV Cache 压缩之间的权衡。其主要结论是：推理模型不存在统一的最优显存配置，小模型更可能受模型容量限制，而较大模型更值得把显存投入更长的测试时计算和 KV Cache。
- 与 R-KV 的关系：论文把 R-KV 作为 KV eviction 方案，与 FullKV、StreamingLLM 和 KV 量化等策略比较，用它分析不同模型尺度和显存预算下“淘汰 KV”是否优于“量化 KV”。这篇工作主要评估 R-KV 的适用边界，而不是提出 R-KV 的直接替代算法。

### 9.4 KaVa：用压缩 KV Cache 蒸馏隐式推理

- 论文：[KaVa: Latent Reasoning via Compressed KV-Cache Distillation](https://arxiv.org/abs/2510.02312)
- 发表：ICLR 2026 Poster，CCF-A
- 核心思路：KaVa 先让教师模型根据显式 Chain-of-Thought 产生 KV Cache，再将其压缩为较短的连续表示，并以此监督学生模型生成 latent reasoning token。这样，学生在推理时不必输出冗长的自然语言思维链，也能学习教师推理轨迹中的内部状态。
- 与 R-KV 的关系：R-KV 支持了 KaVa 的核心动机——长 Chain-of-Thought 的 KV Cache 含有大量可压缩冗余。KaVa 进一步把“压缩后的 KV 状态”从推理时的存储优化对象变成了知识蒸馏的监督信号，因此两者的目标和使用阶段不同。

### 9.5 FlowCache：把重要性—冗余性压缩扩展到视频生成

- 论文：[Flow Caching for Autoregressive Video Generation](https://arxiv.org/abs/2602.10825)
- 发表：ICLR 2026 Poster，CCF-A
- 核心思路：FlowCache 面向分块自回归视频生成，为处于不同去噪状态的视频块分别决定复用还是重新计算，并对已经生成的历史视频块执行 KV Cache 压缩。其选择准则同时考虑历史状态的重要性和相互冗余性，以在固定显存预算下维持时间一致性和生成质量。
- 与 R-KV 的关系：FlowCache 明确借鉴了 R-KV 的“重要性 + 冗余性”联合筛选原则，但将应用对象从语言推理 token 改为自回归视频块，并针对视频的时空冗余和去噪过程重新设计缓存管理。

### 9.6 Sparse-RL：在强化学习 rollout 中使用稀疏 KV Cache

- 论文：[Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning](https://aclanthology.org/2026.acl-long.2000/)
- 发表：ACL 2026 Long Paper，CCF-A
- 核心思路：Sparse-RL 在推理模型强化学习的 rollout 阶段压缩 KV Cache，以减少大批量采样的显存开销。由于稀疏 rollout 会使行为策略与训练策略产生偏差，论文进一步设计 off-policy correction，使模型可以利用压缩缓存生成的数据训练，同时尽量保持与密集 FullKV rollout 相近的效果。
- 与 R-KV 的关系：论文分别采用 R-KV 和 SnapKV 实例化稀疏 rollout。实验显示，直接使用 R-KV 进行 rollout 仍可能因策略不一致而导致训练退化，而 Sparse-RL 的校正机制可以恢复性能；因此它解决的是“如何在 RL 训练中可靠使用 R-KV”，而不是修改 R-KV 的 token 评分公式。

### 9.7 LazyEviction：避免过早删除稍后会重新重要的 token

- 论文：[LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning](https://aclanthology.org/2026.acl-long.1683/)
- 发表：ACL 2026 Long Paper，CCF-A
- 核心思路：论文观察到长推理中存在 Token Importance Recurrence：某些 token 会暂时不受关注，但在若干解码步之后重新变得重要。LazyEviction 因此不在每一步立即淘汰，而是利用观察窗口跟踪 token 的重要性复现间隔，再优先删除长期没有重新激活迹象的状态。
- 与 R-KV 的关系：R-KV 是其主要比较基线之一。LazyEviction 指出，R-KV 的当前重要性与冗余性评分仍可能过早移除周期性关键 token；它从时间维度补充了 R-KV 没有显式建模的“未来重新激活”现象。

### 9.8 小结

这 7 篇论文体现了 R-KV 后续影响的几种主要形式：

1. **直接改进或替代 token 淘汰策略**：ThinKV、TRIM-KV、LazyEviction。
2. **评估 R-KV 的资源配置边界**：Not All Bits Are Equal。
3. **把压缩 KV 用作新的训练信号或训练基础设施**：KaVa、Sparse-RL。
4. **将冗余感知原则迁移到其他模态**：FlowCache。

因此，R-KV 的影响不只体现在后续方法是否取得更高准确率，也体现在它把“长推理输出中的 KV 冗余”确立为一个可独立研究和工程化利用的问题。
