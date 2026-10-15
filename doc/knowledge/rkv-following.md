# R-KV 及后续推理模型 KV Cache 压缩方法调研

> 整理时间：2026 年 8 月 21 日
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

## 4. 后续方法：哪些工作真正推进了 R-KV

这里采用严格口径：面向长推理解码、直接以 R-KV 为基线，并在相同或更小 KV 预算下取得准确率或效率优势。按技术脉络看，17 篇代表工作主要在解决 R-KV 的三个局限：**评分只反映过去、预算分配过于均匀、冗余计算本身较重**。

| 改进方向 | 方法（首次公开 / 发表） | 相对 R-KV 的关键信息 |
|---|---|---|
| 建模重要性的时间变化 | [LazyEviction](https://arxiv.org/abs/2506.15969)（2025.06，ACL 2026） | 延迟淘汰以观察重要性回归；MATH-500、50% 压缩下部分模型为 $75.2$，R-KV 为 $73.2$。 |
|  | [G-KV](https://arxiv.org/abs/2512.00504)（2025.11，预印本） | 融合局部注意力与历史衰减分数，在多数预算、尤其极小预算下优于 R-KV。 |
|  | [Crystal-KV](https://arxiv.org/abs/2601.16986)（2026.01，预印本） | 区分有助最终答案的 CrystalKV 与临时 SlipKV；代码、数学任务平均高约 $12.46$、$7.97$ 个百分点。 |
| 学习或预测未来价值 | [DynTS](https://arxiv.org/abs/2601.18383)（2026.01，ICML 2026） | 用答案对思考 token 的注意力训练预测器；六个基准同预算平均 Pass@1 高 $2.6$ 个百分点。 |
|  | [ForesightKV](https://arxiv.org/abs/2602.03203)（2026.02，ICML 2026） | 学习长期贡献；Qwen3-4B、AIME24 上 1K cache 为 $54.5$，R-KV 用 2K 为 $44.8$。 |
|  | [TriAttention](https://arxiv.org/abs/2604.04921)（2026.04，ICML 2026） | 用 pre-RoPE 几何预测长期价值；Qwen3-8B、AIME25、2K 预算下为 $32.9$，R-KV 为 $17.5$。 |
|  | [BeaconKV](https://openreview.net/forum?id=bgAbffQCrL)（2026.05，ICML 2026） | 用代表性 query 簇保护将被远距离重访的 token；多模型、多任务曲线总体领先 R-KV。 |
| 非均匀分配缓存 | [RLKV](https://arxiv.org/abs/2510.08525)（2025.10，ICML 2026） | 用强化学习识别关键推理头；一项 40% 稀疏度实验为 $84.6$，R-KV 为 $77.8$。 |
|  | [KARA](https://arxiv.org/abs/2607.01237)（2026，预印本） | 近期双向注意力结合 Token2Chunk；逐样本匹配实际预算后总体更优，少数配置接近或略低于 R-KV。 |
|  | [DBTrimKV](https://arxiv.org/abs/2605.09649)（2026.05，预印本） | 让层、头和模态在全局预算中竞争；128-token 预算下为 FullKV 的 $104.10\%$，R-KV 等为 $60$–$64\%$。 |
|  | [AMS](https://arxiv.org/abs/2605.23200)（2026.05，预印本） | 用区域配额保护连续推理块；作为插件将 R-KV 在 MATH-500、512 预算下从 $46.4$ 提至 $53.6$。 |
|  | [ReasonAlloc](https://arxiv.org/abs/2606.11164)（2026.06，预印本） | 动态分配层级和头级预算；MATH-500、512 预算下为 $82.50$，R-KV 为 $76.48$。 |
| 降低冗余评分代价或增强多样性 | [R-KVHash](https://openreview.net/forum?id=UTRuEFJ57H)（2026.03，ICLR 2026 Workshop Oral） | 用 SimHash/LSH 近似冗余度，准确率有竞争力，解码吞吐最高约为 R-KV 的 $2\times$。 |
|  | [VaSE](https://arxiv.org/abs/2606.03928)（2026.06，预印本） | 保护大幅值 Value 并随机淘汰以维持多样性；$4\times$ 压缩下平均高 $4.4$–$4.9$ 个百分点。 |
| 联合量化或轻量训练 | [ThinKV](https://arxiv.org/abs/2510.01290)（2025.10，ICLR 2026 Oral） | 按思考阶段联合量化与淘汰；不足 5% 原始 KV 时接近无损，吞吐最高约为 R-KV 的 $5.8\times$。 |
|  | [Fast KVzip](https://arxiv.org/abs/2601.17668)（2026.01，预印本） | 为冻结模型训练轻量 gate；固定长度解码下总体优于 R-KV，4K 左右预算接近无损。 |
| 重新审视基础淘汰策略 | [SnapKV-D](https://arxiv.org/abs/2512.12008)（2025.12，NeurIPS 2025 Workshop） | 在 8 个推理集、128–512 预算下总体优于 R-KV，并指出极小预算可能反而拉长推理链。 |

**核心趋势**：后续工作不再只优化“当前哪些 token 重要”，而是转向预测未来重访、按层/头/区域分配预算，并降低评分与缓存管理的实际开销。

下列方法改变了存储位置、读取方式、生成轨迹、任务形态或模型配置。它们可以作为扩展方向，但不能仅凭准确率或吞吐数字宣称替代 R-KV。

| 改变的口径 | 方法 | 与 R-KV 的主要差异 |
|---|---|---|
| 生成长度 | [SkipKV](https://arxiv.org/abs/2512.07993)（MLSys 2026） | 删除语义重复句并抑制冗余生成，同时减少 KV 和生成 token 数。 |
| 生成长度与控制策略 | [ReCo](https://arxiv.org/abs/2608.04771) | 联合控制 KV 压缩、反思 token 与提前结束。 |
| 完整 KV 仍被保存 | [FASA](https://arxiv.org/abs/2602.03152)（ICLR 2026） | 通过 CPU offload 或 GPU 上的完整 cache 做稀疏读取；10% 上下文预算报告 $86.4\%$。 |
|  | [Not All Thoughts Need HBM](https://arxiv.org/abs/2605.09490) | 将低重要性 KV 移至 CPU，节省 HBM 而非总存储。 |
|  | [Locks](https://arxiv.org/abs/2607.24555) | 用低秩页摘要筛选读取页，主要降低带宽。 |
| 系统实现 | [Zipage](https://arxiv.org/abs/2603.08743) | 将 R-KV/G-KV 接入 PagedAttention、连续批处理和异步压缩；冗余计算由 $O(N^2b^2)$ 降至 $O(Nb^2)$。 |
|  | [HARD-KV](https://arxiv.org/abs/2606.28831)（ICML 2026） | 在 scorer 外加入逐头动态预算和三级缓存布局，固定预算下也能增强 R-KV。 |
| 多轮智能体 | [SideQuest](https://arxiv.org/abs/2602.22603) | 由模型主动判断多轮历史是否仍有用。 |
|  | [CommitKV](https://arxiv.org/abs/2608.07855) | 按工具调用的“提交状态”淘汰已完成事件。 |
| 额外模型 | [KV-Rescue](https://arxiv.org/abs/2608.15797) | 引入保留完整上下文的 1.5B helper，恢复采用 R-KV 的大模型所丢信息。 |

## 5. 正式 CCF-A 论文如何继承 R-KV

截至整理时间，共确认 17 篇正式 CCF-A 主会论文在正文、实验或参考文献中引用 R-KV；此外补充 5 篇未引用 R-KV 但主题相关的工作。为避免与第 4 节重复，这里只回答两个问题：**它们与 R-KV 有什么关系，以及证据关系有多强。**

关系分为四类：**直接比较**表示实验中以 R-KV 为基线；**使用/改造**表示把 R-KV 用作组件或迁移其思想；**仅引用**表示论文引用了 R-KV，但没有同口径实验；**未引用但相关**表示研究问题相近，但论文未明确引用 R-KV。后两类均不能据此判断与 R-KV 的胜负。

| 关系 | 论文（发表） | 对 R-KV 的继承或扩展 |
|---|---|---|
| 直接比较 | [ThinKV](https://arxiv.org/abs/2510.01290)（ICLR 2026 Oral） | 从 token 级评分提升到思考阶段级的量化与淘汰。 |
| 直接比较 | [Cache What Lasts / TRIM-KV](https://arxiv.org/abs/2512.03324)（ICLR 2026） | 用轻量 gate 学习 token 的长期保留价值，替代手工评分。 |
| 直接比较 | [Not All Bits Are Equal](https://arxiv.org/abs/2510.10964)（ICLR 2026） | 将 R-KV 作为资源配置选项，研究模型规模、量化、生成长度与 KV 淘汰的权衡。 |
| 直接比较 | [LazyEviction](https://aclanthology.org/2026.acl-long.1683/)（ACL 2026） | 建模 token 重要性的周期性回归，避免过早淘汰。 |
| 直接比较 | [FASA](https://arxiv.org/abs/2602.03152)（ICLR 2026） | 用 RoPE 主导频率块做 query-aware 选择，但采用 offload 或完整 cache 稀疏读取。 |
| 直接比较 | [DynTS](https://arxiv.org/abs/2601.18383)（ICML 2026） | 学习对最终答案真正关键的思考 token。 |
| 直接比较 / 组件 | [RLKV](https://arxiv.org/abs/2510.08525)（ICML 2026） | 先识别关键推理头，再对非关键头采用 R-KV 等压缩策略。 |
| 直接比较 | [ForesightKV](https://arxiv.org/abs/2602.03203)（ICML 2026） | 用未来注意力监督和 GRPO 学习长期贡献。 |
| 直接比较 | [TriAttention](https://arxiv.org/abs/2604.04921)（ICML 2026） | 用 pre-RoPE 几何预测长期 KV 价值。 |
| 直接比较 | [BeaconKV](https://openreview.net/forum?id=bgAbffQCrL)（ICML 2026） | 用代表性 query 簇预判远距离重访。 |
| 使用/改造 | [HARD-KV](https://arxiv.org/abs/2606.28831)（ICML 2026） | 将 R-KV scorer 接入逐头 Top-$p$ 预算与三级缓存布局。 |
| 使用/改造 | [Sparse-RL](https://aclanthology.org/2026.acl-long.2000/)（ACL 2026） | 用 R-KV 生成稀疏 rollout，并通过 off-policy correction 修正训练偏差。 |
| 思想迁移 | [FlowCache](https://arxiv.org/abs/2602.10825)（ICLR 2026） | 将“重要性 + 冗余性”迁移到自回归视频块缓存。 |
| 概念扩展 | [KaVa](https://arxiv.org/abs/2510.02312)（ICLR 2026） | 将压缩 KV 从推理时存储对象变成隐式推理蒸馏信号。 |
| 仅引用 | [SeerAttention-R](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b56d827a2b8433517e722e0272c7f464-Abstract-Conference.html)（ICLR 2026） | 学习每步读取哪些 KV block；完整历史仍保留。 |
| 仅引用 | [LouisKV](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html)（ICLR 2026） | 将永久删除改为 CPU 上可召回的语义簇检索。 |
| 仅引用 | [LessIsMore](https://icml.cc/virtual/2026/poster/61079)（ICML 2026） | 跨头共享稀疏读取集合，但未给出同资源口径的 R-KV 对照。 |
| 未引用但相关 | LongFlow、CASK、EpiKV、ShotKV、DELTA | 均涉及长推理中的 KV 压缩、淘汰或稀疏注意力，但未明确引用 R-KV，也未在统一口径下稳定优于 R-KV。 |

总体而言，R-KV 的影响沿着三层扩散：算法层从历史启发式转向未来价值预测，资源层从统一 token 预算转向层/头/阶段级分配，系统层从永久淘汰扩展到量化、分层存储、稀疏读取和训练基础设施。第 4 节用于判断“谁在同口径下更强”，本节则用于判断“R-KV 的思想被如何继承”，二者不应混用。

## 7. 后续实验如何选基线

先固定 **FullKV、R-KV 和一个通用淘汰基线**，再按研究问题补充方法，避免堆叠大量不可比结果。

| 研究问题 | 建议基线 |
|---|---|
| 训练无关的 token 淘汰 | LazyEviction、SnapKV-D、TriAttention、BeaconKV、VaSE |
| 层/头/区域预算分配 | RLKV、AMS、ReasonAlloc；其中 AMS、ReasonAlloc 可与现有 scorer 组合 |
| 学习长期价值 | Fast KVzip、DynTS、ForesightKV、DBTrimKV |
| 量化与淘汰联合 | ThinKV |
| scorer 或实现开销 | R-KVHash、Zipage、HARD-KV |
| 分层存储或稀疏读取 | FASA、Locks、HBM/CPU offload 方法 |
| 生成过程或智能体 | SkipKV、ReCo、KV-Rescue；SideQuest、CommitKV |

若目标是做 **R-KV 的训练无关直接改进**，最小而有区分度的组合是：FullKV、R-KV、SnapKV-D、TriAttention、BeaconKV、VaSE，再根据创新点加入 AMS 或 ReasonAlloc。
