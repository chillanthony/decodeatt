# ThinKV（用户所称 THINKV）后续工作调研

检索日期：2026-09-14。本文将用户所称“THINKV”核对为 **ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models**。论文 arXiv:2510.01290 v2（2026-05-07）首页明确标注“Published as a conference paper at ICLR 2026”；论文脚注说明名称也可读作 “Think KV”。

检索范围：arXiv 全文与摘要检索（ThinKV/ThinkKV、reasoning KV cache、thought-aware KV、long-output reasoning）、OpenAlex works/full-text 索引、后续论文 PDF 的参考文献与正文核对。由于 ThinKV 公开时间较晚，OpenAlex 在检索日返回的结构化 cited-by 数为 0，不能据此断言“没有后续工作”；下文将直接引用与仅做相关比较严格分开。论文标题均链接到可核验的 primary source。

## 1. ThinKV 做了什么

### 问题

推理模型的 chain-of-thought（CoT）输出很长，decode 阶段 KV cache 随生成长度线性增长。论文以 GPT-OSS-20B 生成约 32K token、batch size 32 为例：KV cache 约 50 GB，加上约 40 GB 权重会超过 80 GB A100 显存。传统方法的两个失配是：

1. 统一量化会让推理轨迹变长（量化误差导致额外重试/循环），抵消节省的 cache；统一或 token-level eviction 又会删掉对后续推理关键的 token。
2. 非连续 eviction 需要 gather/compaction，造成 HBM 带宽与 TPOT（time per output token）开销；PagedAttention 的连续物理内存没有被充分利用。

### 方法与证据

- **Thought decomposition**：用归一化 attention 的稀疏率（离线 KDE 校准阈值，在线每 $\tau=128$ 步刷新）把 CoT 粗分为 reasoning（R）、execution（E）、transition（T）三类。经验重要性大致为 $R>E>T$，但少量高重要性的 T（回溯/转折）不能删除。
- **TBQ（Think Before you Quantize）**：按 thought type 分配精度，而非所有 token 统一 bit-width。
- **TBE（Think Before You Evict）**：利用 thought 间的依赖随 transition 逐渐衰减这一现象，对不再需要的 thought 逐步 eviction，同时保护异常重要的 T。
- **Continuous Thinking kernel**：扩展 PagedAttention，在逻辑 token 被 evict 后直接复用物理 block，避免 gather compaction；量化/解量化与矩阵乘尽量融合。
- **Hybrid rationale**：量化减少每个 token 的 bytes，eviction 控制 token 数量；论文声称二者形成比单独量化或单独 eviction 更好的 Pareto frontier。

原文在 DeepSeek-R1-Distill、GPT-OSS、NVIDIA AceReason 上的数学与代码 benchmark 报告：KV cache 保留不到原始 **5%** 时接近无损；相对基线最高 **1.68× 更低 TPOT**、**5.80× 更高吞吐**。这些数字是论文自报结果，应结合模型、预算、batch、kernel 实现解读，不能直接外推为所有硬件/任务的保证。

原文唯一显式 limitation 是：ThinKV 面向 long-output reasoning，对 long-input 为主的场景“不直接适用”。此外，从方法和实验设定还能看到若干隐含限制，见第 5 节。

## 2. 论文与引用链

### 2.1 直接引用 ThinKV 的后续论文

| 论文 | 时间 / 状态 | 如何发展 ThinKV | 证据与链接 |
|---|---|---|---|
| [Epiphany-Aware KV Cache Eviction Without the Attention Matrix (EpiKV)](https://arxiv.org/abs/2606.26472) | 2026-06-25，arXiv preprint | 明确把 ThinKV 作为 reasoning-aware eviction 对照；保留“推理轨迹中远期重要 token”目标，但以 forward hidden-state 的 **epiphany score** 替代 attention matrix，不需要 classifier、训练或 custom kernel，兼容 FlashAttention。MATH-500 4,096 cache 报告 EpiKV 72%、ThinKV 71%；AIME-2024 8,192 cache 的 lag-normalized 版本 37%（最佳对照 33%），最高 2.8× speed。 | 正文 related work、表格和参考文献均出现 ThinKV；论文 PDF 第 1、3、10–14 页可核验。 |
| [KARA: Efficient Reasoning LLM Serving via Sliding-Window KV Cache Compression](https://arxiv.org/abs/2607.01237) | 2026-05-01 v1，2026-07-03 v2，arXiv preprint | 参考文献明确收录 ThinKV。KARA 不采用 thought 分类，而在最近生成窗口内用 bidirectional attention 选取任意位置、可变长度的重要 chunk；KvLLM 用周期触发压缩解决阈值触发造成的并发吞吐反降。报告保留约 20% cache 仍接近 full-KV accuracy，受限显存下平均吞吐提升 12.75%。 | [PDF 参考文献 ThinKV 条目](https://arxiv.org/pdf/2607.01237)；正文第 1、4、6 节。 |

> **引用状态说明。** 截至 2026-09-14，OpenAlex 对 ThinKV 的结构化 cited-by 返回 0，而 EpiKV/KARA 的 PDF 已能看到 ThinKV 条目或正文比较。这反映索引延迟、预印本版本差异或引用图尚未同步，不应把“0”解读为没有学术影响。Semantic Scholar API 在本次检索触发限流，未将其结果作为引用数量证据。

### 2.2 同一问题的并行/后续推进（不一定直接引用）

以下论文在 ThinKV 之后公开，或与其同期，核心是 long-output reasoning 的 KV 管理；它们的 primary source 可用于判断该方向的技术演进。

| 论文 | 核心推进 | 与 ThinKV 的关系 |
|---|---|---|
| [Crystal-KV: Efficient KV Cache Management for Chain-of-Thought LLMs via Answer-First Principle](https://arxiv.org/abs/2601.16986) | 用最终 answer 的 attention 反推 think-stage KV，区分维持流程的 SlipKV 与真正影响答案的 CrystalKV；再用 attention-based LRFU 和自适应 budget 保留后者。 | 与 ThinKV 同为“推理结构感知”，但信号来自 answer-first 反事实/注意力，而不是 R/E/T 稀疏率；名称中的 ThinkKV 是其内部术语，不等于 ThinKV，也未确认直接引用。 |
| [SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning](https://arxiv.org/abs/2602.22603) | 让 LRM 自己判断外部检索 token 的有用性，把 cache 管理作为并行 auxiliary task；仅 215 个训练样本，agentic 任务峰值 token 使用最多降 65%。 | 把 ThinKV 的“模型内部推理结构”扩展到多网页、多轮 agent context；从 hand-designed sparsity 转向 model-driven utility。 |
| [Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning](https://arxiv.org/abs/2605.09490) | 四级存储 HBM/DDR/compressed/evicted；低重要 token 先 offload 到 CPU，attention 前以 full precision 预取，形式化为 zero-approximation-error offloading。 | 质疑“低重要 token 必须永久删除”的前提。ThinKV 优化 GPU 内 cache 与 kernel；该工作把问题拆成 HBM 容量与永久 eviction ratio，强调 offload 可保精度。 |
| [ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning](https://arxiv.org/abs/2605.22106) | 面向 Tree-of-Thoughts 的分支/回溯，value estimator + tree-aware allocation，纯 token-extractive eviction 和 lazy rehydration。 | 将 ThinKV 的线性 CoT thought 结构推广到树结构；inactive subtree 可暂时移出但必须可恢复。 |
| [Adaptive Mass-Segmented KV Compression for Long-Context Reasoning](https://arxiv.org/abs/2605.23200) | 发现 global top-k 会造成 contiguous **Region Wipe-out**；按 attention mass 分段、给每段 quota，EMA 平滑边界，可插入 TOVA、Expected Attention、KeyDiff、R-KV、TriAttention。 | 把“thought block 级别预算”做成与 scorer 解耦的通用层；直接回应 ThinKV 的 R/E/T 三分类过粗和边界抖动风险。 |
| [TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks](https://arxiv.org/abs/2605.17170) | 按 temporal recency × modality × semantic role 三轴打 tag，校准敏感度，在 INT2/INT4 固定预算下混合精度。 | 将 ThinKV 的 thought-type mixed precision 扩展到 agent 的模态、工具调用、用户/模型角色交互；不依赖 R/E/T 假设。 |
| [ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models](https://arxiv.org/abs/2606.11164) | 离线 layer-wise “Reasoning Wave”预分配 + 在线 head-wise utility 重分配，针对 decoding 而非 prefill 的非均匀预算。 | 从 ThinKV 的 token/thought 预算推进到 layer/head 层级；可与 thought-aware eviction 组合。 |
| [Value-Aware Stochastic KV Cache Eviction for Reasoning Models (VaSE)](https://arxiv.org/abs/2606.03928) | 保护大幅值 value outlier；引入随机 eviction 增加 cache diversity，Qwen3 六项推理任务 4× 压缩下平均准确率超过若干 selection/eviction 基线。 | 揭示 ThinKV attention-sparsity 之外的 value-state 风险；随机性可能缓解确定性删错 token 与重复循环。 |
| [ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval](https://arxiv.org/abs/2604.10898) | 将 verbose thought 形成 summary keys，先粗粒度索引，再按 query“zoom in”取细粒度 KV。 | 从“保留/删除”推进到 hierarchical retrieval；可能避免 ThinKV 在高压缩率下只能二元决策。 |
| [KV-Rescue: Recovering Reasoning Language Model KV Eviction Loss via Stepwise Interleaving](https://arxiv.org/abs/2608.15797) | 观察 eviction 造成的是 information gap 而非 capability gap；用轻量 full-context helper 与被压缩模型交错推理，entropy/compressibility detector 提前终止退化轨迹。 | 不是改进 scorer，而是对 ThinKV/R-KV 类 eviction 的错误进行运行时恢复；直接针对“删错后进入重复循环”。 |
| [Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression](https://arxiv.org/abs/2608.01631) | fixed-trace replay 固定可见 reasoning 内容，测 final accuracy、answer-chain consistency、perturbation faithfulness；提出 **answer-evidence gap**。 | 指出 ThinKV 以答案准确率/吞吐为主的评测不足：答案对不代表可见 rationale 仍是证据。 |
| [GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving](https://arxiv.org/abs/2609.03494) | 双时间尺度 query summary 估计工作集，在 capacity boundary 动态压缩或申请新 page，兼容 continuous batching/prefix caching。 | 把 ThinKV 固定预算与固定 refresh 变成 demand-adaptive capacity；关注服务系统的请求间异质性。 |
| [MetaKV: Adaptive KV Cache Compression for Constrained LLM Inference](https://arxiv.org/abs/2609.07966) | 预测候选压缩配置的 latency、peak memory、correctness，在用户约束下逐 prompt 选择配置。 | 将 ThinKV 的单一 Pareto 点推进到 budget-conditioned policy；目前摘要未显示直接引用 ThinKV，属于同问题后续。 |
| [MEMENTO: Teaching LLMs to Manage Their Own Context](https://arxiv.org/abs/2604.09852) | 训练模型分块并生成 memento summary，只 attend mementos；OpenMementos 228K traces，约 2.5× KV 降低、vLLM 吞吐约 1.75×。 | 由 inference-time token eviction 转到 training-time persistent summary；可能从根本上减少对 attention sparsity proxy 的依赖。 |
| [State commitment learning: training language models to distinguish computation from memory](https://arxiv.org/abs/2606.05201) | Counterfactual Erasure RL（CERL）训练模型区分可丢弃 computation 与需持久化 state，定义 persistent-state sufficiency。 | 为 ThinKV 的“thought importance”提供可学习、可反事实验证的语义目标，而不只是稀疏率相关性。 |

### 2.3 重要前序基线（理解后续推进所必需）

ThinKV 自身将下列工作作为对照或组成背景： [H2O](https://arxiv.org/abs/2306.14048)（heavy-hitter + recent window）、[ScissorHands](https://arxiv.org/abs/2305.17118)（attention persistence）、[SnapKV](https://arxiv.org/abs/2404.14469)（prefill attention clustering）、[KIVI](https://arxiv.org/abs/2402.02750)（低比特 KV quantization）、R-KV、LazyEviction、RaaS、[PM-KVQ](https://arxiv.org/abs/2505.18610) 等。R-KV、LazyEviction、RaaS 的 arXiv 标识在 ThinKV v2 参考文献中未直接给出，本文不为其补写未经核验的 URL。ThinKV 的新意不是首次做 eviction 或 quantization，而是把 **thought-level signal + hybrid precision/eviction + memory-layout co-design** 组合到 long-output LRM 场景。

## 3. 后续发展脉络

一句话地形：研究重点正从“按 token 选谁留下”转向“推理结构、存储层级、动态预算和系统调度共同决定何时、以何种精度、在哪一级内存保留信息”。

| 范式 | 内化了什么 | 代表论文 | 仍未解决 |
|---|---|---|---|
| Thought/region-aware selection | CoT 不是平坦 token 序列；块、转折、区域需要不同预算 | ThinKV、Crystal-KV、TAM、AMS | thought 边界和重要性是否跨模型/任务稳定；全局 top-k 的 region wipe-out 仍可能发生 |
| Signal replacement | attention sparsity 只是 proxy，可换成 hidden-state、value outlier、query cluster | EpiKV、VaSE、BeaconKV | 信号与未来依赖的因果关系未统一；不同信号组合的开销/鲁棒性缺少标准协议 |
| Hierarchical/recoverable memory | 不必永久删除：CPU offload、summary、lazy rehydration、multi-granularity retrieval | Not All Thoughts Need HBM、ArborKV、ZoomR、MEMENTO | PCIe/NVLink 传输、预取 miss、summary 误差在真实并发服务中的端到端代价 |
| Dynamic allocation/system co-design | 预算应随 layer/head/request/时间变化，物理 page 也应动态管理 | ReasonAlloc、GrowPage、KARA/KvLLM、MetaKV | 跨请求 SLA、continuous batching、prefix cache、speculative decoding 的统一控制器 |
| Training/objective-level memory | 训练模型区分 transient computation 与 persistent state | State commitment learning、MEMENTO | 训练数据/奖励昂贵；压缩后 rationale faithfulness 与最终答案的关系尚无共识 |

## 4. 核心论文（按对 ThinKV 主线的决定性排序）

1. **EpiKV**：最直接的继承者。它保留 ThinKV 的 long-CoT eviction 目标与对比基线，但移除 attention matrix 依赖，使方法更接近 FlashAttention 生产栈；这正击中 ThinKV 的部署耦合点。
2. **KARA/KvLLM**：把问题从单请求 accuracy/TPOT 推向受限显存下的并发吞吐。其 periodic sliding-window policy 说明“压缩触发时机”与“保留哪些 token”同等重要。
3. **AMS-KV**：对 ThinKV thought-level 分块思想的通用化。它指出即使 scorer 正确，global top-k 仍会整段抹除，因此 quota + EMA 是结构稳定性的补丁。
4. **ReasonAlloc**：将非均匀性从 thought/token 推到 layer/head，补足 ThinKV 固定 $L^*$ 与固定预算的粒度限制。
5. **Not All Thoughts Need HBM**：改变问题表述，实验证明在其设定中精度主要由永久 eviction ratio 决定，而非 HBM 保留比例；这对“激进删除”路线构成重要反例。
6. **VaSE**：发现大幅值 value outlier 与随机 eviction 对稳定性很关键，揭示 attention-sparsity 之外的 failure mode。
7. **ArborKV**：把线性 CoT 推进到 ToT 树搜索和可恢复回溯，测试 ThinKV 结构假设在 branching reasoning 中是否成立。
8. **KV-Rescue**：不要求 eviction scorer 完美，而是用 helper model 运行时恢复 information gap，为极端压缩提供正交路线。
9. **Does Accuracy Equal Evidence?**：建立压缩评测的新问题：答案正确率可能掩盖 rationale 被破坏，直接挑战 ThinKV 主要指标的充分性。
10. **MEMENTO / State commitment learning**：把 token cache 管理上移到训练和表示层，可能是长期范式替代，而非 ThinKV 的小修补。

## 5. 仍然存在的 gap

### G1：thought 分类的跨模型、跨语言、跨任务稳定性

ThinKV 用固定 $T=\{R,E,T\}$、校准层子集 $L^*$ 与 KDE 阈值。论文虽报告了多种 LRM，但仍有三个未闭环问题：

- R/E/T 是描述性标签，分类函数依赖 attention sparsity，尚未证明与“未来 token 的因果依赖”一致；
- 数学/代码任务占主导，agent 工具调用、视觉 token、长文档检索可能需要更多类别（TriAxialKV 的语义角色轴给出旁证）；
- $\tau=128$、校准集规模和分布漂移对线上质量的影响没有统一 scaling law。

### G2：高压缩下的可恢复性与 faithfulness

ThinKV 的“near-lossless accuracy”不等于 reasoning trace 被保真。KV-Rescue 的重复循环现象与 Does Accuracy Equal Evidence? 的 answer-evidence gap 表明，需要同时测：最终答案、步骤一致性、对被删 thought 的反事实敏感性、退化/循环率。当前还缺一个被社区广泛接受的统一指标。

### G3：端到端系统数字与硬件可迁移性

ThinKV 的 Continuous Thinking kernel 是重要贡献，但结果依赖 Triton/PagedAttention、A100/H200、batch size、block size 和压缩比例。以下开销仍缺少跨栈报告：

- FlashAttention-2/3、不同 vLLM 版本、TP/PP、多 GPU NVLink/PCIe；
- continuous batching、prefix caching、speculative decoding、beam/ToT 分支；
- page fragmentation、压缩触发抖动、动态预算下的 tail latency（p95/p99），而非仅平均 TPOT/throughput。

### G4：固定预算不是服务现实

ThinKV 主要以给定 token budget 比较 accuracy/TPOT。真实服务是每请求不同 SLA、上下文长度、并发度和剩余显存；GrowPage、MetaKV、KARA 说明需要联合优化 $\{\text{quality},\text{latency},\text{HBM},\text{DDR traffic}\}$，并处理预算在生成中动态变化。

### G5：长期依赖、树搜索与多轮 agent

ThinKV 的“transition 后旧 thought 影响衰减”对线性 CoT 有效，但 ToT 回溯、工具观察结果、多轮会话中旧信息可能再次激活。ArborKV、BeaconKV、SideQuest 说明需要显式 revisit/retrieval 机制，而非单向 decay eviction。

### G6：训练与部署的权衡

ThinKV 是 training-free，优点是易部署；MEMENTO/CERL 则通过训练获得可压缩表示或 persistent-state 语义。尚不清楚在相同总成本（训练 GPU 小时 + 推理节省）下，何时训练值得，何时 kernel-only 方法更优。

## 6. 可证伪研究机会

| 状态 | 可证伪机会 | 具体成功标准 |
|---|---|---|
| 🟢 | **因果 thought importance + 动态可恢复 cache**：把 attention/hidden/value/query 信号与“未来答案对该块的反事实依赖”联合建模，支持 revisit | 在 DeepSeek-R1/Qwen3/AceReason、数学+代码+agent 三类任务，4×/10× 压缩下，相对 ThinKV/EpiKV/AMS 的平均 accuracy 高 ≥2 个百分点，循环率低 ≥50%，p95 TPOT 不增 >10% |
| 🟢 | **HBM-DDR-压缩三级控制器**：ThinKV eviction、Not All Thoughts Need HBM offload、GrowPage 动态 page 统一调度 | 固定准确率下降 ≤1 个百分点时，相对 ThinKV 将 HBM 峰值再降 ≥30%，并发吞吐提升 ≥20%；报告 PCIe/NVLink 流量和 p99 latency |
| 🟡 | **Faithfulness-aware compression objective**：把 answer-chain consistency/perturbation faithfulness 纳入预算分配 | 在 fixed-trace replay 上，最终答案与 evidence 指标的差距缩小 ≥50%，同时 accuracy 不低于 ThinKV 1 个百分点；风险是 faithfulness 指标与用户真正效用不一致 |
| 🟡 | **树/多轮结构感知的 ThinKV**：R/E/T 之外增加 branch、tool-observation、revisit 状态，支持 lazy rehydration | ToT/agent benchmark 在相同 HBM 与搜索宽度下，成功率至少持平 FullKV 的 95%，并把可搜索深度提高 ≥2×；风险是元数据与预取开销吞噬收益 |
| 🔴 | 只改一个 eviction score、只换一个量化 bit-width、只在 MATH-500 报平均准确率 | 方向已出现 EpiKV、VaSE、AMS、ReasonAlloc 等多条强基线；小改和单点提升难形成 CCF-A 级贡献 |

### P1：因果 thought importance + 动态可恢复 cache

**问题定义。** 给定训练好的 LRM、生成中的 thought blocks $Y_i$ 和显存预算 $B_t$，系统在每个 decode 窗口决定 token 保留、量化 bit-width 或可恢复 offload；测试时允许远期 block 被 revisit。与 ThinKV 的差别是重要性标签不只由 attention sparsity 决定，而由屏蔽 $Y_i$ 后未来答案/步骤分布的变化估计。

**形式化目标。** 令 $p(\cdot\mid S_t)$ 为完整 cache 分布，压缩状态为 $\tilde S_t$，定义块重要性 $I_i=\mathbb{E}_{r}[D_{KL}(p_r\|p_{r,-Y_i})]$；优化
$$
\min_{\pi}\; \mathbb{E}[\text{quality loss}(\pi)] + \lambda_1\,\text{HBM}(\pi)+\lambda_2\,\text{TPOT}(\pi)+\lambda_3\,\text{cycle-rate}(\pi),
$$
约束每步 HBM $\le B_t$，并允许 offload/retrieve 操作。成功不应只看 pass@1。

**评测协议。** 模型至少 DeepSeek-R1-Distill-Llama-8B、Qwen3-8B/14B、GPT-OSS；数据至少 AIME-2024、MATH-500、LiveCodeBench、一个工具调用 agent benchmark；baseline 为 FullKV、H2O、R-KV、LazyEviction、ThinKV、EpiKV、AMS、VaSE。报告 accuracy、answer-chain consistency、perturbation faithfulness、循环率、平均/p95 TPOT、吞吐、HBM/DDR 峰值。目标：4×压缩平均 accuracy ≥ ThinKV +2pp，循环率下降 ≥50%，p95 TPOT 增幅 ≤10%。

### P2：多级内存与动态 page 的统一控制

**问题定义。** 请求 $j$ 有质量下限 $q_j$、延迟 SLA $\ell_j$ 和动态工作集；cache entry 可处于 HBM、DDR、低比特压缩或永久删除。与 ThinKV 固定 cache budget 不同，容量可在 decode 中申请/释放。

**形式化目标。** 在服务时间窗 $T$ 内最小化
$$
\alpha\,\text{HBM}_{peak}+\beta\,\text{p99-latency}+\gamma\,\text{energy},
$$
满足每个请求 $\Pr[\text{correct}]\ge q_j$、OOM=0，并把传输、解量化和 page fragmentation 计入成本。

**评测协议。** 在 A100 80GB、H200、消费级 24GB GPU，vLLM continuous batching，batch 1–256、prompt 2K–32K、generation 4K–64K，比较 ThinKV kernel、KARA/KvLLM、GrowPage、MetaKV、Not-All-Thoughts offload。成功阈值：同等 accuracy（下降 ≤1pp）下 HBM 峰值下降 ≥30%、p99 latency 不恶化、并发吞吐提升 ≥20%。

## 7. CCF-A 投稿标准（本方向具体版本）

1. **Motivation 层级。** 仅“压缩更多”不够。ThinKV 的可发表现象是 R/E/T 稀疏率三模态与 transition 后依赖衰减；EpiKV 的可发表现象是 attention matrix 在生产 kernel 中本身是瓶颈；Does Accuracy Equal Evidence? 则揭示答案准确率与证据保真可分离。新工作至少要实证一种新 failure mode 或服务瓶颈。
2. **方法差异化。** 一句话必须能说清相邻 SOTA 的轴：是“从 attention proxy 转为因果/hidden-state signal”、 “从永久 eviction 转为 recoverable hierarchy”、还是“从固定预算转为 SLA-conditioned controller”。
3. **实验体量 minimum bar。** 至少 3 个模型家族、4 个任务（数学/代码/agent/长上下文之一）、8 个强 baseline（含 ThinKV、EpiKV、KARA、AMS/ReasonAlloc 等）、4 个压缩预算；完整 accuracy + 平均/p95 latency + throughput + HBM/DDR/energy 表。只报 MATH-500 单点不够。
4. **消融与可解释性。** 需要 thought segmentation、score、bit allocation、eviction/offload、kernel/policy 各自消融；画预算-质量 Pareto、attention/hidden/value 重要性相关性、循环案例与跨任务稳定性。ThinKV 的 $\tau$、$L^*$、block size 曲线是应有基线。
5. **写作故事线。** Related Work 要逐篇 contrast ThinKV、EpiKV、KARA、AMS、ReasonAlloc；贡献句应包含新现象、方法差异轴和系统收益；Limitations 需正面说明 long-input、ToT、硬件迁移边界。
6. **会议偏好。**

| 会议 | 偏好 | 本方向命中案例 |
|---|---|---|
| ICLR / NeurIPS | 新现象、训练/推理范式与可解释性 | ThinKV 的 thought decomposition；CERL/MEMENTO 的 persistent state |
| MLSys / 体系方向 | kernel、memory hierarchy、端到端吞吐与 tail latency | ThinKV Continuous Thinking、KARA/KvLLM、GrowPage |
| ICML | 可泛化的优化/预算分配与严谨评测 | ReasonAlloc 的 hierarchical allocation、MetaKV 的 constrained selection |
| ACL/EMNLP | reasoning/agent 任务与 rationale 质量 | SideQuest、Does Accuracy Equal Evidence?、MEMENTO |
| KDD/AAAI/IJCAI | 实际工作负载、agent 与资源约束 | ArborKV、TriAxialKV、三级 memory controller |
