# 长推理 Decoding 阶段稀疏注意力 — 核心论文分析与研究机会

## 研究方向定位

推理模型(DeepSeek-R1、QwQ、o1、Claude thinking 等)将 LLM 的算力分布从 prefilling 转移到 long-CoT decoding,decoding 阶段的注意力计算成为新的瓶颈。该方向与传统的"长上下文 prefilling 优化"(MInference、FlexPrefill 等)形成清晰对比:

| 维度 | 长上下文 prefilling | 长推理 decoding |
|------|-------------------|----------------|
| 输入分布 | 已知完整 prompt | 自回归生成,长度未知 |
| 注意力 mask | 可一次性全局优化 | 必须随生成步骤增量更新 |
| 关键 token | 集中在 prompt 中 | 分布于不断增长的 CoT 中 |
| 主要现象 | A-shape, vertical-slash, block-sparse(MInference) | milestone, recurrence, reflection 冗余 |
| 端到端代价 | 仅注意力 FLOPs | 注意力 FLOPs **+ 生成长度膨胀**(Lil 论文) |

---

## 核心论文识别(8 篇)

从 28 篇候选中识别 8 篇定义该方向的核心工作,按"模式刻画 / 算法 / 系统"三类分布:

### A. 模式刻画类(理解 long CoT 注意力是什么样的)

**1. RaaS** ([2] in decodeatt-papers.md) — *Milestone token 模式*
- **核心贡献**:首篇形式化 CoT decoding 注意力模式。提出"中间引理 token 在被使用前持续高注意力,使用后骤降"的 milestone 现象,并据此设计 O(L) 时间 + O(L) 内存的稀疏化。
- **不可替代性**:目前唯一对推理任务的"使用-淘汰生命周期"做时间维刻画的工作。

**2. LazyEviction** ([5]) — *Token Importance Recurrence(TIR)*
- **核心贡献**:发现注意力重要性的**复发**现象——token 在被冷落数千步后再次被关注(典型场景:模型回头检索早期推理步骤)。提出 MRI(Maximum Recurrence Interval)量化指标。
- **不可替代性**:与 RaaS 的"一次性 milestone"互补,刻画"重要性的非单调演化"。

**3. R-KV** ([3]) — *Reflection 冗余分析*
- **核心贡献**:首次定量验证 reflection、回溯导致的语义冗余,基于 key 相似度去重 + 注意力打分联合决策,**16% KV 反超 100% KV 的精度**——揭示"压缩本身有降噪效果"。
- **不可替代性**:推理特有的语义冗余维度,DuoAttention/Quest 等通用 KV 优化未触及。

### B. 算法类(基于模式如何设计稀疏)

**4. SeerAttention-R** ([1]) — *自蒸馏门控锚点工作*
- **核心贡献**:首个面向 long-reasoning decoding 的稀疏注意力训练框架。轻量门控 + 0.4B token 自蒸馏,90% 稀疏度下 **9× over FA-3**。
- **不可替代性**:事实上的"训练时 decoding 稀疏"基准方法,后续工作均以其为对照。

**5. Quest** ([16]) — *Decoding KV 选择基线*
- **核心贡献**:Page-level min/max 上界 + query-aware top-K 选择。decoding 稀疏注意力的**事实标准基线**,几乎所有后续工作都以其为对照。

**6. NSA** ([12]) — *训练时稀疏的代表 + decoding 加速*
- **核心贡献**:三类稀疏(compression + selection + sliding window)端到端可训练。**11.6× decoding 加速 at 64k**,ACL 2025 best paper。
- **不可替代性**:训练时稀疏注意力的代表,与本方向"通过自蒸馏微调"路线形成"全训练 vs 轻量训练"对照。

### C. 头功能 / KV 差异化类(头粒度优化基础)

**7. DuoAttention** ([17]) — *头功能分类奠基*
- **核心贡献**:首次系统提出注意力头的"检索 vs 流式"功能分类,为头粒度差异化压缩奠定基础。
- **不可替代性**:任何"头功能图谱"方向工作的必需前置基线。

**8. RLKV** ([8]) — *RL 引导的推理关键头识别*
- **核心贡献**:发现"推理关键头 ≠ 检索头",直接以推理结果质量为奖励信号识别推理关键头。
- **不可替代性**:把头功能从"通用检索能力"细化到"推理任务质量贡献",对头分类提供新维度。

### 备选核心(候补)

- **Lil**([9]) — 揭示稀疏负效应"生成长度膨胀",任何端到端实验设计必须考虑。
- **DMS**([24]) — 把 KV 压缩与 test-time scaling 协同,改变了"压缩 = 精度损失"的评估范式。

---

## 研究机会分析

### 现状总结(2024 → 2026 的演进)

1. **2024**:Quest、DuoAttention、Ada-KV、SnapKV 等定义"长上下文 KV 选择"的通用框架,主要面向 prefilling + 短 decoding。
2. **2025 Q1-Q2**:NSA、MoBA、SeerAttention-R 把稀疏注意力推向训练时 / decoding 时;RaaS 首次形式化推理任务的 decoding 注意力模式。
3. **2025 Q3-Q4**:R-KV、ThinKV、LazyEviction、LessIsMore、DELTA、RLKV、Lil 等井喷式涌现,推理模型专用 KV 优化成为热点。
4. **2026 初**:Crystal-KV、Hold Onto That Thought 等开始反思和系统评估,DeepSeek-V3.2 DSA 工业落地。

### 已被部分覆盖的子方向(谨慎进入)

- **基础 KV 选择**:Quest 系延伸已饱和。
- **检索头识别**:DuoAttention、HeadKV、Retrieval Head 充分覆盖。
- **冗余去除**:R-KV、ThinKV、Crystal-KV 已系统覆盖。

### 研究机会


---

#### 机会 1 ★★★★★ 长 CoT 注意力模式的统一系统刻画与"理论指导稀疏 mask"

**核心动机**
现有"模式刻画"工作各自侧重一个现象:RaaS 关注 milestone(单次生命周期),LazyEviction 关注 recurrence(重要性回访),R-KV 关注 reflection 冗余,ThinKV 关注 thought-type。但**没有任何工作对 long CoT 的完整注意力结构做统一系统刻画**,各方法的稀疏 mask 都基于经验式启发,缺乏理论依据。

**研究问题**
1. 推理 decoding 与 prefilling 的注意力模式在哪些维度上有本质差异?
2. long CoT 中"思考-验证-回退"如何统一映射到注意力结构?
3. 是否存在统一的"reasoning attention pattern taxonomy"?
4. 能否基于该 taxonomy 设计**理论指导的、无需自蒸馏的稀疏 mask**?

**方法路径**
- **第一步(纯分析,低算力)**:在 R1-Distill、QwQ 等开源模型上系统采集 8K-32K 长度的 CoT,可视化注意力矩阵,统一标注 4-6 类模式:reasoning sink / milestone / recurrence / reflection-redundant / step-locator / answer-anchor。
- **第二步(模式 → mask 的映射)**:为每类模式设计对应的稀疏算子,如 milestone → 时间戳淘汰,recurrence → 滞后窗口,reflection → 语义去重,step-locator → 逻辑步骤分块。
- **第三步(组合方案)**:稀疏 mask + 门控 + α-entmax(函数稀疏)三类机制混合,构造"模式无关 + 模式特化"两级架构。
- **第四步(无需自蒸馏验证)**:理论分析每类模式的稀疏化误差界,直接 zero-shot 应用,与 SeerAttention-R 的自蒸馏路线对照。

**创新点**
- **完整 taxonomy** 是空白。现有工作都是某一个现象的局部分析。
- **理论指导稀疏**(而非经验自蒸馏)是范式上的差异。
- **mask + 门控 + entmax** 三类机制的统一组合首次出现。

**算力估计**
- 模式分析:1-2 张 A800 一周。
- 算法验证:4-8 张 A800 两周(LoRA 适配,无需预训练)。
- **总预算 ≤ 50 A800·天**

**投稿契合度**
- ICLR / NeurIPS:方法+分析双线,极契合(参照 ASEntmax 的 ICLR 2026)。
- ICML:若强化理论分析,契合度高。
- ACL:推理任务+长生成应用线,可作第二选择。
- AAAI:相对最稳但影响力低,可作保底。

**风险**
- ★ 此方向最大的风险是"taxonomy 讲故事"易被审稿人质疑分类的客观性。需配套**定量指标**(每类模式的占比、跨模型一致性、消融贡献)。
- ★ 时间窗口:RaaS、LazyEviction、ThinKV 各占了一个角度,半年内可能有人把它们组合,需要快。

---

#### 机会 2 ★★★★ Reasoning Sink 现象的存在性、机制与利用

**核心动机**
Gated Attention(NeurIPS 2025 best paper)系统揭示标准 SDPA 的"attention sink"现象,Streaming-LLM 也利用了此现象。**但推理模型中 sink 现象是否仍然存在?其机制和位置是否变化?是否能利用推理特有的 sink 设计稀疏化?**

**研究问题**
1. 推理模型(R1-Distill、QwQ、Gated Attention 训练的 Qwen3-Next)中 sink 现象是否仍然存在?
2. 长 CoT 中是否存在新的"reasoning sink"——稳定承担推理锚点的 token?
3. Reasoning sink 与传统 sink(BOS、首 token)的位置、动力学差异?
4. 如何利用 reasoning sink 简化稀疏 mask 设计?

**方法路径**
- 在多个推理模型上做 sink 探测,可视化 token-level attention 集中度。
- 与 Gated Attention 的"无 sink"模型对比,验证 sink 是 architectural artifact 还是 reasoning-induced。
- 如果发现 reasoning sink,设计"sink-aware sparse mask":sink 永久保留 + 其他位置稀疏化。

**创新点**
- 此问题完全是空白,基础分析价值高。
- 与 Gated Attention 形成对话(他们消除 sink,本工作探究推理诱导的新 sink)。

**算力估计**:< 20 A800·天(以分析为主)。

**投稿契合度**:ICLR / ACL 适合"现象发现"类工作。

**风险**:可能发现 reasoning sink 不存在或与传统 sink 没有本质差异,沦为否定性结果。但即使如此也有 short paper 价值。

---

#### 机会 3 ★★★★ 注意力模式的 RL-aligned 稀疏化(扩展 RLKV 的方向)

**核心动机**
RLKV 用 RL 优化"哪些头"的预算分配,但稀疏 mask 本身仍是手工设计。能否用 RL **直接**优化稀疏 mask 的结构(块大小、Top-K vs Top-p、recurrence window 等),让稀疏化与推理质量端到端对齐?

**研究问题**
- 不同推理任务(数学、代码、定理证明)的最优稀疏 mask 是否不同?
- RL-aligned 稀疏与自蒸馏稀疏(SeerAttention-R)的差异?

**方法路径**
- 参数化稀疏 mask 的若干超参数(块大小、Top-K、TIR 窗口),用 RL/CISPO 优化。
- 与 RLKV、SeerAttention-R 对照。

**算力估计**:50-100 A800·天(RL 训练较重)。

**投稿契合度**:ICLR / NeurIPS。

**风险**:RL 训练稳定性,且与 RLKV 增量需要画清。

---

#### 机会 4 ★★★★ 推理阶段感知的动态稀疏注意力(Phase-Aware Dynamic Sparsity)

**核心动机**
Long CoT 并非均质序列,而是由 **探索(exploration)→ 演算(derivation)→ 验证(verification)→ 回溯(backtracking)→ 收尾(conclusion)** 等阶段交替组成,不同阶段对历史上下文的依赖结构截然不同(回溯阶段需回访早期步骤——对应 LazyEviction 的 recurrence;演算阶段以局部 milestone 为主——对应 RaaS)。现有方法用**一套固定的稀疏策略贯穿整个生成**,必然在某些阶段过度稀疏(损精度)、在另一些阶段稀疏不足(损效率)。ThinKV 做了 thought-type 分类但用于 KV 预算分配,**没有把"阶段"作为在线信号去切换稀疏算子本身**。

**研究问题**
1. Long CoT 的推理阶段能否从 decoding 时可得的轻量信号(注意力熵、近期 token 分布、特定 marker token)**在线检测**?
2. 不同阶段的最优稀疏结构(窗口大小、Top-K、是否启用 recurrence 回访)差异有多大?
3. 一个"阶段检测器 + 阶段特化稀疏算子库"的动态调度,能否在同等平均稀疏率下显著优于固定策略?

**方法路径**
- **第一步(分析)**:在 R1-Distill / QwQ 上标注阶段边界,统计各阶段的注意力依赖跨度分布,验证"阶段→最优稀疏"的映射假设。
- **第二步(轻量检测器)**:训练一个 <10M 参数的在线阶段分类器(输入为当前层注意力统计 + 近期 token embedding),逐步输出阶段标签。
- **第三步(动态调度)**:为每个阶段配一个稀疏算子(演算→局部窗口+milestone 淘汰,回溯→开启 recurrence 滞后窗口,验证→保留 answer-anchor),按检测结果在线切换。
- **第四步**:与 SeerAttention-R(固定自蒸馏)、Quest(固定 Top-K)在同等平均预算下对照精度/延迟。

**创新点**
- 把"推理阶段"从离线分类(ThinKV)升级为**在线信号驱动的稀疏策略切换**,是控制粒度上的新维度。
- 直接复用机会 1 的模式 taxonomy 做算子库,二者可协同。

**算力估计**:30-50 A800·天(检测器训练轻,主要在评测与算子调优)。

**投稿契合度**:ICLR / AAAI(方法清晰、可复现、中等创新,AAAI 契合度高);ACL 作应用线第二选择。

**风险**:阶段边界本身模糊,检测器误差会传导到稀疏决策;需证明动态切换的收益大于切换开销与误检损失。

---

#### 机会 5 ★★★★ 稀疏化的"安全边界"——对自我纠正与回溯能力的影响(Safe-Sparsity Boundary)

**核心动机**
现有稀疏方法几乎都以**最终答案准确率**为唯一指标,但推理模型的核心能力是 **自我纠正(self-correction)与回溯**——发现错误后回头修正。**稀疏化(尤其淘汰早期 KV)很可能优先损害这一能力:被淘汰的恰恰是"出错的早期步骤",模型再也无法回访并纠正它。** 这一因果链目前**完全没有被隔离研究**:答案准确率下降究竟来自"算错"还是"无法回溯纠错"?是否存在一个"安全稀疏边界",越过它自我纠正能力骤降而平均准确率尚未明显下降(滞后崩溃)?

**研究问题**
1. 在可控的"含错-需纠正"推理样本上,稀疏率与**自我纠正成功率**的关系曲线是什么形状?是否存在拐点(安全边界)?
2. 不同稀疏策略(滑窗 vs Top-K vs recurrence-aware)对回溯能力的损害是否不同?recurrence-aware(LazyEviction 思想)能否显著抬高安全边界?
3. 能否设计一个**廉价的"回溯保护"机制**(对疑似错误步骤的 KV 给予淘汰豁免),在几乎不增加预算下守住自我纠正能力?

**方法路径**
- **诊断集构造**:用注入错误 + 标注纠正点的数学/代码推理样本(可基于现有 benchmark 改造),把"是否成功回溯纠错"作为独立可测指标。
- **隔离实验**:固定模型,扫描多种稀疏策略 × 稀疏率,分别记录答案准确率与自我纠正成功率,定位二者的脱钩点(安全边界)。
- **轻量缓解**:基于注意力回访信号(MRI 类指标)识别"高回访概率"的 KV 给予豁免,验证能否以 <5% 额外预算守住边界。

**创新点**
- 首次把**自我纠正能力**作为稀疏化的独立评估维度并定位"安全边界",纠正"只看答案准确率"的评估盲区。
- 给出可直接被后续所有稀疏方法采用的**诊断协议 + 缓解原语**,贡献清晰、结论可操作。

**算力估计**:< 20 A800·天(以分析与评测为主,缓解机制为 training-free)。

**投稿契合度**:**AAAI(强烈推荐,见下)** / ICLR / ACL——现象发现 + 可操作缓解,scope 收敛、严谨、低算力,是 AAAI 的理想画像。

**风险**:诊断集构造需谨慎(错误注入要自然、纠正点标注要客观);若发现安全边界不存在/与普通准确率重合,则退化为否定性结果,但仍有诊断协议的方法论价值。

---

## 综合推荐与时间窗口

| 机会    | 方向                      | 创新性   | 算力             | 时间窗口     | 投稿首选                         |
| ----- | ----------------------- | ----- | -------------- | -------- | ---------------------------- |
| **1** | **统一 CoT 注意力模式 + 理论稀疏** | ★★★★★ | 50 A800·天      | 紧迫(6 个月) | **ICLR 2027 / NeurIPS 2026** |
| 2     | Reasoning sink 探究       | ★★★★  | 20 A800·天      | 充裕       | ICLR / ACL                   |
| 3     | RL-aligned 稀疏 mask      | ★★★★  | 100 A800·天     | 中等       | ICLR / NeurIPS               |
| 4     | 推理阶段感知动态稀疏              | ★★★★  | 30-50 A800·天   | 中等       | ICLR / AAAI                  |
| **5** | **稀疏化安全边界(自我纠正)**       | ★★★★  | **<20 A800·天** | 充裕       | **AAAI(首选)**                 |

---

## CCF-A 投稿标准分析(长推理 decoding 稀疏注意力子方向)

本节不照搬通用模板,所有判断均基于本调研中实际出现的论文做支撑。该子方向 2025 年井喷,arxiv 投稿数远超会议接收容量,审稿人对"小改 + 小提升"的容忍度已经趋近 0,标准比 2024 年的"长上下文 KV 选择"通用方向高出至少一档。

### 1. Motivation 的层级要求

不接受"我把 Quest 的 Top-K 换成 Top-p 涨了 0.5 分"这类增量。本子方向 CCF-A 命中论文的 motivation **必须满足以下至少一条**:

- **新现象/新瓶颈的实证发现** —— 代表:
  - **RaaS**([2]):首次形式化推理 decoding 的 *milestone token 生命周期*——中间引理 token 在被使用前持续高注意力、使用后骤降。这是一个之前 prefilling 文献中**不存在**的现象,直接催生 O(L) 时间内存算法。
  - **LazyEviction**([5]):发现 *Token Importance Recurrence(TIR)*——重要性的**复发**现象,token 冷落数千步后再次被关注,并提出 MRI 量化指标。同样是 prefilling 工作完全没观察到的新瓶颈。
  - **DefensiveKV**([36]):提出 *KV 淘汰脆弱性 / 重要性骤变*,以"最坏情况风险"重塑评估视角。
- **范式级冲突揭露** —— 代表:
  - **R-KV**([3]):**16% KV 反超 100% KV** 的精度结果直接颠覆"压缩 = 精度损失"的默认假设,揭示压缩本身的降噪效果。
  - **Lil**([9]):指出稀疏化触发 *生成长度膨胀* 的负效应,把"端到端代价 = 注意力 FLOPs"的默认范式撕开。
  - **Hold Onto That Thought**([27]):实证揭示 SnapKV / H2O 等通用 KV 压缩在推理任务上的副作用,直接挑战已被广泛接受的方法。
- **工业规模数据证据** —— 代表:
  - **NSA**([12])与 **DSA**([14]):分别在 ACL Best Paper 与 DeepSeek 生产部署上给出"超大规模训练 + 真实端到端 latency 收益"的硬证据,门槛极高,小团队复刻不可行,但作为**对照基线必须引**。
  - **MiniMax-M1**([15]):工业级混合注意力的端到端推理体系。

**对本研究者的启示**:机会 1 必须把 taxonomy 实证化(类比 RaaS+LazyEviction 的现象发现路径);机会 5 必须把"自我纠正崩溃"现象量化(类比 R-KV 的反直觉精度结果)才能进入 CCF-A 视野。

### 2. 方法的差异化锐度

审稿人会在 30 秒内决定是否继续读。**必须一句话讲清 vs. 相邻 SOTA 的差异**。本子方向的"差异化轴"非常拥挤,新工作必须明确占住下列**至少一条**:

- **轴 A:稀疏化对象的转移** —— 从 token/page 转移到 head(DuoAttention [17]、HeadKV [25]、RLKV [8])、layer(DELTA [7])、channel(SparK [41])、graph propagation(GraphKV [42])。**RLKV 的锐度公式**:"识别推理关键头 ≠ 检索头,并用 RL 端到端优化预算分配"。
- **轴 B:稀疏判据的语义化替换** —— 从 attention score 转移到 key 相似度(R-KV [3])、long-term contribution(ForesightKV [35])、global attention(G-KV [37])、未来注意力预测(LookaheadKV [38]、Expected Attention [26]、AttentionPredictor [22])、token 创建时内在重要性(Cache What Lasts [43])。
- **轴 C:稀疏机制的非平凡组合** —— sparse + speculative(SpecAttn [11]、CaliDrop [39])、sparse + TTS(DMS [24])、sparse + offloading(NOSA [29])、SnapKV+top-K 两阶段(RocketKV [23])。组合必须是"非平凡"——CaliDrop 的"淘汰后不删而 offload + query 校准"是范式新,SnapKV+top-K 仅做两阶段则不够。
- **轴 D:训练范式的转换** —— training-free(R-KV、G-KV、Quest 后续)→ 轻量自蒸馏(SeerAttention-R [1],0.4B token)→ 端到端原生稀疏训练(NSA、DSA)→ RL aligned(RLKV、ForesightKV)。每跨一档算力门槛升一档,但锐度也升一档。
- **轴 E:误差/界的理论刻画** —— SSA([32]) 的稀疏注意力*误差线性上界*与*特征空间对齐*。本轴几乎空白,凡是能讲出"我有 bound"的工作锐度都极高。

**反例(锐度不足注定被拒)**:"我们在 Quest 的 page 选择上加了一个温度参数,在 LongBench 上涨了 0.3 分"——所有 5 条轴都没占住。

**对本研究者的启示**:机会 1 占轴 E + 部分轴 B;机会 4 占轴 B + 轴 C(在线阶段切换稀疏算子);机会 5 实际占的是**第六条隐藏轴 —— 评估维度的拓展**,这条轴对 AAAI 友好但对 NeurIPS/ICML 锐度需要更强方法贡献支撑。

### 3. 实验体量(硬门槛)

本子方向 CCF-A 论文的实际 **minimum bar**,从本调研论文统计:

| 项目              | 硬门槛                                                                         | 优秀 bar                                      | 本调研标杆                                                            |
| --------------- | --------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| **基准任务**        | AIME24 / 25 + GSM8K + MATH + LongBench 至少 3 个                               | + LiveCodeBench + GPQA + AIME I/II 完整       | R-KV、SeerAttention-R、G-KV、ForesightKV、Crystal-KV                 |
| **模型规模**        | 至少 2 个推理模型(典型 R1-Distill-Qwen-7B/14B + QwQ-32B)                             | 加上 DeepSeek-R1-Distill-Llama-70B、Qwen3-Next | SeerAttention-R、NSA                                              |
| **baseline 数量** | ≥ 6 个,**必含 Quest + DuoAttention + SnapKV + H2O + StreamingLLM** + 1 个推理专用基线 | ≥ 10 个,补 R-KV / Ada-KV / HeadKV / RaaS      | R-KV、ForesightKV、Crystal-KV                                      |
| **稀疏率扫描**       | 至少 3 档(如 25% / 50% / 75% 或 1K/2K/4K budget)                                 | 含 90% 极端稀疏 + 16%/8% 异常低预算                   | SeerAttention-R 的 4K budget、R-KV 的 16% 反超实验                      |
| **效率实证**        | **必须有 wall-clock latency + memory 实测,不能只 FLOPs**                            | 含 H100 / A100 双卡型对照,含端到端 throughput         | SeerAttention-R(9× over FA-3 H100)、NSA(11.6× at 64k)、NOSA(5.04×) |
| **生成长度**        | 至少 16K 输出长度评测                                                               | 32K-64K 长 CoT 实测                            | RaaS、Lil(对 long decode 信息损失专门分析)                                 |

**硬门槛(任一缺失大概率拒)**:
- 没有 Quest 作 baseline —— 立刻显得不了解 decoding 稀疏领域。
- 只有 FLOPs / 没有 wall-clock —— SeerAttention-R / NSA 已经把"实测加速"做成标配,再退回 FLOPs 等于自爆。
- 只测 short benchmark 不测 long CoT —— 等同于宣布方法可能在长生成下崩溃。
- **必须有 Lil 风险评估**:即至少报告生成长度是否膨胀,否则审稿人会质疑端到端代价。

**对本研究者的启示**:机会 5 的"安全边界"诊断本质是评测协议贡献,但仍需配 ≥ 3 个推理 benchmark × ≥ 4 个 sparse baseline 才能站住,不可裸跑分析。

### 4. 消融与可解释性(标杆模板)

本子方向消融做得最规范的模板论文:

- **逐模块消融模板 → SeerAttention-R**([1]):分别消融 query pooling 去除、GQA 共享、TileLang 内核三件套,每件量化其贡献。
- **超参敏感性曲线模板 → Quest**([16])与 **Ada-KV**([18]):Top-K / page size / budget 三个维度完整扫描曲线,而非只报一个点。
- **可视化模板 → RaaS**([2])与 **LazyEviction**([5]):attention map 时间轴可视化 + 现象统计直方图,把"milestone"与"recurrence"画给审稿人看。
- **可解释性模板 → DuoAttention**([17])与 **Retrieval Head**([21]):头功能图谱可视化 + 跨模型一致性验证,直接奠定"头分类"研究范式。
- **诊断/因果实验模板 → Thought Anchors**([30]):*black-box 重采样 + white-box 注意力聚合 + 因果 ablation* 三件套,凡涉及"识别关键 token / 关键 step"的工作必须借鉴。

**反例**:很多 arxiv 论文只放一个总表,没有"换 X 为 Y 涨了多少"的对位消融。本子方向 CCF-A 论文几乎没有放过这一关的。

**对本研究者的启示**:机会 1 的 taxonomy 必须按 RaaS+DuoAttention 的双重模板做(现象可视化 + 跨模型一致性);机会 5 必须按 Thought Anchors 的因果 ablation 做"淘汰这批 KV 后自我纠正成功率怎么变"。

### 5. 写作和故事线

- **一句话 contribution** —— 必须能在 abstract 第 2 句说清。本调研典范:
  - SeerAttention-R:"首个 long-reasoning decoding 自蒸馏稀疏 + 9× over FA-3"
  - R-KV:"reflection 冗余的语义去重,16% KV 反超 100% 精度"
  - NSA:"compression + selection + sliding 三类稀疏端到端可训练,11.6× decoding 加速"
  - 三句话**都同时包含{现象/动机 + 方法骨架 + 关键数字}**,缺一项即落档次。
- **Related Work 必须显式 contrast**,不能堆砌。本子方向有 **4–5 篇必须逐篇 contrast 的相邻论文**(不 contrast 等于自杀):
  1. **SeerAttention-R**([1]) —— decoding 稀疏的事实锚点,所有训练相关工作都要对比
  2. **Quest**([16]) —— decoding KV 选择的事实基线,所有 KV 选择工作都要对比
  3. **R-KV**([3]) —— 推理特化压缩范式,所有推理任务 KV 工作都要对比
  4. **DuoAttention**([17]) / **RLKV**([8]) —— 头功能图谱,所有头粒度工作都要对比
  5. **RaaS**([2]) / **LazyEviction**([5]) / **ThinKV**([4]) —— CoT 模式刻画三件套,所有模式分析工作都要对比
- **Limitation Section 加分** —— Lil([9]) 实际上把"稀疏负效应"写成了一篇独立论文,如果新工作在 Limitation 主动报告"在 X 设置下我们方法退化为 baseline"反而显成熟,**ICLR 审稿人尤其偏爱**。
- **故事线两种模板**(从本调研提炼):
  - **现象驱动型**(RaaS、LazyEviction、R-KV):现象 → 量化指标 → 算法 → 加速。适合机会 1、机会 2。
  - **范式冲突型**(NSA、DMS、Hold Onto That Thought):指出主流方法的隐含假设错 → 重新定义评估 → 新方法。适合机会 5。

### 6. 会议偏好差异

| 会议                                  | 偏好                                                               | 命中率(本子方向)          | 本子方向案例                                                                       |
| ----------------------------------- | ---------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------- |
| **NeurIPS**                         | 完整方法 + 充分实验 + 一个 surprising finding;允许偏经验,但要 SOTA                | 中高(本子方向最大输出口)      | R-KV、Ada-KV、Twilight、AttentionPredictor、DMS、Hold Onto That Thought(workshop) |
| **ICLR**                            | 现象发现 + 表征 / 理论 / 长度外推 / 学习算法新范式;偏好"机制理解"                         | 高(对机会 1、2、5 最契合)   | DuoAttention、HeadKV、Retrieval Head、ASEntmax、Thought Anchors(在审)              |
| **ICML**                            | 算法新颖性 + 较强理论分析;偏向有形式化的 sparse / pruning 工作                       | 中(本子方向偏经验,需补理论)    | Quest、RocketKV                                                               |
| **ACL**                             | 推理任务 + 长生成应用 + 真实 NLP benchmark;**接受 NSA 这种带工业体量训练的 Best Paper** | 中(算法+NLP 评测结合时高)   | NSA(Best Paper)                                                              |
| **EMNLP**                           | NLP 评测 + plug-and-play 方法;比 ACL 友好但影响力略低                         | 中                  | GraphKVni                                                                    |
| **AAAI**                            | scope 收敛 + 严谨可复现 + 不追极致新颖;**对评测协议/诊断类贡献友好**                      | **中高(对机会 5 强烈契合)** | (本调研中较少专属案例,但是机会 5 的最优出口)                                                    |
| **IJCAI**                           | 与 AAAI 类似但偏 reasoning / symbolic                                 | 中                  | —                                                                            |
| **KDD / SIGIR / WWW / ICDE / CVPR** | **与本子方向几乎不相关**,除非把方法落到 RAG / 搜索 / 推荐 / 视觉推理具体场景                  | 低                  | —                                                                            |

**关键提示**:
- **NeurIPS**:本子方向 2025 已经吸收了 R-KV / Ada-KV / Twilight / AttentionPredictor / DMS / Gated Attention(Best Paper) 等,**容量饱和**,2026 投 NeurIPS 必须有"比 R-KV 更狠的反直觉结果"或"比 NSA 更扎实的训练体量"。
- **ICLR**:对"机制理解类"工作(ASEntmax 的长度外推、Thought Anchors 的因果 ablation)接受度高,适合机会 1 / 机会 2 押注。
- **ACL Best Paper 给了 NSA** 说明 ACL 已经接受"工程体量碾压"的工作进入,但**普通研究者难以复刻**这条路。
- **AAAI 对诊断类工作的友好度被低估了**,机会 5 的"自我纠正安全边界"恰好对位。

### 7. 一句话命中公式

**本子方向 CCF-A 命中公式**:

> **(新现象 OR 范式冲突) × (一句话锐度) × (≥3 benchmark + ≥6 baseline + wall-clock latency) × (逐模块消融 + 现象可视化) × (4-5 篇相邻工作逐篇 contrast)**

五项缺一档,中稿概率立刻减半。

**对照本调研 5 个机会的命中预期**:

| 机会 | 新现象/范式冲突 | 锐度 | 实验体量门槛 | 消融/可视化 | Contrast 难度 | 综合最可能命中 |
|------|----------------|------|-------------|------------|--------------|----------------|
| **1 统一 taxonomy + 理论 mask** | ★★★★★(完整 taxonomy 空白) | ★★★★(占轴 E + B) | 高(需多模型多 benchmark) | ★★★★★(可视化是强项) | 难(几乎所有模式刻画论文都要 contrast) | **ICLR 2027 / NeurIPS 2026** |
| **2 Reasoning sink** | ★★★★(若现象存在则极强) | ★★★(占轴 A,与 Gated Attention 对话) | 中(可分析为主) | ★★★★(sink 可视化经典) | 中 | **ICLR / ACL** |
| **3 RL-aligned 稀疏 mask** | ★★(没有新现象) | ★★★★(占轴 D) | 高(RL 训练 + 完整对照) | ★★★(RL 消融要求高) | 难(与 RLKV / SeerAttention-R 都要画清) | **ICLR / NeurIPS** |
| **4 阶段感知动态稀疏** | ★★★(阶段 = 已知概念,新在"在线切换") | ★★★(占轴 B + C) | 中高 | ★★★★ | 难(必须与 ThinKV 划清) | **ICLR / AAAI** |
| **5 安全边界(自我纠正)** | ★★★★(评估维度新现象) | ★★★(占第六条隐藏轴) | 中(分析主导) | ★★★★(诊断协议+因果 ablation) | 中(主对照 R-KV / ThinKV / Hold Onto That Thought) | **AAAI(首选)/ ICLR** |

**最优配比建议**:
- 若追求**最高影响力**:押机会 1 → ICLR/NeurIPS,但必须接受 50 A800·天 + 6 个月时间窗口压力。
- 若追求**最高命中率**:押机会 5 → AAAI,<20 A800·天,scope 收敛,即便主结论偏负面也可成文。
- 若资源中等且能承受 RL 训练:机会 3 + 机会 4 组合,把 RL-aligned 与阶段感知做成一篇,投 ICLR / NeurIPS。
