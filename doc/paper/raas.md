
# RaaS 精读笔记

**论文**：RaaS: Reasoning-Aware Attention Sparsity for Efficient LLM Reasoning
**作者**：Junhao Hu 等（Peking University / Nanjing University / Huawei Cloud）
**出处**：arXiv 2502.11147v2（2025年5月）
**代码**：https://github.com/DerekHJH/raas

---

## 1. 研究背景与动机

**问题转移：从 long-prefill 到 long-decode。** 推理模型（o1/o3、DeepSeek-R1）在 decode 阶段生成数万 token 的 long CoT，作者用实测说明：在固定 32k token 工作负载下，decode 时间随 decode token 数增加而**远快于** prefill 时间增长，推理任务中 decode 阶段占据约 **99% 的 JCT（Job Completion Time）**（Figure 1c）。但学界对 long-decode 的优化远不如 long-prefill 充分。

**"不可能三角"（impossible trinity）。** 现有稀疏算法在 accuracy / time / memory 三者上无法同时取优：
- **H2O**：理论 O(L) 时间内存，但无法用高效注意力 kernel、缺乏 page 级 KV 管理，实际不可用且精度低（基于累积注意力分数淘汰，会过久保留过时的 milestone token）。
- **StreamingLLM/Sink**：O(L) 时间内存，但静态只保留 initial + recent，激进策略在推理任务上精度极低。
- **Quest**：精度高、O(L) 时间，但保守地缓存**全部** token 的 KV → O(N) 内存。

**核心 insight。** 作者通过手工检视 Qwen2.5-Math-7B 在 100 条 MATH500 上、28 层 ×28 头的注意力图，发现 decode 阶段出现一种新模式：
- **milestone token（里程碑 token）**：类比数学证明中的引理（lemma），先获得高注意力、被使用后注意力骤降且**永不回升**，表现为注意力图上逐渐变暗的"亮列"（占比 24.2% 的注意力图）。
- **phoenix token（凤凰 token）**：被冷落足够久（可能已被淘汰）后**重新获得**重要性的 token，主要出现在 prefill 部分（如 user query，模型回头看题目），占比约 1.5%。
- 超过 70% 的注意力图呈 StreamingLLM 模式（74.3%）。

---

## 2. 方法详解

**核心思路**：识别 milestone token 并用 LRU 思想保留其 KV，直到它们彻底不再被需要；同时**完整保留所有 prefill token 的 KV** 以避免 phoenix token 丢失关键信息（query 回看）。

**关键设计：**
1. **基于时间戳的淘汰**：每个 decode step 中，注意力分数高于 median 的 token 视为"used"，赋予最新时间戳。milestone token 持续被刷新时间戳直到永久无关；cache 满时淘汰**时间戳最旧**的 token。
2. **保留全部 prefill token**：prefill 通常较短，且 phoenix token 几乎都出现在 prefill 内，故全保留以防关键 query 信息丢失。RaaS 仅作用于 decode token。
3. **比例参数 r = 0.5**：每个 decode step 给注意力分数 top-r（r=0.5 即 median 之上）的 token 刷新时间戳。r 太大→区分度不足；r 太小→丢失 milestone。消融显示 r=0.5 最佳。
4. **Page-Based RaaS**（实际采用版）：token 级管理碎片化严重且与 FlashAttention 不兼容。改为 page_size=16（同 vLLM），每个 page 选代表性 K，用新 decode token 的 Q 与之计算单个代表分数来更新 page 时间戳（代表选择策略沿用 Quest，保证公平）。

**与已有方法的本质区别**：H2O 用累积注意力分数（会被早期高分长期"占位"），RaaS 用最近时间戳（LRU 思想），更契合 milestone "用完即弃"的生命周期；同时通过保留 prefill 解决 phoenix 问题，从而把内存压到真正的 **O(L)**（Quest 是 O(N)）。

---

## 3. 实验设计评估

- **模型（4 个）**：Marco-o1、Qwen2.5-Math-7B-Instruct、Mistral-Math-7B、DeepScaleR-1.5B-Preview。
- **数据集（3 个）**：GSM8K、MATH500、AIME（各取前 200 题）。
- **基线**：Dense、H2O、StreamingLLM(Sink)、Quest。
- **指标**：JCT（延迟）、Accuracy。环境：单张 A100-80GB，vLLM。

**主要结果：**
- 精度（Figure 5）：H2O、Sink 在固定 cache budget 下精度差；Quest 与 RaaS 精度最好且相当。cache budget 1024 时基本可匹配 Dense。
- 延迟/内存（Figure 6）：Dense JCT 呈 O(N²) 增长；Quest 和 RaaS 呈线性（O(NL)）；内存上 Dense/Quest 线性增长（O(N)），**RaaS 在超过 cache budget 后趋于平台（O(L)）**——这是 RaaS 相对 Quest 的核心优势。
- 微基准：丢弃 milestone token（H2O-128、Sink-128）会**增加 decode 长度**——模型一开始推理正确，丢失 milestone 后 lose track，反复 re-reasoning 直至卡死（Figure 7，与 Lil/Hold-Onto-That-Thought 现象呼应）。

**【⚠️ 存疑】** 当 cache budget 很小时，RaaS 因全保留 prefill 反而把预算几乎全分给 prefill、丢弃几乎所有 decode token，精度低于 Quest。作者自己建议"小预算/long-prefill 场景用 Quest，decode 用 RaaS"——这其实削弱了 RaaS 作为独立方法的普适性。

**【⚠️ 存疑】** milestone/phoenix 的占比统计（24.2% / 1.5% / 74.3%）来自**单一模型、100 条样本、人工检视**，作者在 Limitations 中也承认"缺乏全面的注意力图统计 + 无自动化分析工具"，模式结论的客观性证据偏弱。

---

## 4. 局限性与未来方向

作者明确指出：
1. **缺乏全面的注意力图统计**：人工检视、小规模，未来计划做自动化注意力模式分析工具。
2. **RaaS 适用范围有限**：只适合 "prefill 短、decode 长" 的推理任务；long-prefill 场景失效。
3. **评测覆盖窄**：仅 4 模型 ×3 数据集，全组合评测算力上不可行（200 题 16k token 单卡需一天）。

我认为还未解决的问题：
- median（r=0.5）作为"used"阈值是固定的，不同层/头的注意力集中度差异巨大，固定阈值可能不最优（可对照 Ada-KV 的逐头自适应预算）。
- milestone 与 phoenix 的边界依赖 cache budget 定义（milestone 在小预算下会退化为 phoenix），定义不够内生稳定。
- 未在端到端"是否生成更长"上做系统量化（仅 Figure 7 定性）。

---

## 5. 评价

**贡献为何有效**：RaaS 抓住了推理 decode 的关键结构性事实——milestone token 是"一次性生命周期"（emerge→utilized→fade），因此**时间戳（最近性）比累积分数（H2O）更匹配**淘汰决策；而 phoenix token 集中在 prefill，故"全保留 prefill + 仅对 decode 做时间戳淘汰"这一简单组合就同时拿下了 accuracy 和真正的 O(L) 内存，打破不可能三角。Page-based 实现保证了工程可用性。

**启发/借鉴价值**：
- 是继 SeerAttention-R 之后，**最早系统形式化"推理 decode 注意力模式"**的工作之一，milestone / phoenix 概念为后续 LazyEviction（recurrence）、R-KV（reflection）、ThinKV（thought-type）提供了经验起点。
- 对本研究方向（统一 CoT 注意力 taxonomy）：milestone 可直接映射为 taxonomy 中的一类（"用完即弃"），phoenix 对应 recurrence/answer-anchor 边界——但 RaaS 仅给出现象与启发式 mask，**没有理论误差界**，正是本方向"per-pattern bound"可补的空白。
- 警示：稀疏会导致生成变长，端到端评估必须看 JCT 而非仅注意力 FLOPs。

**关联**：与 [[lazyeviction]]（recurrence 视角互补）、[[r-kv]]（reflection 冗余）、[[quest]]（核心基线，page 代表选择沿用）、[[lil]]（生成变长副作用）密切相关。
