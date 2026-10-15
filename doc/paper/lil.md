# Lil 精读笔记

**论文**：Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage
**作者**：Junhao Hu, Fangze Li, Tao Xie 等（Peking University / Nanjing University / Tencent，与 RaaS 同一核心团队）
**出处**：arXiv 2601.03043v3（2026年4月）
**简称含义**：Lil = "Less is Less"（去掉 "tt" 和 "e"，暗示稀疏注意力收益"少"）

---

## 1. 研究背景与动机

**问题**：Post-Training Sparse-attention in the Decode stage（**PTSD**，免训练、即插即用的 decode 阶段稀疏，如 H2O、StreamingLLM、Quest、InfLLM、RaaS）被认为能降低 decode 时间/内存。但作者发现一个被忽视的"房间里的大象"：

**核心 insight（Lil 问题）**：稀疏注意力**反直觉地增加端到端时间与内存复杂度**：
- 时间：`JCT = TTFT + decode_length↑ × TBT↓`——虽然每步 TBT 下降，但**信息损失迫使模型生成更长序列**来补偿，decode_length 上升，整体 JCT 反而上升。
- 内存：虽然每步存的 KV 少，但生成过程拉长 → KV 在显存中的驻留时间变长，抵消节省。

实证（Figure 1）：广泛使用的稀疏算法在推理任务上输出长度增加最高 **90%**。输出呈"信息损失 → 尝试重建"模式：模型其实早已得到正确答案，但因丢失上下文而继续验证、忘了自己已经答对、反复 re-checking（与 [[raas]] Figure 7、Hold-Onto-That-Thought 现象一致）。

---

## 2. 方法详解

**第一部分：实证刻画 Lil（§3）。** 跨 5 算法 ×3 数据集 ×3 模型系统实验，两点发现：
1. 精度随 cache budget 增加而升（H2O/Sink 因丢 KV 不可恢复，精度低；Quest/InfLLM 保留全 KV 故精度高但内存不省）。
2. 输出长度随 cache budget 增加而降，但仍比 full attention 长（最高 90%）。小预算下信息损失 > 信息增益，模型重复内容、可能无限生成解不出题。

**第二部分：信息论分析（§4）。** 用 **LZ77** 压缩算法量化"句子的信息量"：
- LZ77 把重复子串替换为对早期出现的引用（offset-length），压缩比 ρ（压缩长度/原长）越小 → 新信息越少、冗余越多。
- 理论保证：`ρ − ε(L_s) ≤ h(L_s−1) ≤ ρ`，其中 h 是 per-symbol entropy，ε=O(log L_s / L_s)。即 ρ 可估计信息熵。
- 发现：稀疏注意力生成的序列虽长但 LZ77 信息量远低于 full attention（大量重复引用早期内容来重建丢失信息）。压缩比随生成呈"先快速增长后 plateau"（Figure 3）——plateau 即不再产生新信息。

**第三部分：Guardian 早停算法（§5）。** 基于"信息增益停滞即应停"：
- 每 f 个 decode step 用 LZ77 压缩当前序列得 curCompress，与 lastCompress 比较；增量 < 阈值 t 时判定"新 token 几乎全是冗余"，提前终止生成。
- 超参：f=250（频率），t=20（使 t/f≈0.08，落在 Figure 3 增长段与 plateau 段斜率之间）。LZ77 压缩 128k token 仅约 34ms，开销可忽略。

---

## 3. 实验设计评估

- **模型**：DeepScaleR-1.5B-Preview（DSR）、R1-Distill-Llama-8B（DSL）、Qwen1.5-MoE-A2.7B-Chat（Qwe，dense vs MoE 跨架构）。
- **数据集**：GSM8K、MATH-500、AIME（各前 200 题）。
- **基线稀疏算法**：H2O、Sink(StreamingLLM)、Quest、InfLLM（+ Full）。
- **指标**：Token savings（用 Guardian 前后 token 数比）、Accuracy。环境：单 A100-80GB。

**主要结果：**
- Guardian 减少 token 用量最高 **90%**，精度下降 **< 2%**（Table 1）。
- 有时反而**提升精度**：模型早已答对仍继续验证导致丢失答案，早停防止这种退化。
- 越保守的稀疏配置（Sink→Quest）冗余越少、可省 token 越少，但即便最保守配置仍有可省冗余。
- Token savings 主要来自终止那些会无限生成的**错误案例**（Table 2，correct/incorrect 分别统计）。
- **Guardian 对 full attention 也有效**（Table 3）：说明冗余 CoT 不仅来自稀疏，也来自 ill-pattern（数据/reward hacking）；但作者强调这两类长 CoT 根因正交。
- Qwen MoE（非数学专精、生成短）上 Guardian 收益甚微——能力弱、CoT 短的模型早停空间小。

**【⚠️ 存疑】** 用 LZ77 压缩比作"信息量"代理是巧妙但粗糙的——LZ77 只捕捉字面重复，无法识别"语义重复但措辞不同"的冗余（这类正是推理模型常见的）。因此 Guardian 可能在"换了说法重复"时漏判，低估冗余。

**【⚠️ 存疑】** t/f≈0.08 的阈值是在这些模型/数据上经验调出的；作者称对 t 鲁棒（只要 t/f 在区间内），但跨模型迁移性未充分验证，本质仍是启发式停止准则。

---

## 4. 局限性与未来方向

作者明确指出：
1. **评测覆盖有限**：3 模型 ×3 数据集 ×4 稀疏算法，全组合算力不可行；但论证 Lil 是稀疏注意力的"内生"问题（所有稀疏都假设少数 token 重要、必丢信息），故认为结论普适。
2. **未区分 ill-pattern CoT**：Guardian 对 full-attention 的 ill-pattern 长 CoT 也有效，但与 Lil 根因不同，留作未来。

我认为未解决：
- Guardian 是"事后止损"，不解决稀疏导致信息损失这一根本问题（不能让稀疏既快又不变长）。
- 早停可能误杀真正需要长推理的难题（在 plateau 后才出现关键转折的情况）。

---

## 5. 评价

**贡献为何有效**：第一次系统、定量地揭示了 PTSD 社区的"皇帝新衣"——**只看注意力 FLOPs/单步延迟是误导的，必须看端到端 JCT**。`JCT = TTFT + decode_length × TBT` 这个简单分解一针见血：稀疏降 TBT 但抬 decode_length。用 LZ77 + 信息论给"稀疏导致重复"一个可计算的刻画，再据此设计零成本早停，逻辑闭环完整。

**启发/借鉴价值**：
- **对本研究方向是必读的"警示性"工作**：任何 decode 稀疏方法的实验都必须报告端到端 wall-clock / token 数，而非仅注意力 FLOPs 或单步加速，否则结论可能失真。
- "信息损失 → 生成变长"现象与 [[raas]] 的"丢 milestone 导致 re-reasoning"、[[lazyeviction]] 的"误杀复发 token"、Hold-Onto-That-Thought 互相印证，是同一现象的不同观测面。
- LZ77 压缩比作为"序列信息量"的廉价代理，可用于本方向衡量不同 pattern 稀疏化后的信息损失。
- 提示：本方向若主张"理论指导稀疏 + 误差界"，应把"生成长度膨胀"也纳入误差/代价模型，而非只看单步注意力近似误差。

**关联**：[[raas]]（同团队、互补现象、被列为基线）、[[lazyeviction]]、[[r-kv]]、[[quest]]（基线）、[[nsa]]（training-aware，被明确划在本文 scope 之外）。
