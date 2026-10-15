# RLKV 精读笔记

**论文**：Which Heads Matter for Reasoning? RL-Guided KV Cache Compression
**作者**：Wenjie Du, Li Jiang, Keda Tao, Xue Liu, Huan Wang（Westlake / McGill / Mila / ZJU / MBZUAI）
**出处**：arXiv 2510.08525v2（2026年1月）
**主页**：https://kurt232.github.io/RLKV

---

## 1. 研究背景与动机

**问题**：推理 LLM 的 long CoT 对信息损失极脆弱，KV 压缩在推理模型上容易崩。现有两类方法都有结构性缺陷（Figure 1）：
- **Token-dropping（H2O、R-KV）**：对每个 head 统一逐 token 淘汰，不可避免破坏 CoT 一致性 → 退化为重复循环（repetitive loop）无法收敛。R-KV 虽为推理设计，仍逃不出此限。
- **Head-reallocation（DuoAttention）**：给检索头 full KV、其余压缩。但其检索头是为**检索任务**（passkey recall）识别的，**不是为生成式推理**——在推理模型上常产生过长 CoT、耗尽长度预算仍答不出（overlength error）。

**关键 case study（Figure 1b）**：MBPP 代码任务上，reasoning 模型与 instruct 模型用**相同**未压缩性能，但压缩后 reasoning 模型显著退化——说明退化主因是**extended CoT generation**（推理 KV 承载推理行为/状态），而非模型能力差异。

**核心问题与 insight**：到底**哪些注意力头对推理关键**？现有方法答不上来，因为它们用的是注意力分数或检索代理。RLKV 主张：**识别推理关键头必须直接观察"压缩该头如何影响实际生成的推理结果"**。

---

## 2. 方法详解

**操作性定义**：reasoning-critical head = "在 local KV 访问下会显著降低推理性能的 KV head"。

**核心思路（Figure 2）**：用 RL 作为 probe，直接以**推理结果质量（可验证 reward）**为信号优化每个 head 的 KV 使用门控，找出哪些 head 必须 full KV。

**关键设计：**
1. **Mixed Attention with Gating Adapters（§3.1）**：给每层每个 KV head 一个门控 `α_{l,h} ∈ [0,1]`：`out_mix = α·full_attn + (1−α)·local_attn`（local = sink + recent，用 StreamingLLM mask 保数值稳定）。**仅 L×H 个门控参数可学，冻结全部模型权重**——把 RL 优化空间压到极小使其可行。
2. **RL for Head Identification（§3.2）**：用 **GRPO**（GroupRelativePolicyOptimization）在数学推理题上优化门控。两处关键改动：
   - **去掉 KL penalty**：传统 RLVR 用 KL 防过优化，这里去掉以**最大化 reward 信号的判别力**（让必要的 head 保持高 α）。
   - **加 scaled L1 正则**（β=‖α‖₁/(L×H)）：推动可压缩 head 的 α→0。
   - 目标（公式 2）：reward signal（GRPO advantage）− L1 penalty。advantage 用组内 reward 标准化（公式 3），正确推理为正、错误为负 → reasoning-critical head 保持高 α。
3. **RL 训练稳定化（§3.3，关键）**：稀疏 reward vs 稠密 penalty 的冲突会致 **training collapse**（Figure 4）——adapter 变稀疏→性能降→reward 更稀疏不稳→dense L1 相对更强→进一步驱 α 归零的恶性循环。两个对策：
   - **Self-distillation Sampling**：先筛模型能解对的题、curriculum 到 3k 题，保证稳定 reward 信号（不同于典型 RLVR 用难题提能力，这里要保能力）。
   - **Adaptive Penalty Weighting**（公式 4）：`β'(r̄,τ) = 𝟙(r̄>τ)·β·(exp(r̄)−1)`，以目标 reward r̄≈0.7 为中心调节 penalty；性能降时减 penalty，且 reward 严重退化时 hard cutoff（τ）完全关掉正则。
4. **推理部署**：用学到的门控分数排序，选 top-k 高 α head 给 full KV，其余给压缩 KV（16 sink + 64 recent）。

---

## 3. 实验设计评估

- **模型（均 GQA）**：Llama-3.1-8B-R1、Qwen-2.5-7B-R1、Qwen-3-4B-Thinking。
- **数据集**：GSM8K、Math500、AIME24（数学）、MBPP（代码）、MMLU-Pro 四子集（Chem/CS/Law/Physics，测跨域泛化）。
- **基线**：Full、H2O、R-KV（token-dropping）、DuoAttention（head-reallocation）。为公平，H2O/R-KV 用 dynamic budget。
- **实现**：AReaL（RL 训练）+ SGLang（rollout）。GRPO 4 samples/query，3000 题，185 步，2×A100。

**主要结果：**
- 主结果（Figure 5、Table 1）：RLKV 在四推理 + 四知识基准上全面最优，高稀疏（0.4/0.6）优势尤大。**20–50% KV 削减近无损**，部分任务反超 Full（如 Llama AIME24 +3.3、Qwen MBPP +1.2）。基线在高稀疏显著掉点。
- 头敏感性（Figure 7）：按门控分数从高到低压缩头——压 reasoning-critical head（RLKV 识别）比压 retrieval head（DuoAttention 识别）或随机头**精度跌得快得多**，证明 RLKV 找的头确实是推理主导驱动。
- 错误模式（Figure 9）：压 retrieval head → 主要 overlength error（语言流畅但推不出结论，验证 DuoAttention 在推理上的缺陷）；压 reasoning-critical head → repetitive + incorrect error（CoT 一致性破坏）。
- 响应长度（Figure 10、Table 2）：token-dropping 在激进压缩下"看似"输出短（因只解出简单题）；R-KV 常致更长；DuoAttention 需更长 CoT。RLKV 保持竞争性长度。端到端加速 **1.09–1.21×**。
- 消融（Figure 8）：去自蒸馏采样 / 去自适应 penalty 都显著掉点；β=1e-3 最优。

**【⚠️ 存疑】** 用 **RL 训练**来识别头，成本远高于 DuoAttention 的优化式识别（虽只训 L×H 门控，但 GRPO rollout 在数学题上跑 185 步 + 需 SGLang rollout 引擎）。"训练稳定化"需两个额外技巧（self-distillation + adaptive penalty），说明 RL probe 本身脆弱、调参敏感，工程门槛高。

**【⚠️ 存疑】** 门控是**静态的**（训练后固定 top-k head）——但推理过程中头的重要性可能随 CoT 阶段动态变化（与 Quest 揭示的 query-dependent criticality 矛盾）。作者在 Future Work 也承认应转向 query-adaptive dynamic guidance。

**【⚠️ 存疑】** 端到端加速仅 1.09–1.21×，且 head-reallocation 引入 Q/K/V 重组开销；当前用 vanilla PyTorch 实现，需自定义 kernel 才能兑现理论收益。实用性目前有限。

---

## 4. 局限性与未来方向

作者明确（Future Work）：
1. **头可进一步细分**：reasoning-critical 内部还可分 retrieval/induction head 等更细类别。
2. **静态 → query-adaptive 动态门控**：当前静态分数，应做动态引导（轻量训练而非从头训）。
3. **需协同设计专用 kernel**：弥合理论 KV 削减与实际端到端加速的差距。

我认为还有的问题：
- 仅在 7B/8B/4B 小模型、数学为主验证，大模型/多领域泛化未充分检验。
- RL probe 的稳定化技巧使方法复杂，可复现性存疑。

---

## 5. 评价

**贡献为何有效**：RLKV 最核心的洞见是——**"重要的头"必须用"任务结果"定义，而非注意力分数或检索代理**。它把这一原则推到极致：直接用可验证 RL reward 优化每个 head 的 KV 使用，让"压缩后推理是否还对"成为唯一裁判。冻结主干、只训 L×H 门控使 RL 可行；自蒸馏采样 + 自适应 penalty 解决了稀疏 reward vs 稠密正则的崩溃问题。由此首次定量回答"哪些头对推理关键"，并实证 reasoning head ≠ retrieval head。

**启发/借鉴价值**：
- 对本研究方向（头功能图谱 × KV 差异化）：RLKV 揭示了 **reasoning-critical head** 这一新头类别，是对 DuoAttention 检索/流式二分的关键细化与批判。本方向的头分类 taxonomy 应纳入"推理头"，并可借鉴其"用结果定义头重要性"的方法论。
- "RL/结果驱动" vs DuoAttention "输出偏差驱动" vs 注意力分数驱动，构成头识别的三层信号强度谱。
- 错误模式分析（overlength vs repetitive vs incorrect）是诊断压缩副作用的好框架，与 [[lil]]/[[raas]] 的"生成变长/重复"现象呼应——压不同头触发不同 failure。
- 静态门控的局限提示：本方向若做头级差异化，应考虑 query-adaptive 动态性（[[quest]] 的核心论点）。

**关联**：[[duoattention]]（被批判的检索头路线，retrieval≠reasoning）、[[r-kv]]（token-dropping 基线，被指破坏 CoT 一致性）、[[quest]]（query-dependent criticality vs RLKV 静态门控）、[[lil]]/[[raas]]（生成变长/重复的错误模式呼应）、Thought Anchors（receiver head 是另一种推理相关头类别）。
