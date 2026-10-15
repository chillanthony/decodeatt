# Gated Attention 精读笔记

**论文**：Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
**作者**：Zihan Qiu, Zekun Wang, Bo Zheng, Junyang Lin 等（Qwen Team, Alibaba Group / Edinburgh / Stanford / MIT / Tsinghua）
**出处**：NeurIPS 2025 Oral（top 1.5%）（arXiv 2505.06708v1，2025年5月）
**代码**：https://github.com/qiuzh20/gated_attention

---

## 1. 研究背景与动机

**问题**：门控机制（gating）从 LSTM、Highway Network 到 SSM、linear attention、SwiGLU 被广泛使用，但在**标准 softmax attention** 中其具体作用、机制与影响**缺乏系统研究**。已有工作（Switch Heads 的 top-K head gating、NSA 的门控）都把门控收益与路由/稀疏架构混在一起，**没有把"门控本身的贡献"从其他架构组件中剥离出来**。

**核心动机**：系统性地、在大规模（3.5T token、30+ 变体）下回答——在标准 attention 中加门控到底有没有用、加在哪里、为什么有用？

**关键预观察**：Switch Heads 即使退化到单 expert（去掉路由）仍有显著增益，说明**门控本身**提供内在价值，独立于路由机制。

---

## 2. 方法详解

**门控形式化（公式 5）**：`Y' = g(Y, X, W_θ, σ) = Y ⊙ σ(X W_θ)`——对要调制的 Y，用另一输入 X 算门控分数 σ(XW_θ) 做动态过滤。默认 head-specific、multiplicative、sigmoid。

**系统性探索五个维度：**
1. **位置（Positions）**：Q/K/V 投影后（G2/G3/G4）、SDPA 输出后（**G1**）、最终 dense 输出后（G5）。
2. **粒度（Granularity）**：headwise（单标量调制整个 head 输出）vs elementwise（逐维向量调制）。
3. **Head-specific vs Head-shared**：每个 head 独立门控 vs 组内共享。
4. **Multiplicative vs Additive**：`Y·σ(Xθ)` vs `Y + σ(Xθ)`。
5. **激活函数**：sigmoid（[0,1]，带稀疏）vs SiLU（无界）。

**核心发现**：**在 SDPA 输出后加 head-specific、elementwise、multiplicative、sigmoid 门控（G1）最有效**（MoE-15B 上最高 −0.2 PPL、+2 MMLU；wall-time 开销 <2%）。

**两个机制解释（§4，论文核心理论）：**
1. **Non-linearity（非线性，§4.1）**：标准 attention 中 `o = (Σ S·X W_V) W_O`，W_V 与 W_O 可合并为一个**低秩线性映射**（d_k < d_model，GQA 下 W_V 组内共享更降秩）。在 G1/G2 处加门控相当于在 W_V 与 W_O 间插入非线性（公式 7/8），提升低秩映射的表达力。这解释了为何 G5（W_O 之后）加门控**无效**——它没有处理 W_V–W_O 之间缺非线性的问题。
2. **Input-dependent Sparsity（输入依赖稀疏，§4.2）**：SDPA 输出门控分数高度集中在 0 附近（Figure 3，最强稀疏），即 query-dependent 地过滤掉与当前 query 无关的上下文信息。**Query-dependency 关键**：G1（依赖当前 query 的 hidden state）优于 G2（依赖 past key/value 的 hidden）；NS-sigmoid（去稀疏版 0.5+0.5·sigmoid）效果变差，证明稀疏本身有价值。

**最重要的副产品——消除 Attention Sink（§4.3）：**
- baseline 模型把平均 **46.7%** 注意力分给首 token（attention sink），Gated Attention 降到 **4.8%**（Figure 2）。
- 机制：input-dependent、head-specific 的 SDPA 输出门控引入稀疏 → 过滤无关上下文 → 减少 massive activation → 消除 sink。
- **反直觉发现**：消除 massive activation 后 sink 仍可被门控独立消除 → **massive activation 不是 attention sink 的必要条件**（修正了 Sun et al. 2024 的观点）。

---

## 3. 实验设计评估

- **模型**：15B MoE（15A2B，2.54B 激活，128 expert top-8，GQA）+ 1.7B dense。3.5T token 高质量数据（多语言/数学/通用），context 4096。
- **评测**：Hellaswag、MMLU、GSM8K、HumanEval、C-eval、CMMLU + 多领域 PPL；长上下文用 RULER。
- **对照**：vanilla baseline + 参数扩展基线（加 KV head / query head / expert，确保参数量可比甚至更多）。

**主要结果：**
- 门控位置（Table 1）：SDPA 输出 G1 elementwise 最优（PPL 5.761 vs baseline 6.026，MMLU 60.82 vs 58.79）；head-specific >> head-shared；multiplicative > additive；sigmoid > SiLU。
- 训练稳定性 + scaling（Table 2、Figure 1 右）：门控大幅减少 loss spike，容忍更大 learning rate 与 batch size（baseline 在高 LR 下发散，门控模型不发散且涨点）；sandwich norm 只能恢复收敛但不涨点。
- 非线性消融（Table 3）：RMSNorm（几乎无参）也降 PPL，验证非线性贡献；G1 加 SiLU（无参，row 6）也小幅降 PPL。
- 稀疏消融（Table 4）：门控分数 mean 普遍 <0.5（稀疏）；NS-sigmoid（去稀疏）变差；input-independent 门控（row 6）仍因非线性涨点，但不如 input-dependent。
- 长上下文外推（Table 5）：32k 内门控略优；YaRN 扩到 64k/128k 时门控模型**显著优于** baseline（RULER 上 64k/128k 大幅领先）——sink-free 使 RoPE 外推更稳健。融入 Qwen3-Next。

**【⚠️ 存疑】** 本文是面向**预训练架构改进**（需从头训练 3.5T token），不是对现成模型的推理时优化——与 decode 稀疏（RaaS/Quest/SeerAttention-R）的场景完全不同。其"稀疏"指门控诱导的**函数级稀疏**（gating score→0），而非 KV 选择/淘汰的结构稀疏，两者不可混为一谈。

**【⚠️ 存疑】** "消除 attention sink" 是在 Gated Attention **自己训练的**模型上观测的——这是"训练出无 sink 的新模型"，而非"解释现有推理模型（R1/QwQ）是否有 sink"。对本研究方向"reasoning sink 是否存在"的问题，Gated Attention 只提供了"sink 是 architectural artifact、可被门控消除"的一面证据，不能直接回答推理诱导的 sink 是否存在。

**【⚠️ 存疑】** 作者在 Limitations 中坦承：**未给出 attention sink 如何影响长度外推的严格理论解释**，机制论证主要靠消融与相关性，因果链不够硬。

---

## 4. 局限性与未来方向

作者明确（Limitations）：
1. **缺乏严格理论**：非线性对 attention 动力学/训练过程的更广影响 under-explored；sink 影响外推能力无严格理论解释。
2. 主要是经验性消融，机制解释（非线性、稀疏、sink）多为相关性论证。

我认为还有的问题：
- 仅在自家架构 + 3.5T 预训练验证，门控收益是否依赖特定数据/MoE 结构未完全隔离。
- 未在推理模型（long CoT）场景验证门控对推理质量/稀疏模式的影响。

---

## 5. 评价

**贡献为何有效**：本文最大价值是**把"门控"从各种混杂架构中干净地剥离出来做大规模对照**，并给出两个机制解释（非线性、输入依赖稀疏），逻辑自洽地解释了"为何 G1 最优、G5 无效、sigmoid 优于 SiLU、head-specific 关键"。最具影响力的发现是 **SDPA 输出门控自然消除 attention sink**，且证明 massive activation 非 sink 必要条件，修正了既有理解。NeurIPS 2025 Oral，3.5T token 工业级验证，可信度高。

**启发/借鉴价值**：
- **对本研究方向（reasoning sink 现象的存在性、机制与利用）是最直接的对话对象**：Gated Attention 消除了 sink，本方向应反向探究——在推理模型（R1/QwQ）及 Gated-Attention 训练的 Qwen3-Next 上，sink 是否仍存在？是否存在推理诱导的"reasoning sink"？两者构成天然对照（他们消 sink，本方向探新 sink）。
- **对统一 CoT 注意力 + mask/门控/entmax 三层组合**：Gated Attention 提供了"门控（gate）"这一机制的最强经验背书与机制解释（非线性 + 输入依赖稀疏），是本方向"mask + gate + entmax"组合中 gate 部分的理论与实证基础。
- "稀疏应是 input-dependent / query-dependent" 的结论，与 [[quest]] 的 query-aware criticality、本方向的"模式特化算子"一致——稀疏不应静态。
- sink 是 architectural artifact 且可消除的证据，提示本方向若发现 reasoning sink，需辨析它是架构残留还是推理诱导（与传统 BOS sink 的位置/动力学差异）。
- 长上下文外推受益于 sink-free，提示 sink 与长度泛化的关联值得在推理 long-decode 场景再检验。

**关联**：[[quest]]（input/query-dependent 稀疏一致）、[[seerattention-r]]（门控诱导稀疏 vs 自蒸馏门控选择 KV，两种"gate"含义不同）、[[nsa]]（NSA 也用 sigmoid 门控融合分支，本文剥离了门控独立贡献）；与本方向 reasoning sink、entmax 函数稀疏（ASEntmax）、mask+gate+entmax 组合直接相关。

---

## 6. 门控的依据与数学过程

### 6.1 详细版

**门控的"依据"是什么**

依据 = **当前 token 自己的 hidden state** $X$（即 query 侧输入）。门控分数不是凭空给的，而是用当前位置的输入向量 $X$ 过一个线性层再 sigmoid 算出来——所以它是 **input-dependent（输入依赖）+ query-dependent（依赖当前 query）**。论文证明依赖"当前 query 的 hidden"（G1）优于依赖"过去 key/value 的 hidden"（G2），因为门控语义是"**就当前这个 query 而言，哪些上下文该留、哪些该滤掉**"。代码中门控参数 $W_\theta$ 被合并进 `q_proj`，输入正是 `hidden_states`。

**数学过程**

第 1 步——通用门控形式（公式 5）：
$$
Y' = g(Y, X) = Y \odot \sigma(X W_\theta)
$$
- $Y$：被调制对象（这里是 SDPA 输出）；$X$：算门控依据的输入（当前 token hidden state）
- $W_\theta$：门控线性投影（代码并进 q_proj）；$\sigma$：sigmoid，压到 $[0,1]$；$\odot$：逐元素乘

第 2 步——标准 attention 输出（被门控的 $Y$）：
$$
Y = \text{SDPA} = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)\,V
$$

第 3 步——把门控插进 G1 位置：
$$
o = \Big[\;\underbrace{\text{SDPA}(X)}_{Y}\;\odot\;\underbrace{\sigma(X W_\theta)}_{\text{门控分数}}\;\Big] W_O
$$
对应代码：
```python
attn_output = attn_output * torch.sigmoid(gate_score)  # Y ⊙ σ(XWθ)
attn_output = self.o_proj(attn_output)                 # 再乘 W_O
```

**这个数学过程做了两件事**

① **注入非线性**：标准 attention 中 $o=(\sum S\cdot XW_V)W_O$，$W_V$ 与 $W_O$ 是两个线性矩阵、中间无非线性，可合并成低秩映射（GQA 下更低秩）。门控的 $\sigma(\cdot)$ 恰好插在 $W_V$ 与 $W_O$ 之间，打破"两线性层直连"，给低秩通路加非线性。→ 解释了 **G5（$W_O$ 之后）无效**：那里已无"缺非线性"问题。

② **输入依赖的稀疏**：$\sigma(XW_\theta)$ 值高度集中在 0 附近（Figure 3），按当前 query 把无关上下文乘成近 0，做动态过滤。→ 副产品消除 attention sink；消融中换成去稀疏版 $0.5+0.5\,\sigma(\cdot)$ 效果变差，证明稀疏本身有价值。

### 6.2 简版

**依据**：当前 token 自己的 hidden state $X$（query 侧输入）。

**数学**：
$$
o = \big[\;\underbrace{\text{softmax}(QK^\top/\sqrt d)V}_{\text{SDPA 输出 }Y}\;\odot\;\sigma(XW_\theta)\;\big]\,W_O
$$
即给 SDPA 输出逐元素乘一个 $[0,1]$ 的 sigmoid 门控分数，再过 $W_O$。

**两个作用**：① 在 $W_V$–$W_O$ 之间插入**非线性**；② 做 query-dependent 的**稀疏过滤**（顺带消除 attention sink）。
