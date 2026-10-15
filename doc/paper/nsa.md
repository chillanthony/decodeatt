# NSA 精读笔记

**论文**：Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
**作者**：Jingyang Yuan, Damai Dai, Wenfeng Liang 等（DeepSeek-AI / PKU / UW）
**出处**：ACL 2025 Best Paper（arXiv 2502.11089v2，2025年2月）

---

## 1. 研究背景与动机

**问题**：长上下文建模（推理、仓库级代码、多轮 agent）中，vanilla attention 的 O(N²) 是瓶颈——64k 解码时注意力占 70–80% 延迟。稀疏注意力是方向，但作者系统批判现有方法两大"幻觉"：

**§2.1 高效推理的幻觉（The Illusion of Efficient Inference）**：
- **Phase-Restricted Sparsity**：H2O 只在 decode 稀疏但 prefill 仍需重计算；MInference 只在 prefill 稀疏。至少一个阶段没加速。
- **与先进架构不兼容**：MQA/GQA 下，KV cache 访问量是组内**所有 query head 选择的并集**。像 Quest 每个 head 独立选 KV，在 GQA 下并集很大，**减了计算却没减内存访问**。

**§2.2 可训练稀疏的迷思（The Myth of Trainable Sparsity）**：
- **Performance Degradation**：post-hoc 稀疏强迫模型偏离预训练轨迹。引 [Chen 2024b]：top 20% 注意力只覆盖 70% 总分数，retrieval head 等结构在推理时被剪易受损。
- **训练效率需求**：长序列训练（长文档预训练 + 长上下文微调 + RL）需要稀疏，但现有方法只针对推理。
- **不可训练组件**：ClusterKV（k-means）、MagicPIG（SimHash）含离散操作，破坏计算图、梯度无法回传。
- **低效反传**：token 粒度选择（HashAttention）导致非连续内存访问，无法用 FlashAttention 的连续块计算。

**结论（§2.3）**：必须把稀疏作为**训练时的原生（native）一等公民**重新设计，同时满足硬件对齐推理加速 + 可训练。

---

## 2. 方法详解

**核心思路（Figure 2）**：把 KV 组织成时间块，对每个 query 走**三条并行注意力分支**，门控加权融合：
`o*_t = Σ_{c∈{cmp,slc,win}} g^c_t · Attn(q_t, K̃^c_t, Ṽ^c_t)`，g 由 MLP+sigmoid 从输入特征算出。

**三个 remapping 策略（§3.3）：**
1. **Token Compression（cmp，粗粒度）**：把连续块（block length l、stride d，通常 d<l 防碎片）的 key/value 用**带块内位置编码的可学 MLP φ** 压成单个压缩 token，捕捉粗粒度全局语义、降算力。
2. **Token Selection（slc，细粒度）**：
   - **Blockwise 选择**：按空间连续块选（GPU 对连续块吞吐高 + 注意力分数有空间连续性）。
   - **重要性分数复用**：直接用压缩分支的中间注意力分数 p^cmp 推导 selection block 重要性 p^slc（公式 9，处理 block scheme 不同的情形），**几乎零额外开销**。
   - **GQA 组内共享**：组内所有 query head 的重要性分数相加（公式 10），确保组内一致块选择 → decode 时最小化 KV 加载（直击 Quest 的 GQA 痛点）。
   - Top-n 块选择（含固定激活 1 个 initial block + 2 个 local block）。
3. **Sliding Window（win，局部）**：单独分支处理最近 w 个 token。**关键设计**：把局部、压缩、选择分到独立分支并给独立 K/V，防止局部模式因学得快而"shortcut"主导、压制其他分支学习（梯度隔离）。

**Kernel Design（§3.4，硬件对齐）**：基于 Triton，针对 GQA/MQA。
- **Group-Centric Data Loading**：每个 query 位置加载整个 GQA 组的所有 head 的 Q + 它们共享的稀疏 KV block 索引到 SRAM。
- **Shared KV Fetching** + **Outer Loop on Grid**（内循环长度≈selected block count n 对不同 query 块基本恒定，放到 Triton grid scheduler）。
- 通过组内共享消除冗余 KV 传输 + 跨 SM 平衡负载，达近最优 arithmetic intensity。

**创新点**：首次实现**端到端可训练 + 硬件对齐**的稀疏注意力；三分支（压缩+选择+滑窗）层次化设计；选择分支复用压缩分支注意力分数；GQA 组共享块选择解决内存访问痛点。

---

## 3. 实验设计评估

- **Backbone**：27B 总参 / 3B 激活的 GQA+MoE（30 层、64 head、group=4、72 routed+2 shared experts、top-6）。
- **预训练**：270B token（8k 长度）+ YaRN 续训/SFT 到 32k。NSA 与 Full Attention 都训到收敛。
- **NSA 超参**：l=32, d=16, l'=64, n=16（含 1 initial + 2 local）, w=512。
- **基线**：Full Attention、H2O、InfLLM、Quest、Exact-Top。

**主要结果：**
- 预训练 loss（Figure 4）：NSA 收敛更平滑、loss 更低。
- 通用基准（Table 1）：9 项中 7 项超 Full Attention，平均 0.456 vs 0.443。推理类增益显著（DROP +0.042、GSM8K +0.034）——稀疏预训练像在过滤无关注意力噪声。
- 长上下文（Figure 5）：64k needle-in-a-haystack **全位置完美检索**。LongBench（Table 2）平均 0.469，超所有基线含 Full Attention（0.437）；多跳 QA、code 上尤强。
- CoT 推理：因需长文本 SFT，仅与 Full Attention 比（稀疏基线不支持训练）。
- 效率（Figure 1）：64k 序列下 **decode 11.6×、forward 9×、backward 6×** 加速，且序列越长加速比越大。

**【⚠️ 存疑】** NSA 必须**从头预训练**（270B token + MoE 27B），算力门槛极高，普通团队无法复现或应用到已有 dense 模型。它解决的是"训练新模型"，对"加速已部署的现成推理模型"（RaaS/Quest/SeerAttention-R 的场景）不适用——这也是 Lil 把 NSA 明确划在 PTSD scope 之外的原因。

**【⚠️ 存疑】** "稀疏反而超 Full Attention" 的解释（过滤噪声）是事后归因，缺乏机制性证据；也可能与超参调优、MoE 结构、训练随机性混淆。同尺寸 Full Attention 是否被充分调优值得追问。

**【⚠️ 注】** 三分支 + 门控 + 三套独立 K/V 增加了参数与实现复杂度；门控分支学不好可能退化。

---

## 4. 局限性与未来方向

作者未设独立 Limitations。可总结：
1. **需从头训练**，无法 plug-and-play 到现成模型。
2. 仅在自家 27B MoE backbone 验证，架构/规模泛化性未独立检验。
3. 超参（block 大小、n、w）较多，跨任务最优配置未充分探索。

未来：适配更多架构、降低训练成本的稀疏适配（这正是 SeerAttention-R 等"轻量适配"路线的切入点）。

---

## 5. 评价

**贡献为何有效**：NSA 的力量在于"原生"——它不是给训练好的 dense 模型打补丁，而是让模型**在预训练中就学会稀疏**，从而稀疏模式与任务分布深度对齐（不偏离优化轨迹），甚至因过滤噪声而超越 full。同时三个硬件对齐设计（blockwise 选择、压缩分数复用、GQA 组共享）让理论稀疏真正转化为 decode/forward/backward 全阶段加速。DeepSeek 工业级验证 + ACL Best Paper，可信度高。

**启发/借鉴价值**：
- 代表"训练时稀疏（training-aware）"范式，与 Quest/RaaS（training-free）、SeerAttention-R（轻量自蒸馏适配）形成完整谱系。本研究方向需明确定位在这个谱系的哪一段。
- 三分支（粗压缩 + 细选择 + 局部窗）的层次化设计是可借鉴的 mask 组合思路；本方向"mask + 门控 + entmax 三层组合"可参照其门控融合多分支的做法。
- **GQA 组共享块选择**解决 Quest 在 GQA 下"减算力不减内存访问"的痛点，是 [[seerattention-r]]/[[nosa]] 也采用的关键设计——本方向若涉及内存访问优化必须考虑。
- §2.1/§2.2 对现有稀疏方法的系统批判（phase-restricted、GQA 不兼容、不可训练组件）是很好的 related work 框架，可直接借用其分类视角。
- NOSA 正是针对 NSA "KV 访问无约束、offloading 时触发大量 CPU→GPU 传输"的弱点而提出。

**关联**：[[quest]]（被批判的 GQA 内存访问问题）、[[seerattention-r]]（轻量适配 vs 从头训练；GQA 共享稀疏同源）、[[nosa]]（针对 NSA offloading 弱点）、[[lil]]（NSA 被划在 PTSD scope 外）、[[dms]]（另一条 training-aware 路线：淘汰感知训练）。
