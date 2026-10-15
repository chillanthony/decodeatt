# DuoAttention 精读笔记

**论文**：DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
**作者**：Guangxuan Xiao, Jiaming Tang, Song Han 等（MIT / Tsinghua / SJTU / Edinburgh / NVIDIA，MIT-Han Lab）
**出处**：ICLR 2025（arXiv 2410.10819v1，2024年10月）
**代码**：https://github.com/mit-han-lab/duo-attention

---

## 1. 研究背景与动机

**问题**：长上下文 LLM 缓存所有注意力头的 KV 消耗巨大内存（Llama-3-8B 服务 1M token FP16 KV 需 >137GB，超单卡 80GB），且 decode 延迟、prefill 延迟随长度增长。已有 KV 压缩方法要么损害长上下文能力（H2O、TOVA、StreamingLLM、FastGen），要么效率提升有限，且多与 GQA 不兼容。

**核心 insight（Figure 1）**：注意力头分两类：
- **Retrieval Heads（检索头）**：少数头，捕捉上下文中相关 token（如 "The best fruit is orange. What is the best fruit? Orange" 中解码第二个 "orange" 时回看前文），**需要全 KV**。压缩它们会丢失关键上下文。
- **Streaming Heads（流式头）**：多数头，只关注 attention sink + recent token，**只需常数长度 KV**。压缩它们对长上下文能力几乎无影响。

预实验（Figure 1 右）：剪检索头的中间 token 使 passkey retrieval 精度暴跌；剪流式头的中间 token 几乎无影响。

---

## 2. 方法详解

**核心思路**：给每个 head 分两套 KV cache——检索头用 full KV，流式头用常数长度（sink + recent）KV。关键是如何**准确识别检索头**。

**检索头的定义（§2.2）**：与前人（基于注意力分数 profiling 的 FastGen/Retrieval Head/RazorAttention）不同，DuoAttention 定义检索头为"**当限制到 recent + sink token 时显著改变模型输出的头**"——直接度量输出偏差，捕捉那些"注意力分数上不明显但对输出关键"的头（含 value 状态作用）。

**优化式识别（§2.2，Figure 2）**：
1. 给每个 KV head 分配可学门控 `α_{i,j} ∈ [0,1]`，初始为 1（全是检索头）。前向时混合：`attn = α·full_attn + (1−α)·streaming_attn`（streaming 用 Λ 形 sink+recent mask）。
2. **冻结模型权重，仅训 N×H 个门控值**（几千个浮点数）。
3. **合成数据集（Figure 3）**：在长文本中嵌入 10 个 32-word passkey，要求末尾召回。**蒸馏损失只在 passkey token 上算**（L2(full 模型末隐状态, mixed 模型末隐状态)），比自然语言建模监督信号更聚焦检索能力。
4. **L1 正则（Lasso）** 鼓励门控稀疏（公式 2）。总损失 `L = L_distill + λ·L_reg`（λ=0.05）。仅 2000 步、8×A100。

**部署（§2.3）：**
- 用阈值 τ（由稀疏分位决定）把门控**二值化**为检索/流式头。
- **Reorder Attention Heads**：按头类别重排 Q/K/V 投影输出通道，把检索头/流式头聚成连续簇，便于高效切片拼接（避免 scatter/gather）。
- **Decoding（Figure 5）**：两套 KV cache，检索头存全部、流式头存 sink+recent（常数）。
- **Chunked Pre-filling**：流式头在 prefill 时即时剪枝（只 attend 常数 token），prefill 复杂度 O(L²)→O(LK)、内存 O(L)→O(K)，无需专用 kernel。

---

## 3. 实验设计评估

- **模型**：Llama-2-7B-32K、Llama-3-8B-1048K（GQA）、Llama-3-70B、Mistral-7B-v0.2。配置：Llama-2-7B（MHA）检索头比 25%，Llama-3-8B（GQA）50%。
- **长上下文**：NIAH、LongBench（14/21 任务）。**短上下文**：MMLU、MBPP、MT-Bench。
- **基线**：H2O、TOVA、FastGen、StreamingLLM、Full。

**主要结果：**
- NIAH（Figure 6）：基线在不同深度丢失答案；DuoAttention 用 25%（MHA）/50%（GQA）full-attention 比例即匹配 full attention。
- LongBench（Figure 7）：KV budget–精度 trade-off 全面优于基线。
- 短上下文（Figure 8、Table 1）：50% budget 下近无损（Llama-3-70B MMLU 79.35 vs full 79.38，MT-Bench 9.14 > full 8.93）——不损害原能力。
- 效率：decode 最高 **2.55× 内存降 / 2.18× 延迟降（MHA）**、1.67×/1.50×（GQA）；prefill 1.73×/1.63× 延迟、2.38×/1.53× 内存。结合 8-bit 权重 + 4-bit KV 量化，单 A100-80G 可跑 Llama-3-8B **3.3M token**（6.4× 容量）。
- 门控分布（Figure 4）：MHA 模型检索头比例低于 GQA 模型。
- 消融（Figure 13）：(1) 优化式识别 >> 注意力 profiling；(2) 合成数据 >> 语言建模监督；(3) 优化时 sink+recent 都需要；(4) 部署时 16 sink + 64 recent 即够。

**【⚠️ 存疑】** 检索/流式的**二分**过于粗糙——后续 HeadKV 指出头能力是"检索 vs 推理"连续分布，RLKV 指出"检索头 ≠ 推理头"。DuoAttention 的二值化（按阈值 τ）会把中间地带的头强行归类，可能在推理任务上次优（本文未在推理模型/long CoT 上评测）。

**【⚠️ 存疑】** 检索头识别依赖 **passkey 合成数据**，本质度量的是"信息检索/复制"能力。对于推理任务，"重要的头"未必是检索头（RLKV 的核心论点）。因此 DuoAttention 的头分类**面向长上下文检索而非推理**，迁移到 long-decode 推理场景需重新验证。

---

## 4. 局限性与未来方向

作者未设独立 Limitations。可总结：
1. **二分粒度粗**：检索/流式过于简化，无法刻画头功能的连续谱。
2. **检索头数需预设比例**（25%/50%），非完全自动。
3. **面向检索任务训练门控**，未覆盖推理任务的"推理关键头"。
4. 流式头的 sink+recent 是固定结构，对需要中距离回看的任务可能不足。

未来：更细粒度头分类（HeadKV、RLKV 已沿此推进）、面向推理的头识别、自适应检索头比例。

---

## 5. 评价

**贡献为何有效**：DuoAttention 的关键是**用"输出偏差"而非"注意力分数"定义检索头**，并用聚焦于 passkey 的合成数据 + 可学门控 + L1 稀疏，把识别做得既准又轻（仅训几千个门控值、2000 步）。这避开了注意力 profiling 漏掉"低分但关键"头的问题。两套 KV cache + 头重排的工程设计让"头粒度差异化"真正落地为内存/延迟收益，并天然兼容 GQA 与量化。

**启发/借鉴价值**：
- **头功能分类方向的奠基性工作**（note.md 标注"头粒度差异化"），为 Ada-KV、HeadKV、RLKV 提供了"异构 KV cache"先例。
- 对本研究方向（头功能图谱 × KV 差异化）：检索/流式二分是**起点**，本方向可推向"reasoning head / receiver head（Thought Anchors）/ recurrence head"等更细分类，DuoAttention 是必须对比与超越的 baseline。
- "用输出偏差而非注意力分数定义头重要性"的方法论，与 RLKV "用生成质量（RL reward）直接优化"一脉相承——提示本方向头识别应锚定**任务结果**而非代理信号。
- 合成数据 + 轻量门控 + 蒸馏的训练范式，与 [[seerattention-r]] 的自蒸馏门控思路相通（同 MIT-Han Lab/相关团队）。

**关联**：[[seerattention-r]]（门控+蒸馏思路相通）、[[quest]]（同团队，KV 选择 vs 头分类两条路）、RLKV [[rlkv]]（指出检索头≠推理头，是对本文二分的细化与批判）、HeadKV（双维度连续打分）。
