# NOSA 精读笔记

**论文**：NOSA: Native and Offloadable Sparse Attention
**作者**：Yuxiang Huang, Pengjie Wang, Xu Han, Zhiyuan Liu 等（Tsinghua NLP Group / OpenBMB / BUPT / BIT）
**出处**：arXiv 2510.13602v2（2026年1月）
**代码**：https://github.com/thunlp/NOSA

---

## 1. 研究背景与动机

**问题**：decode 吞吐受限于 GPU 显存，而显存主要被 KV cache 占据。batch 越大吞吐越高，但 KV 限制了 batch 大小。**KV cache offloading**（把大部分 KV 放 CPU，只取稀疏子集到 GPU）是解法，但现有两条路都有缺陷（Figure 1）：
- **Training-free offloading（InfLLMv2、ShadowKV、ArkVale）**：训练-推理稀疏模式不匹配，long-generation 任务上选择错误累积、精度下降。
- **Trainable sparse attention（NSA、InfLLMv2 trainable）**：与高效 offloading 不兼容——KV 访问**无约束**，可能触发大量 CPU→GPU 传输，抹掉吞吐收益（成为 communication-bound）。

**核心洞察（§2.2 两个 Observation）**：
- **Observation 1（Locality）**：trainable sparse attention 在相邻 decode step 间的 token/block 选择有**内生局部性**（locality γ(t)，定义为相邻步选择重叠率）。实测 16K 序列上多数层 γ(t)≥0.8——每步变化的 KV <20%，意味着可缓存复用、减少通信。
- **Observation 2（PCIe Bound）**：但 ~80% 的 locality 仍不足以缓解 PCIe 瓶颈（>80% decode 时间在 attention，仍 communication-bound）。

**结论（Takeaway）**：要高吞吐，必须在**训练时**显式施加约束以提高 cache hit rate → 设计原生可 offload 的稀疏注意力。

---

## 2. 方法详解

NOSA = **Algorithm（NOSA）+ System（NOSI）协同设计**。

**算法 NOSA（§3.1，Offloading-aware Trainable Sparse Attention）**：
- **核心：加 locality 约束**。要求所有 step 的 locality γ(t) ≥ 下界 γ₀。
- **KV 选择分解为两部分**：
  - **Query-aware selection Γ_q(t)**：不加约束，保留模型回看远距离信息的能力（用 s^q = QKᵀ 选 top-k_q block）。
  - **Query-agnostic selection Γ_e(t)**：天然带 eviction 约束（t 时未选则之后都不选 → 固定淘汰模式 → 强局部性、零通信）。用 **eviction head**（基于 DMA 的可训练 KV 淘汰）给每个 KV 算重要性分 s^e_j = τ(v_j W₁)W₂，选 top-k_e。
  - 总预算 k = k_q + k_e。**Theorem 2**：此选择过程保证 γ(t) ≥ k_e/k（locality 下界）。
- **Block-wise selection（适配 offloading）**：element-wise 选择导致碎片化 host-to-device 传输、PCIe 带宽利用低；NOSA 在 InfLLMv2 上做两阶段块选择（先 sub-block mean pooling、再 block max pooling）。
- **ED-DMA（Exp-Delayed DMA）**：query-agnostic 选择对数值精度敏感，去掉 token 选择前的 exp 操作（基于 pre-exponential 分数选 top-k），延迟到 attention 计算阶段——消融证明 ED-DMA 最稳最优（vanilla DMA 的 exp 操作引入显著精度损失）。
- **可训练**：eviction head 通过 attention bias b 接收梯度。
- **训练流程**：vanilla attention 短上下文预训练 → 切到 NOSA 做长上下文 continual pretraining → SFT 时也用 NOSA，交付内建 offloading 能力的模型。

**与 DMA 区别**：DMA 是 element-wise + 纯 query-agnostic；NOSA 是 block-wise + query-aware & query-agnostic 结合（query-aware 保证 recall，DMA 无 recall 能力）。

**系统 NOSI（§4，Inference System for NOSA）**：
- **Kernel Fusion**：用 FlashInfer 的 LayerNorm/RoPE/FFN；把 eviction head 与 QKV 拆分融合成单 kernel；两个 pooling 融合；CUDA Graphs 减 launch 开销。
- **Memory Layout & Block Selection**：KV 在 GPU/CPU 都按最小 block 单位组织；decode 不需 GPU 上 KV 连续，故只换出当前步未用的 block、用新选的替换（解 set-difference 问题，自定义 O(k) CUDA kernel）。
- **Offloading Communication**：每 block 仅 16KiB，碎片化传输；用自定义 Triton kernel + UVA（Unified Virtual Addressing）直接访问 CPU 内存，达 **83% PCIe 峰值带宽**（vanilla PyTorch <2GB/s）。

---

## 3. 实验设计评估

- **模型**：1B/3B/8B，4K 预训练 → NOSA 续训扩到 16K（k=4096, k_q=1024）→ SFT。
- **任务**：LongBench、HELMET（长上下文）；general tasks（MMLU 等）；reasoning tasks（Math-500、Gaokao，8B）；效率 benchmark。
- **基线**：FullAttn、InfLLMv2、DMA（trainable）；ShadowKV、InfLLM、ArkVale（training-free offloading）；vLLM、SGLang（效率）。

**主要结果：**
- 长上下文（Table 2）：12 设置中 NOSA 拿 9 个最高分。**hard retrieval（PR/Recall）任务上**，training-free 基线掉 ≥10%，NOSA 仅轻微退化（query-aware 选择保 recall；DMA 因无 recall 能力在 recall 任务上差）。
- 模型越大越稳：1B 上稀疏方法普遍 recall 掉 >20%，8B 上 NOSA 退化轻微——稀疏注意力受益于更大模型容量。
- general tasks（Table 3）：NOSA 与 FullAttn/DMA 无显著差距（短上下文不损能力）。
- reasoning tasks（Table 4，8B）：training-free offloading（ShadowKV）大幅掉点；NOSA/DMA 因稀疏模式 learned 而更好，NOSA 平均 49.6 ≈ FullAttn 51.6。Figure 7 案例：ShadowKV 困惑度随生成快速上升（误差累积），NOSA 保持低困惑度。
- 效率（Table 5）：offloading 方法（ShadowKV、NOSA）因可放大 batch 而吞吐高于非 offloading。NOSA 最高 **5.04× vs FullAttn、1.92× vs InfLLMv2、1.83× vs ShadowKV**（input 96K, EB=64，on-GPU KV 匹配时 batch 大 16×）。
- locality（Figure 2d）：NOSA 比 InfLLMv2 locality 高 ~5.5%（cache miss rate 近 2× 下降）。
- 消融：ED-DMA 优于 LoCRet/vanilla-DMA/S-DMA（Table 1）；block-wise 通信吞吐远高于 element-wise（Figure 4a）；query-aware 在训练与推理都必要（Figure 4b）。

**【⚠️ 存疑】** NOSA 需**长上下文 continual pretraining + SFT 全程用 NOSA**，训练成本高（虽不像 NSA 从头训）。对已部署的现成推理模型不能直接用，与 RaaS/Quest 等 training-free 方法的适用场景不同。

**【⚠️ 存疑】** locality 下界 γ(t)≥k_e/k 是靠"固定 k_e 比例的 query-agnostic 淘汰"硬性保证的——这等于强制一部分 KV 用静态淘汰模式。若任务需要大量动态远距离回看（k_e 占比挤压 k_q），recall 与 locality 之间存在内在张力，k_q/k_e 配比的最优性依赖任务，泛化性待验证。

**【⚠️ 存疑】** 效率优势高度依赖自研 NOSI 系统（kernel fusion + UVA + 自定义 CUDA kernel）；vanilla HF 实现下 NOSA 反而慢于 InfLLMv2。即"算法收益需配套系统工程才能兑现"，复现门槛高。

---

## 4. 局限性与未来方向

作者未设独立 Limitations。可总结：
1. **需训练**（continual pretraining + SFT），非即插即用。
2. **效率依赖 NOSI 系统**，脱离专用 kernel 收益打折。
3. **小模型上稀疏退化明显**（1B recall 掉点），方法偏好大模型。
4. k_q/k_e 配比、locality 下界 γ₀ 等超参的任务自适应未充分探索。

未来：更大规模验证、自适应 locality 约束、与 reasoning long-decode 场景的深度结合。

---

## 5. 评价

**贡献为何有效**：NOSA 精准指出 trainable sparse attention 与 KV offloading 长期不兼容的根因——**KV 访问无约束 → PCIe 通信爆炸**，并给出优雅解法：把选择分解为 query-aware（保 recall）+ query-agnostic（带天然淘汰约束、强局部性），用 Theorem 2 给出 locality 下界的**形式化保证**，从而在训练时就让稀疏模式"对 offloading 友好"。算法-系统协同（NOSA + NOSI）是把 locality 收益真正兑现为吞吐的关键。是首个统一"可训练稀疏 + KV offloading"的工作。

**启发/借鉴价值**：
- 对本研究方向：NOSA 把"带宽/通信约束"显式纳入稀疏设计，提示本方向若考虑系统部署，**显存带宽与 CPU-GPU 传输**是 decode 稀疏不可忽视的维度（note.md 已标注 NOSA "显式设置预算约束带宽，把 sa offload 到 kvcache"）。
- **query-aware + query-agnostic 分解**是可复用的设计模式——动态选择保 recall、静态淘汰保局部性/带宽，对应本方向 taxonomy 中"动态回看"与"固定淘汰"两类机制的组合。
- locality 下界的**形式化定理（Theorem 2）**是少见的理论保证，与本方向"per-pattern 误差界"主张方向一致（虽 NOSA 给的是 locality bound 而非误差 bound）。
- 是回答"NSA 训练稀疏对 offloading 是否有帮助"的直接实证，与 [[nsa]] 高度互补——NSA 解决训练，NOSA 解决训练稀疏的 offloading 落地。
- ED-DMA 揭示 query-agnostic 选择对 exp 数值精度敏感，是实现层面的有用经验。

**关联**：[[nsa]]（针对 NSA 的 offloading 不兼容弱点；同属 trainable native sparse 谱系）、[[seerattention-r]]（K Compression Cache 可 offload 思路相通、GQA 共享稀疏）、[[quest]]（query-aware 选择 vs query-agnostic 淘汰的谱系）、[[dms]]/[[lazyeviction]]（eviction head/淘汰策略相关）。
