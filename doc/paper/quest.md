# Quest 精读笔记

**论文**：Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
**作者**：Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han（SJTU / MIT / UW / NVIDIA）
**出处**：ICML 2024（arXiv 2406.10774v2）
**代码**：https://github.com/mit-han-lab/Quest

---

## 1. 研究背景与动机

**问题**：长上下文（128K/1M）LLM 推理慢，根因是 decode 阶段每步都要加载整个 KV cache 做 self-attention。Llama-7B 32k context 的 KV cache 占 16GB，加载需约 11ms、占 >50% 推理延迟。decode 主导端到端（16k prompt + 512 输出时 >86% 时间在 decode）。

**已有方法不足**：
- KV eviction（H2O、TOVA、StreamingLLM）基于**历史信息或当前状态永久丢弃** token，但被丢的 token 可能对**未来** query 重要 → 召回率低。
- **核心 insight（Figure 2）**：token 的 criticality 是**动态的、强依赖于当前 query**。例 "A is B. C is D. A is"——在最后的 "is" 之前 "B" 不重要（低注意力），但当 query 是 "is" 时 "B" 变得 critical。一旦永久淘汰，未来需要时就回不来了。
- Figure 4 量化：H2O 在 passkey retrieval 上 Top-10 recall 仅约 0.3–0.5，Quest 接近 full attention。

**结论**：不应永久丢弃 KV，而应**保留全部 KV、根据当前 query 动态选择**要 attend 的关键部分。

---

## 2. 方法详解

**核心思路**：query-aware 的 page 级 KV 选择。保留全部 KV（故内存 O(N)），但每步只把 query 与"最关键的 Top-K page"做 attention，减少内存搬运。

**关键设计（Figure 5，两阶段）：**
1. **Page metadata**：基于 PageAttention（vLLM），每个 page（如 16 个 KV）记录其内 Key 向量的**逐 channel max（M_i）和 min（m_i）**作为元数据。
2. **Stage 1 — Criticality estimation**：对当前 query Q 与每个 page 的 (m_i, M_i) 算**注意力上界**：每个 channel 取 `U_i = max(Q_i·m_i, Q_i·M_i)`（不论 Q_i 正负，U_i 都 ≥ 该 page 内任何 key 的该 channel 乘积），`score = Σ_i U_i` 即该 page 注意力的上界估计。选 score 最高的 Top-K page。
3. **Stage 2 — Sparse attention**：只加载选中的 Top-K page 做真正的 self-attention。
4. **首两层保持 dense**：Figure 3 显示前两层稀疏度 <10%（query-aware 稀疏潜力低），故前两层不做 Quest（正交于 KV 选择算法）。
5. **内存节省**：估计阶段加载元数据 ~2M·L/S，attention 阶段加载 2M·K·S，全 KV 是 2M·L → Quest 加载 `1/PageSize + K/PageNum` 比例。Top-K 算子开销仅 5–10us，可忽略。

**与已有方法本质区别**：从"永久淘汰（query-agnostic）"转向"全保留 + query-aware 动态选择"，用 page 内 min/max 上界做廉价 criticality 估计；page 粒度兼顾选择精度与 GPU 友好。

---

## 3. 实验设计评估

- **模型**：LongChat-7b-v1.5-32k、Yarn-Llama-2-7b-128k。
- **任务**：PG19（语言建模困惑度）、passkey retrieval（10k/100k）、LongBench 六数据集（NarrativeQA、Qasper、TriviaQA、HotpotQA、GovReport、MultiFieldQA）。
- **基线**：H2O、TOVA、StreamingLLM、Full。
- **内核**：基于 FlashInfer 的 CUDA 实现，Top-K 用 RAFT；NVBench/PyTorch profiler 测延迟。

**主要结果：**
- 语言建模（Figure 6）：Quest（budget 4096）困惑度紧贴 full cache。
- Passkey（Table 1）：H2O/TOVA/StreamingLLM 因丢弃了"答案"KV（在问题到达前就被淘汰）几乎全错（1%）；**Quest 用 64/1024 token budget（约总长 1%）即近完美**——证明全保留 + 动态选择对长依赖任务的价值。
- LongBench（Figure 7）：Quest 在六数据集全面超基线，多数 **1K budget 即接近 full**；考虑前两层 full 后，在多个任务以 1/6~1/10 稀疏度无损。
- 效率：32K context、budget 2048 时 self-attention 加速 **7.03×**（Figure 9），端到端 1.74×（FP16）/ **2.23×（4-bit AWQ）**。同精度约束下（Figure 11），相比 TOVA 自注意力加速 3.8~4.5×（TOVA 需更大 budget 才无损，如 NarrativeQA TOVA 需 14k、Quest 仅 5k）。

**【⚠️ 存疑】** criticality 用 page 内 min/max 算注意力**上界**，是乐观估计——一个 page 只要有单个 channel 极值大就可能被高估为 critical（实际只有少数 token 高分）。page 越大、内部越异质，上界越松，精度越掉（这正是 SeerAttention-R §5.1 揭示的：Quest 随 block size 增大精度显著下降）。

**【⚠️ 存疑】** O(N) 内存是 Quest 的根本短板——它只省**带宽/算力**不省**显存**。在 RaaS 等强调 O(L) 内存的工作里，这是被针对的弱点（long-decode 推理场景 KV 持续膨胀，内存才是瓶颈）。

**【⚠️ 注】** Quest 面向**长 prefill / 长上下文检索**任务（LongBench、passkey），并非为推理模型 long-decode 设计；但因其 query-aware 选择范式通用，成为后续推理 decode 稀疏工作（RaaS、R-KV、SeerAttention-R、LazyEviction）的**事实标准基线**。

---

## 4. 局限性与未来方向

作者未设独立 Limitations。可总结：
1. **O(N) 内存不降**：只减带宽不减显存，long-decode 场景内存仍是瓶颈。
2. **上界估计随 page size 变松**：大 block 精度下降。
3. **前两层需 dense**：非全自动，需人工指定 dense 层。
4. 仅在 7B 模型、长上下文检索类任务验证，未涉及推理模型 long CoT。

未来可改进：自适应 page size / 上界更紧的 criticality 估计 / 与内存压缩（量化、淘汰）结合以同时降显存。

---

## 5. 评价

**贡献为何有效**：精准指出前人"永久淘汰"的致命缺陷——**criticality 是 query-dependent 且动态的**，淘汰即不可恢复。Quest 用"全保留 + page 级 query-aware 动态选择"绕开召回率问题，并用 min/max 上界把 criticality 估计做得极廉价（每 page 一个分数），page 抽象又对 GPU/PageAttention 友好，因此 7× 自注意力加速且近无损。是 decode 阶段 KV 选择的奠基工作。

**启发/借鉴价值**：
- 是本研究方向**最核心的对照基线**（note.md 明确标注"最核心基线"）。任何新方法都需在精度/带宽/显存三轴上与 Quest 对比。
- "全保留 KV + 动态选择" vs "淘汰" 是两条根本不同的路线：本方向若做推理 decode，需明确自己更接近哪条，并针对 Quest 的 O(N) 内存短板或上界估计松弛做差异化。
- page min/max 上界、Top-K page 选择是可复用组件（RaaS 的 page 代表选择即沿用 Quest）。
- 局限提示本方向：推理 long-decode 场景 KV 持续增长，**内存（O(L)）比带宽更关键**，这是 RaaS/R-KV 相对 Quest 的立足点。

**关联**：[[raas]]（针对 Quest O(N) 内存提出 O(L)）、[[seerattention-r]]（直接对照、揭示 Quest 大 block 掉点）、[[lazyeviction]]/[[r-kv]]（均以 Quest 为基线谱系）、[[nsa]]（训练时稀疏的另一范式）。
