# SeerAttention-R 精读笔记

**论文**：SeerAttention-R: Sparse Attention Adaptation for Long Reasoning
**作者**：Yizhao Gao, Shuming Guo, Shijie Cao 等（Microsoft Research / HKU / HUST / PKU / Tsinghua）
**出处**：arXiv 2506.08889v1（2025年6月）
**代码**：https://github.com/microsoft/SeerAttention

---

## 1. 研究背景与动机

**问题**：推理模型靠 test-time scaling 生成更长 CoT 来提升能力，但自回归 decode 中后续 token 要 attend 越来越长的 context，单 token 成本线性增长、整体成本二次增长，KV cache 成为瓶颈。原版 SeerAttention 针对 **prefill**，对 query 做 sequence 维 pooling，**不兼容**逐 token 的自回归 decode。

**核心 insight（Oracle Sparsity 实验，§4.2）**：作者用 oracle block sparse selection（用训练 ground truth 选块）证明——推理模型的注意力**本身就是稀疏的**，仅激活一小部分重要 token 即可保持推理能力（block size 32/64 下 2k token budget 即近无损）。挑战在于如何高效**识别并利用**这种内生稀疏。

---

## 2. 方法详解

**核心思路**：扩展 SeerAttention 的 self-distilled AttnGate（注意力门控）到 decode 场景——加一个轻量可学门控插件，learn 出哪些 KV block 重要，**冻结原模型权重**，仅训门控。

**关键设计：**
1. **去除 Query 的 sequence-pooling（最核心改动）**：decode 时 Q 是逐 token 的，不能在 sequence 维压缩。仅对 K 做 pooling（公式 1a-1c）。
2. **GQA 对齐的共享稀疏**：用一个 linear layer 把 Q 从原 query head 数投影到 KV head 数（如 32 query→8 KV，group size g=4），使同一 GQA 组共享稀疏决策——与 NSA/SAAP 实践一致，提升内核效率。
3. **K 的 pooling-based 压缩**：对 K 做 Max+Min+Avg 三种 pooling 拼接（kernel/stride = block size，非重叠 chunk pooling）。Max/Min 捕捉 outlier，Avg 保整体分布。
4. **AttnGate 内的位置编码**：对 pre-RoPE 的 Q、K 重新施加 RoPE（block 内用首 token 的位置索引），实测比无位置编码更准。
5. **自蒸馏训练（§2.3）**：ground truth = 原模型注意力图的 **1D column-wise maxpool**（decode 不压 sequence），并在 GQA 组内再 maxpool 到 KV head，归一化后用 **KL 散度** 训 AttnGate。提供改造版 FlashAttention-2 kernel 直接在前向中产出 ground truth，复用 block-level rowmax，训练高效。**仅需 0.4B token（OpenR1-MATH-220K）**。
6. **推理（§3）**：用 **K Compression Cache** 缓存压缩后 K（每生成一个 block 才更新一次），AttnGate 无需重算历史 K branch。block size=64 时这个 cache 仅占原 KV 的 1/128（<1%）——还可把大 KV offload 到 CPU，只取激活块。
7. **稀疏化策略**：token budget（top-k，便于对比）或 threshold（更自适应，高稀疏区精度略好）。
8. **TileLang Block Sparse Flash Decoding Kernel**：三维 launch（batch, heads_kv, num_split），按 max_selected_blocks 沿 num_split 划分以负载均衡，用 wgmma 指令、padding query head 组到 64。

**创新点**：首个面向**长推理 decode** 的稀疏门控适配；大块稀疏（64/128）+ TileLang 内核把实际加速逼近理论上限；可学门控 > 训练免训启发式。

---

## 3. 实验设计评估

- **模型**：Qwen3-4B/8B/14B、R1-Distill-Qwen-14B（均 GQA 架构）。
- **基准**：AIME24、AIME25、MATH-500、GPQA-Diamond。max output 32,768。pass@1 平均（AIME 64 次、MATH 8 次、GPQA 16 次采样）。
- **基线**：Full attention、Quest（training-free，query-aware KV 选择）。为公平，Quest 与 SeerAttention-R 都设 block size 64、全层稀疏。
- **训练**：OpenR1-MATH-220K，global batch 16、800 步、AMD MI300x、lr 1e-3、DeepSpeed ZeRO-2。

**主要结果：**
- Oracle 稀疏（Figure 4）：block 32/64 下 2k budget 近无损，证明内生稀疏存在。选 block size 64 为默认。
- vs Quest（Figure 5）：**所有模型/基准上一致优于 Quest**。AIME24 上 SeerAttention-R 4k budget 近无损，Quest 即使 8k 也达不到；MATH/GPQA 上 SeerAttention-R 2k 即够，Quest 需约 8k。
- **模型越大，稀疏容忍度越高**：14B 比 4B/8B 更易闭合与 dense 的差距，Quest 在大模型上低预算 gap 收缩更明显。
- 内核加速（Figure 6）：H100 上 batch16/seqlen≥32k、90% 稀疏时 TileLang 内核相比 FA3 近 **9× 加速**（接近理论上限），比 Triton 实现快 1.7×。decode 内核是 I/O-bound，序列越长 batch 越大加速越显著。
- 消融：
  - block size（Figure 7）：Quest 随 block 增大精度下降，SeerAttention-R 在 16–128 上**几乎不变**（得益于 GQA 组共享稀疏）。block 16 因训练 OOM 被排除。
  - hybrid dense 前两层（Figure 8）：Quest 加 dense 前两层显著涨点，SeerAttention-R 仅微涨——因其前两层稀疏预测已足够准。
  - threshold vs token budget（Figure 9）：threshold 更自适应、高稀疏区略优。
  - **§5.4：不准的稀疏（预算太小/recall 低）会增加输出长度**——与 Lil 现象一致，作者认为这是 post-training 效率优化的普遍效应（量化也有，§5.4 引 [43]）。

**【⚠️ 存疑】** 训练仅用数学数据（OpenR1-MATH-220K），却在 GPQA-Diamond（科学QA）上评测并宣称有效。门控在 OOD 任务上的泛化（数学训练→非数学推理）证据有限，可能存在领域过拟合风险。

**【⚠️ 存疑】** 与 Quest 比较时强制 Quest 全层稀疏、关掉其默认的"前两层 dense"，虽为隔离 hybrid 影响，但这并非 Quest 的推荐配置，对 Quest 略有不利（不过 §5.2 单独补了 hybrid 消融，部分缓解）。

---

## 4. 局限性与未来方向

作者未设独立 Limitations 节。从内容看：
1. **需要训练**（虽仅 0.4B token、仅门控）：不如 Quest/RaaS 等 training-free 即插即用。
2. **训练数据领域单一**（数学），跨域泛化未充分验证。
3. **不准稀疏会增加生成长度**（§5.4），与 Lil 同源问题，本文未提供解法。

我认为未解决：
- 门控对每个新模型都要重训；与持续演进的模型族（Qwen3-Next 等）适配成本。
- block size 固定 64，未做 per-layer/per-head 自适应块大小。

---

## 5. 评价

**贡献为何有效**：抓住两点——(1) 推理注意力内生稀疏（oracle 实验证明上界），(2) 用**轻量可学门控 + 自蒸馏**比训练免训的启发式（Quest 的 min/max 上界估计）更精准地逼近这个上界，且冻结主干、训练成本极低（0.4B token）。大块稀疏（64）配 GQA 共享 + TileLang 内核，让"高 recall 的块选择"与"高硬件利用率"同时成立，把理论加速真正落地（9×）。

**启发/借鉴价值**：
- 是推理 decode 稀疏方向**事实上的锚点工作**，代表"训练（自蒸馏门控）"路线，与 [[quest]]/[[raas]]/[[lazyeviction]]（training-free）路线形成范式对照。
- 对本研究方向："统一 CoT 注意力 + 无需自蒸馏的理论指导稀疏 mask"正是要与 SeerAttention-R 的自蒸馏路线做**天然对照卖点**——本方向主张 zero-shot/理论指导，SeerAttention-R 是 learned/data-driven。oracle 稀疏实验也为本方向"稀疏可行性"提供了现成证据。
- K Compression Cache 仅 1/128 开销 + 可 offload 的设计，与 [[nosa]] 的 offloading 思路相通。
- §5.4 再次印证 Lil 现象（稀疏→生成变长），强化了端到端评测的必要性。

**关联**：[[quest]]（核心对照基线）、[[nsa]]（GQA 共享稀疏、训练时稀疏的姊妹思路）、[[raas]]（同为推理 decode 稀疏起点，模式 vs 系统两条线）、[[lil]]（稀疏致长现象）、[[nosa]]（offloading）。
