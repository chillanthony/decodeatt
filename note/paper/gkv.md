# 精读笔记：G-KV — Decoding-Time KV Cache Eviction with Global Attention

**论文信息**
- 作者：Mengqi Liao, Lu Wang, Chaoyun Zhang, Zekai Shen, Xiaowei Mao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Huaiyu Wan（北京交通大学, Microsoft）
- arxiv：2512.00504v1（2025-11-29，preprint）
- 代码：https://github.com/microsoft/G-KV
- 关键词：decoding-time KV 淘汰、global score（全局打分）、memory decay rate、推理任务、training-free + RL/distill 后训练

---

## 1. 研究背景与动机

### 现有方法的问题
- 长 CoT 推理生成数千~数万 token，注意力随序列**二次增长**成关键瓶颈。
- **prefill 压缩方法**（SnapKV、KVzip、PyramidInfer、PyramidKV、Ada-KV、HeadKV）只压缩 prompt 的 KV；但推理任务的 output 远长于 input，仅压 prompt 收益边际。
- **decoding 淘汰方法**（H2O、MorphKV、Song et al.、R-KV）支持生成时淘汰，但**只依赖最近少数生成 token 的局部注意力分数（local attention score）**决定淘汰。论文核心批判：**token 重要性在生成过程中会漂移（shift）**，局部视角忽略 token 的 long-term importance，会误杀未来会复现的关键 context。
- 此外原模型未适配 KV 压缩诱导的稀疏注意力模式，导致次优；pre-training 稀疏注意力（NSA 等）成本过高。

### 核心 insight（4 节 Observation）
- 用 DeepSeek-R1-Distill-Qwen-7B 在 AMC2023 跑 32 rollouts，把最后 512 token 分成 4 个 observation window，比较各窗口高分 token 集合的 overlap：
  - **Observation 1**：last window 关注的 token 与 earlier window 关注的**不完全一致**，保留率越低不一致越明显。→ 重要性跨窗漂移；单窗口打分会过早淘汰具长期重要性但被暂时忽视的 token。
  - **Observation 2**：last window 与**所有 preceding window 并集**的 overlap 明显更高——即便只保留 55% token，overlap 也接近 **95%**。→ token 注意力是**间歇性（intermittent）**的；获得显著注意力的 token 极可能在至少一个之前的窗口里也被关注过。
- 人类记忆类比：被多次重温的记忆愈发巩固，长期不复习的逐渐消退 → 设计带 **memory decay rate** 的 global score。

---

## 2. 方法详解

核心思路：用结合 **local attention score（当前窗口）+ historical score（衰减累积的历史窗口）** 的 **global score** 评估 token 长期重要性，training-free 地指导 decoding 淘汰；再用 RL/distill 后训练让模型适配压缩后的稀疏注意力。

### 统一框架与 local score（3 节）
- 每生成 s 个 token 压缩一次；最近 w 个生成 token 构成 observation window。
- local score：observation window 内 query 与 cached key 的注意力（式1），GQA 下对 group 内 head 做 max-reduce 得 A'，再在窗口内做 mean 得 `S_{i,j}=(1/w)Σ A'_{i,k,j}`（式2）。保留 top-(b−w) 分数对应 KV，最终压到预算 b。

### Global Score（5 节，核心）
- 引入 memory decay rate α∈[0,1]，把历史全局分 F_{t-1} 衰减后与当前 local score S_t 结合。三种形式（式3-5，S_t 先用 head 内 max 归一化）：
  - **max**：`F_t = max(α·F_{t-1}, S_t/max_j S_t)`（式3）
  - **average**：`F_t = α·F_{t-1} + (1−α)·S_t/max(S_t)`（式4）
  - **summation**：`F_t = α·F_{t-1} + S_t/max(S_t)`（式5）
- 基于 F_t 选 b−w 个保留，记录其 F_t 供下次压缩。首次压缩无 F_{t-1} 时直接用 S_t。
- 四种情形（Fig.2 直觉）：
  - low F_{t-1} & low S_t → 持续无注意力，**淘汰**；
  - low F_{t-1} & high S_t → 之前失宠但当前重获注意力，**保留**（重新激活）；
  - **high F_{t-1} & low S_t** → 当前没注意但历史被高度关注，**保留**（区别于 local 方法会淘汰它——这是 G-KV 关键差异，因这些 token 极可能未来再被关注）。

### 后训练适配（6 节）
- 把 KV 压缩视为 sparse attention，原模型 π_θ（全注意力训练）在压缩环境 π'_θ 下次优。
- **RL-Sparse**：用 π'_θ 直接采样，记录实际被淘汰 token 的位置构造 sparse attention mask，用 GRPO 目标（式6，无 KL 正则）优化；对因 max output length 截断的输出，把其优势设为 0。
- **Distill**：对难以设计 verifiable reward 的场景，用蒸馏式方法让压缩 π'_θ 逼近 π_θ（Appendix C）。

### 创新点
1. global score 结合 local + 衰减历史，捕捉 token 长期重要性（克服单窗口局部视角）。
2. memory decay rate 把"间歇性注意力"建模为可调衰减。
3. 专为 KV 压缩设计的 RL 后训练（RL-Sparse），消除 training policy 与 inference policy 的差异——优于在 Full KV 上做 RL。

---

## 3. 实验设计评估

### 设置
- 模型：DeepSeek-R1-Distill-Qwen-7B、DeepSeek-R1-Distill-Llama-8B。
- 基准：AMC2023、AIME2024（数学）、LiveCodeBench（编程）；RL 训练用 DeepScaleR-40k，蒸馏采 27k 正确 trace。
- 指标：pass@1（32 次采样），temperature 0.6 / top-p 0.95，AMC 16k / AIME 32k 长度。
- 基线：StreamingLLM、MorphKV、SnapKV（扩展到 decoding）、R-KV。预算 b=512 起，观察窗 w=16，压缩间隔 s=128。
- 额外指标：**Token Retention Ratio**（正确回答时 KV 长度 / 总序列长度，越低说明固定预算下能处理更长生成）。

### 主要结果
- **global score 消融（Fig.4）**：三种 global score 形式都显著超 local score 及 CAKE 的 attention shift score、MorphKV；mean 形式略差；α∈[0.8,0.9] 最稳最优（推荐 α=0.8）。
- **主对比（Fig.5）**：G-KV（global max + redundancy score，α=0.8）在多数预算/基准 SOTA，**预算越小优势越大**，AMC23 512 预算下近 **+20%**。
- **Token Retention Ratio（Fig.6）**：G-KV 多数场景保留比例最低，说明保留 token 信息密度更高、能在更长序列下有效工作。
- **保留位置分布（Fig.7）**：local score 方法保留 token 集中在序列尾部（因近观察窗 + RoPE 语义相似），G-KV 保留位置**更均匀分布**，保留更全面的信息——解释了它在长生成/低预算下的优势。
- **后训练（Table 1）**：RL-Sparse 全面最佳，显著超在 Full KV 上做 RL 的 RL-Full（AMC23 512 预算 +6.01 vs RL-Full +2.65），证明"直接优化压缩策略 π'_θ 消除训练-推理差异"有效；Distill 是 reward 难设计场景的实用替代。
- **效率（Table 2-4）**：单 A100，DeepSeek-Qwen-7B 在 512/1024/2048 预算下吞吐 4.18×/3.41×/2.73×，更大 batch 下达 12.18×/19.7×；解码时间约 Full 的 40%；16k context 下 KV 内存约降 90%；global score 比 local 仅多约 5ms/步（占总解码 ~1%）。

### 评估合理性判断
- 三基准 + 两推理模型 + 32 次采样，对推理任务较规范；Token Retention Ratio 与保留位置分布（Fig.6/7）是有洞察力的诊断，解释"为何 global 更好"。
- RL-Sparse vs RL-Full 的对照很有价值，论证了"为压缩而训练"的必要性。
- 【⚠️ 存疑】**全为数学/编程推理基准，无通用长上下文任务**，泛化性未验证（与论文泛读定位"中等相关、无纠错语义维度"一致）。
- 【⚠️ 存疑】最终 G-KV 是 "global max + R-KV redundancy score" 的**组合**，纯 global score 相对 R-KV 的独立增益在主图中被混合，拆分贡献不够清晰。
- 【⚠️ 存疑】α 固定 0.8/0.9 全程，不随任务/层自适应；memory decay 的"间歇性"建模较朴素（指数衰减），与 LazyEviction 的 MRI 周期建模相比缺乏对"复发间隔"的显式刻画。

---

## 4. 局限性与未来方向

### 作者明确指出
- RL 方法仅适用于 reward 可验证的任务，故另提 distill 作通用替代。
- 训练对超长截断输出把优势设 0 以避免干扰。

### 我认为尚未解决的问题
- **场景局限于数学/编程推理**，通用长生成/长输入未测。
- global score 与 redundancy score 的组合使"global 单独贡献"不清晰。
- α 全局固定，缺自适应；衰减式历史累积无法区分"周期性复发"与"缓慢衰减"两种不同模式（LazyEviction 的 MRI 在此更细）。
- RL-Sparse 需对每个模型/预算单独训练，部署成本高于纯 training-free。

---

## 5. 评价

### 贡献为何有效
1. **Observation 2（55% 保留 → 95% overlap with union）是核心支撑**：它定量证明 token 注意力是间歇性的、且"重要 token 几乎都曾被某个历史窗口关注"，从而为"用历史全局分而非单窗口局部分"提供了直接依据。诊断 → 方法逻辑闭环。
2. **global score 简单且 training-free 即有效**：仅用一个衰减累积 + 归一化，就能保住"当前低注意力但历史高注意力"的长期重要 token（Fig.2 第四种情形），Fig.7 的保留位置均匀分布直观解释了它在长生成低预算下的优势。
3. **RL-Sparse 揭示"为压缩而训练"的价值**：指出在 Full KV 上做 RL 与压缩推理存在 policy mismatch，直接在 sparse mask 下优化能消除差异，Table 1 的 RL-Sparse > RL-Full 提供了有说服力的证据，是对"训练-推理一致性"的有益贡献。

### 启发与借鉴价值
- **对本项目（RescueKV idea）的定位**：G-KV 同为 decoding-time、training-free（核心部分）、推理基准，可作 RescueKV 的**对照后端之一**。但它用"衰减历史累积"刻画长期重要性，**无自我纠正事件这一语义维度**，且 α 衰减不等于 MRI 复发周期建模——这正是 RescueKV 的差异空间。
- 关键共鸣：G-KV 的 Observation（重要性漂移 + 间歇性注意力 + 历史关注过的 token 极可能复现）与 LazyEviction 的 TIR、RescueKV 的"复发"动机**高度一致**，可互相佐证"被局部视角误杀的关键 token 会复现"这一核心论点。
- 可借鉴方法学：① Token Retention Ratio 指标（固定预算下能否处理更长生成）；② Fig.7 保留 token 位置分布可视化，是论证"为何保留更全面信息"的好工具，RescueKV 可用类似图证明豁免救回了被尾部偏置淘汰的早期关键 token；③ RL-Sparse 的"训练-推理一致性"视角。
- 差异化提示：若做 training-free 路线，G-KV 的 global score（衰减历史）和 R-KV 的 redundancy 是常见基线组合；RescueKV 需强调"周期感知（MRI）+ 纠错事件触发"两个 G-KV 没有的信号，并验证叠加在 G-KV 之上仍有增益。

**一句话总结**：用结合局部 + 衰减历史的 global score 捕捉 token 长期重要性、training-free 地保住"当前低注意力但历史高关注"的间歇性关键 token，并以 RL-Sparse 消除训练-推理 policy mismatch，在数学/编程推理低预算下显著领先；短板是场景局限推理、α 固定无自适应、缺乏对复发周期的显式建模。
