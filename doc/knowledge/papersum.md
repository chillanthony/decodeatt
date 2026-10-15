### Papers

#### ICLR 2026 KV Cache

- [4] ThinKV (Oral): 精读；异质 CoT 分类，不同量化和驱逐策略，硬件适配；limitation 是不适合输入本身就很长的任务。
- [38] LookaheadKV: 精读；有开源代码，前瞻预测 soft token + KL 散度对齐，无 decoding，依赖学习。
- [43] Cache What Lasts / TRIM-KV: 在 token 创建时预测长期保留价值，并让分数随时间衰减；代表轻量 learned retention gate。
- [36] DefensiveKV: 淘汰脆弱性 + max 聚合最坏情况风险 + prior-risk 校正。
- [60] FreeKV: 不永久删除 KV，而是通过 CPU–GPU 分层存储、预测式 recall 和双缓冲隐藏检索延迟；是 eviction 的关键对照范式。
- [61] LouisKV: 利用关键 KV 的时间局部性，只在语义边界触发 retrieval，并区分长输入与长输出；是 long-input + long-output 设定的重要系统基线。
- [62] PM-KVQ: 通过渐进式混合精度避免长 CoT 中量化误差累积；说明“低价值 KV 降精度”是永久删除之外的可逆替代。
- [63] QuoKA: 从代表性 query 出发选择 key，主要优化 chunked prefill；它明确了 prefill 稀疏化与 decode eviction 的不同目标。
- [64] IceCache: 通过语义 token 聚类与 PagedAttention 组织 GPU–CPU 分层 KV，在有限 GPU token 预算下提高缓存命中率与传输效率。
- [65] FusedKV: 针对跨层 KV 共享的信息损失，用底层与中层表示融合重构高层 KV，在降低长序列缓存占用的同时缩小与逐层缓存的性能差距。

#### 其他论文

- [1] seerattention-r 训练稀疏
- [2] raas 注意力骤降现象
- [3] r-kv Nips25 精读+复现 利用长序列KV冗余和重要token选择做decoding阶段cache压缩 依赖重复假设与超参数
- [5] lazyeviction 长复发现象
- [6] Lessismore 跨头共享token选择+缓存 免训练
- [7] delta 不同层用不同的稀疏注意力 中间共享层
- [8] rlkv 头引导关键token识别
- [9] lil decoding稀疏增加成本
- [10] retroattention decoding误差的逐步修正 新kv进入时候重算
- [11] specattn 稀疏-投机解码协同 用verification阶段知道后续drafting
- [12] nsa 训练时稀疏
- [13] moba 块级MoE式稀疏注意力
- [14] dsa deepseek lightning indexer + MLA集成
- [15] minimax-m1 lightning混合注意力推理模型 + CISPO强化学习
- [16] quest 解码阶段kv选择 最核心基线
- [17] duoattention 头粒度差异化
- [18] ada-kv 逐头自适应预算分配 不同头分配不同的kv预算
- [19] gated-attention SDPA + 门控诱导稀疏 无sink
- [20] twilight top-p层次化剪枝 + 可组合优化器
- [21] retrieval-head 检索头机制解释
- [22] attentionpredictor 注意力时空预测 + 预取
- [23] rocketkv 两阶段KV压缩 snapkv粗淘汰+混合稀疏
- [24] dms 用tts让kv压缩后推理更有效
- [25] headkv 检索+推理双打分头分类 + KV预算分配 + 稳定性分析
- [26] expected-attention 期望注意力闭式估计 无需训练的KV选择
- [27] hold-onto-that-thought 推理KV压缩实证基准
- [28] gta/gla 硬件高效decoding注意力 grouped
- [29] nosa decoding阶段加入局部窗口KV选择与动态选择并行，显式设置预算约束带宽，把sa offload到kvcache
- [30] thought-anchors 互补识别方法 + 句级anchor识别 + receiver heads
- [31] seal 三类thought  + latent空间识别方向向量并转向
- [32] ssa 稀疏注意力误差上界分析 + 稀疏决策指导
- [33] asentmax α-entmax函数稀疏 稀疏长度外推
- [34] crystal-kv answer-first + crystal/slip KV压缩 + 后向贡献评分
- [35] foresightkv 学习long-term contribution + golden eviction SFT + MDP/GRPO + 纠错关键token保护 根据淘汰轨迹后训练打分器 icml26
- [37] g-kv decoding-time免训练 + global attention打分淘汰
- [41] spark 沿key channel方向剪枝，并在decoding时动态恢复
- [45] judgeq 训练soft token query，在prefill阶段优化KV淘汰
- [48] snapkv 用生成前的observation window注意力聚合预测重要KV，在固定预算下保留关键上下文并压缩cache
- [49] streamingllm 保留attention sink和最近窗口token，使LLM能在有限KV cache下稳定处理超长流式输入
- [50] h2o 基于heavy hitter token假设累计注意力分数，只保留持续被关注的KV token来降低cache开销
- [51] shadowkv 用低秩key、value offloading和动态选择重构稀疏KV
- [52] infllm-v2 使用dense-sparse switchable attention结构适配短长序列
- [53] protection-is-nearly-all-you-need 保护prompt边界等关键结构，指出结构保护比淘汰打分更重要
- [54] epikv 精读 用跨层隐藏状态变化做差异淘汰，不依赖注意力矩阵；当前实验说服力有限 有开源代码
- [55] beyond-80/20 在RL后训练阶段只更新高entropy token，并在Qwen3系列模型上验证
- [56] sparda 用Forecast head预测下一层所需KV，实现跨层预取
- [57] lethe 按层和时间两个维度动态分配预算与淘汰KV
- [58] indexcache 跨层复用稀疏index减少计算，并提供training-aware优化
- [59] FlashMemory-DeepSeek-V4 arxiv2606 Lookahead预测 两级检索召回 有开源代码 缺点是长输入检索