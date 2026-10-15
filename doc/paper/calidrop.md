# CaliDrop 精读笔记

## 基本信息
- 论文: CaliDrop: KV Cache Compression with Calibration, arXiv:2507.19906v2, 2025-08-04。
- 主题: 在 token eviction 型 KV cache compression 上加入 calibration,缓解高压缩率下的 accuracy degradation。
- 定位: 不是新的 token importance scorer,而是可叠加到 StreamingLLM / H2O / SnapKV 上的补偿层。

## 1. 研究背景与动机
- LLM decoding 依赖 KV cache 避免重复计算历史 K/V,但 KV 显存随 context length、batch size、model size 线性增长。
- Token eviction 通过只保留少数重要 token 降低显存,典型方法包括 StreamingLLM、H2O、SnapKV。
- 现有方法的局限在于:一旦 token 被 eviction,后续即使重新变重要也无法被当前 attention 使用。
- 高压缩率下这种"一次丢弃,永久不可见"会严重伤害 long-context QA、multi-hop reasoning、needle retrieval 等任务。
- 本文出发点是两个经验观察:相邻位置 query 的 cosine similarity 较高;历史 query 在被淘汰 KV 上的 attention output 可近似未来 output。
- 核心 insight:不要把 evicted KV 彻底删除,而是 offload 并保存一个历史校准结果;当未来 query 与历史 query 足够相似时,复用该结果补偿 compressed attention。
- 换句话说,CaliDrop 把 token eviction 的不可逆信息损失改造成一种低频、投机式、可更新的 attention output compensation。

## 2. 方法详解
- 理论基础是 attention decomposition:若完整 KV 集合 S 被拆为保留集合 Si 和淘汰集合 Sj,则完整 attention 可写成两个子 attention 的加权和。
- 公式直觉: `Att(Q,K,V)=alpha_i*Att(Q,K_i,V_i)+alpha_j*Att(Q,K_j,V_j)`,其中 alpha 是对应子集合 softmax exponential sum 在总 denominator 中的占比。
- 因此,只要能近似 evicted KV 上的 `Att(Q,K_j,V_j)`,就能把只看 compressed KV 的输出修正得更接近 FullKV。
- Prefill 阶段:先用已有 eviction strategy 得到 `KVcompress`,再把 `KVevict = KVfull - KVcompress` offload。
- Prefill 还会用最后一个 query `Q_-1` 对 `KVevict` 做 attention,保存 `Cq`(历史 query)、`Cout`(evicted KV attention output)、`Cweight`(evicted KV denominator)。
- Decode 阶段:当前 query `Qt` 先对 `KVcompress` 做正常 attention,得到 compressed output 和 `Aweight`。
- 然后计算 `rho = cos(Qt, Cq)` 判断当前 query 能否复用历史 calibration。
- 若 `rho > theta2`: 当前 query 与历史 query 足够相似,直接把 `Cout` 按 decomposition 权重加回 output。
- 若 `rho < theta1`: 历史 query 太旧,重新加载 `KVevict`,用当前 query 重算 `Cout/Cweight/Cq`,再进行 calibration。
- 若 `theta1 <= rho <= theta2`: 相似度不够高但也不值得重算,跳过 calibration,直接返回 compressed output。
- 主实验设 `theta1=0.7, theta2=0.85`;`theta1` 控制 recomputation frequency,`theta2` 控制 calibration acceptance。
- 与 SnapKV/H2O/SLM 的差异:这些方法负责选择哪些 token 留在主 KV cache,CaliDrop 负责补偿被它们丢掉的 token contribution。
- 与 KV quantization / merging 的差异:量化降低每个 KV 的 bit,merging 改写 token 表示,CaliDrop 主要在 output 层用 offloaded KV 的历史 attention output 做校准。

## 3. 实验设计评估
- 模型: Mistral-7B-Instruct、LLaMA-3-8B-Instruct、LLaMA-3-70B-Instruct。
- Benchmarks: LongBench、RULER、Needle-in-a-Haystack,覆盖 QA、summarization、few-shot、synthetic retrieval、code 与长上下文检索。
- Baselines: FullKV、StreamingLLM(SLM)、H2O、SnapKV。
- KV budget: 主要测试 64、128、256、512;高压缩率下重点看 64/128。
- 实验设置:沿用 SnapKV 风格,只在 prefilling 阶段做 compression;decoding 阶段叠加 CaliDrop calibration。
- SLM 设置 attention sink=32;H2O 将 budget 均分给 important tokens 与 local tokens;SnapKV observation window=32、average pooling kernel=5。
- LongBench 结果:在 KV=64 时收益最明显。Mistral-7B 上 SnapKV avg 33.79→37.90,H2O 33.81→37.81,SLM 28.64→33.62。
- LLaMA-3-8B, KV=64: SnapKV 40.50→43.61,H2O 40.74→44.13,SLM 36.12→40.52。
- LLaMA-3-70B, KV=64: SnapKV 47.45→49.66,H2O 47.53→49.99,SLM 42.64→46.24。
- RULER 结果: LLaMA-3-8B 上 SnapKV 在 KV=64 时 14.95→23.32,KV=128 时 32.19→42.24,KV=512 时 60.64→67.60。
- Needle-in-a-Haystack 结果: LLaMA-3-8B,8k context,KV=64 时 SLM/H2O/SnapKV 分别从 26.6/39.7/61.0 提升到 45.6/67.7/66.6。
- Needle 8k,KV=128: H2O 48.5→83.8,SnapKV 87.2→94.5;KV=256: H2O 50.0→84.3,SnapKV 95.2→98.4。
- 主要结论: CaliDrop 对多种 eviction baseline 都有正增益,尤其在 retrieval-heavy benchmark 和 high compression ratio 下更明显。
- 边界现象: KV size 越大,收益越小;因为 compressed KV 已接近 FullKV,同时 evicted 部分权重 `alpha_j` 下降。
- 阈值消融:固定 `theta2` 时,提高 `theta1` 通常提升 accuracy,因为重算更频繁、calibration 更新更及时。
- 固定 `theta1` 时,`theta2` 不呈单调趋势;过高会漏掉有效 calibration,过低会引入有害 calibration。
- 效率实验:input length=1024,output length=128,KV budget=128,A800 80GB;CaliDrop 相比 SnapKV 只有小幅额外开销,吞吐接近 SnapKV 且明显高于 FullKV。
- 作者报告 `theta1=0.7` 时平均约每 8 个 decoding step recompute 一次,并提出 calibration size 来减少 offloaded tokens 数量。
- 实验公平性评价:baseline 覆盖主流 eviction 方法,模型与 benchmark 较丰富;但系统开销评估偏轻,缺少真实服务端 I/O、batch heterogeneity、长生成持续 eviction 的端到端评测。

## 4. 局限性与未来方向
- 作者明确局限: CaliDrop 的 accuracy gain 以额外计算为代价,介于 KV offloading 与 KV compression 之间,不是纯免费增强。
- 作者明确局限: 当 KV budget 较高、baseline 已接近 FullKV 时,CaliDrop 的上限被 FullKV 限制,边际收益会下降甚至可能变负。
- 作者明确局限: 论文主要研究 prefilling-only compression,虽然声称可扩展到 decoding-phase eviction,但长生成场景仍需要更细设计。
- 【⚠️ 存疑】offloaded KV 仍需存储和访问,真实 serving 中 CPU/GPU 传输、page-level I/O、batch 不齐可能放大尾延迟。
- 【⚠️ 存疑】query locality 只通过部分层/头/样本展示,缺少跨模型、跨任务、跨层头的系统统计。
- 【⚠️ 存疑】topic shift、needle 附近突发检索、推理回溯等场景可能让 historical `Cout` 与当前 query 的真实需求不匹配。
- 未来方向包括:对 offloaded KV 做 quantization/token eviction;设计 CUDA kernel、page-level I/O、communication overlap;按 layer/head 自适应 `theta1/theta2`;联合优化 calibration size 与 token importance。

## 5. 评价
- CaliDrop 有效的核心原因是:它没有假设 eviction scorer 永远正确,而是为 evicted tokens 建立一个低频补偿通道。
- Attention decomposition 让 calibration 有明确数学解释,不是简单 residual trick;`Cout` 的权重来自 softmax denominator 占比。
- Query locality 是它成立的关键工程假设:相邻 decoding step 的 query 相似,使历史 evicted-output 在短窗口内可复用。
- 两阈值设计比较实用:高相似直接复用,低相似触发重算,中间区域跳过,避免无脑 calibration。
- 对 RescueKV 的启发: CaliDrop 是"计算侧救回",RescueKV 更像"存储侧豁免";二者都针对 token eviction 的不可逆损失。
- RescueKV 可借鉴其三分支决策:低风险直接用 compressed KV,高置信补偿,高风险重新救回/重算。
- 论文最大贡献不是提出更强 scorer,而是把 KV compression 从"删或不删"扩展为"删后仍可被校准补偿"。
- 总体判断:思路清楚、实验覆盖充分,在 high compression ratio 和 retrieval-heavy tasks 上收益明显;但系统实现和真实 serving 成本仍是主要短板。
