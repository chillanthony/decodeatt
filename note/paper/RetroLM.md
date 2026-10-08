# RetroLM 精读笔记

## 1. 研究背景与动机
- 论文：RetroLM: Retrieval-Augmented KVs for Long-Context Processing。
- 问题：长上下文 LLM 的主要瓶颈来自长输入 prefill 和 decode 中对大量 KV 的存储与注意力计算。
- 传统 RAG 能减少输入 token，但存在检索不准、片段上下文断裂、重复 prefill 计算等问题。
- KV sparsification 方法多用启发式 attention 选择，缺少可训练的检索模块。
- 核心动机：把 retrieval 从 raw text/chunk 层移动到 KV cache 层，直接检索对当前计算最有用的 KV pages。

## 2. 方法详解
- RetroLM 将输入切成连续 pages，默认 page size 为 128，并在每个 page 末尾加入 bookmark token。
- bookmark token 的 hidden state/KV 表示该 page 的索引，用于后续 KV page retrieval。
- prefill 阶段使用 streaming encoding：当前 page 只检索少量历史 pages 参与每层 attention，其余已编码 KV 可 offload 到 CPU。
- decode 阶段给定 user query，只做一次 page retrieval，选出最关键 KV pages 参与后续生成。
- Page Retriever 是可训练的 plug-in 模块，复用 LLM 结构但对 bookmark token 使用额外投影矩阵。
- 检索分数是当前 bookmark query 与历史 bookmark key 的 dot product，相当于 dense retrieval 在 KV page 上的版本。
- Stage-1：用 MS MARCO 50K pairwise data 与 SlimPajama 5K 合成数据做 contrastive learning，冻结 backbone，只训练 page retriever。
- Stage-2：用 SlimPajama 无监督文本做 post-training，让模型适应 retrieved sparse KV pages，而不是 full attention。

## 3. 实验设计评估
- benchmark：LongBench、InfiniteBench、RULER。
- backbone：Mistral-7B-Instruct-v0.2 与 Llama-3-8B-Instruct。
- baseline：full attention 原模型、LM-Infinite、StreamingLLM、InfLLM、H2O、SnapKV、PyramidKV，以及 BM25/Contriever/BGE RAG。
- LongBench：在 2K KV budget 下，RetroLM-Stage1 已超过 full attention 和启发式 KV 方法；Stage2 进一步提升。
- 关键结果：Mistral 上 Stage1 比 full attention 高约 2.5 分，Stage2 平均到 43.5；Llama 上 Stage2 平均到 44.4。
- RAG 对比：RetroLM 在 LongBench QA 上明显超过 BM25、Contriever、BGE，说明 KV-level retrieval 比 raw chunk retrieval 更稳。
- InfiniteBench：在平均 145K token 输入上，用 6K KV budget 仍优于 full-attention baseline 和其他 efficient methods。
- RULER/NIAH：RetroLM 在 64K 下比原模型更稳，且 GPU memory 增长更慢，例如 64K 时 25.5GB vs full attention 43.3GB。

## 4. 局限性与未来方向
- 需要训练 page retriever，并可选做 Stage-2 post-training；训练成本和数据构造比 training-free eviction 更高。
- page size、top-k pages、bookmark 设计都会影响效果，部署时需要调参。
- 需要修改模型 attention 和引入 bookmark token，工程侵入性高于简单 KV eviction。
- 检索粒度是 page，不是 token 或 channel；若关键证据非常稀疏，page retrieval 可能带入冗余 KV。
- 【⚠️ 存疑】论文标题强调 RAG 是否真的长上下文表现差，但核心方法已超出普通 RAG，公平比较依赖相同预算和实现细节。

## 5. 评价
- 贡献点：把 RAG 的“检索”内化到 KV cache/page 层，缓解 raw-text RAG 的片段断裂和重复计算。
- 方法优势：可训练 page retriever 比启发式 attention pruning 更有语义检索能力，尤其适合多跳 QA 和超长上下文。
- 对 KV 压缩研究的启发：KV 不只是缓存，也可以组织成可检索的外部记忆；token eviction 可扩展为 page-level memory retrieval。
- 代价：训练、模型改造和 KV page 管理复杂，未必适合作为轻量 drop-in baseline。
- 总体判断：RetroLM 是 KV-level retrieval/RAG 方向的强相关工作，和纯 eviction 方法不同，更像“检索式 KV 选择框架”。
