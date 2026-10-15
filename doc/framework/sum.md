# framework 文档简要总结

## 00-数据集

按应用场景整理常用评测集：

- 长上下文：LongBench / LongBench v2 用于长文档理解、跨段推理和 QA；RULER 用于可控长度的上下文压力测试。
- 长会话与长期记忆：LoCoMo 评估跨会话记忆、事件总结和时间推理；LongMemEval 覆盖信息抽取、知识更新、长期推理和拒答。
- Agent 长程任务：SWE-bench 评估软件工程代码修改与测试；WebArena 评估真实网站中的多步浏览、检索和操作。

## 01-评估协议

整理 R-KV 的 decoding-time 长 CoT KV 压缩复现协议。核心是每生成约 128 个 token 触发一次压缩，用最后 8 个 observation tokens 对候选 KV 打分，并结合 attention importance 与 cosine redundancy 选择保留项。主要对比 FullKV、SnapKV，模型为 DeepSeek-R1 Distill Llama-8B/Qwen-14B，数据集为 MATH-500 和 AIME 2024。

协议要求区分 smoke/debug 与 paper-style 评测：后者使用 sampling、`temperature=0.6`、`top_p=0.95`、每题 64 个候选，并启用 chat template；同时报告绝对 KV budget、cache ratio、最终和平均 cache 长度，避免只看压缩比例造成误读。文档也记录了官方实现路径和运行示例。

## 02-脚本

说明 `scripts/cluster/run_arm.sh` 的参数含义、实验命名和运行注意事项。`RUN_NAME` 必须唯一；命令行参数会覆盖 YAML；正式实验前先用少量题目和短生成做 smoke test；离线运行前确认模型与数据集缓存。

实验定义参数包括模型、配置、数据集、策略/budget、最大生成长度、候选数及采样和淘汰超参；性能参数包括 batch、题目 batch、GPU 数量、可见 GPU、通信端口和超时。正式结果应记录 commit、配置、seed、模型和 GPU 信息。

## 03-加速

总结 KV eviction 框架的性能基线、瓶颈和优化路线。当前主要瓶颈是逐 token Python 调度、eager attention、observation 计算、手动 cache compact 及细粒度统计，因此压缩本身暂未带来稳定吞吐收益。

已完成的优化包括减少 observation/debug 开销、attention/cache update 内嵌原型、按策略启用 SDPA/FlashAttention、同题多候选 batch 和跨题长度分桶 batch，并有对应测试验证。后续方向是 metrics 分级、StaticCache/`torch.compile`，以及长期迁移到 vLLM/SGLang paged KV cache。当前限制是无 CUDA 环境，尚未完成真实 GPU batch 吞吐、显存和 OOM 扫描。

## 04-evaluationsum

给出评测汇总的基本框架：主模型为 R1-Llama-8B，数据集为 AIME 和 Math500；策略覆盖 FullKV、Window、Random、SnapKV、StreamingLLM、H2O 和 R-KV；核心指标是精度、压缩率和吞吐量；协议维度包括超参数、最大生成长度、budget 扫描、NLL/准确率和采样消融。

## 05-升级

记录升级 `torch` / `transformers` 前后的测试基线与兼容性结论。升级后环境为 torch 2.11.0（cu128）、transformers 5.17.0、numpy 2.2.6；源码无需修改，5 个测试脚本及 13 个 TideProbe 函数共 49 个用例全部通过，tiny Llama 的 NLL 数值逐位重现。

文档固定了 R-KV、各 baseline、batch generation、日志格式和真实前向的正确性契约，并列出升级风险。当前唯一需要后续关注的是 transformers 5.18 将移除 `crop` 的正值调用；同时应继续保留自实现的 cache 适配层。
