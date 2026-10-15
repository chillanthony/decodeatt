## Idea

### 注意力模式与关键 Token 识别

- attention模式为线索的稀疏mask decodeatt analysis-1
- 探寻reasoning注意力集中度 reasoning sink, sink感知的mask, sink decodeatt analysis-2
- 推理过程关键token定义与识别
- query-agnostic eviction的性质（训练一个对查询分布有感知的救援门在decoding时期动态分配+prefilling时两级预算+创新点落在鲁棒性）
- 长CoT中重复推理检测（可以和最新blog配合）
- 根据不用重建完整attention矩阵的便宜信号（表征变化、头评分、蒸馏评分器）

### 动态策略与预算控制

- 用rl优化稀疏mask decodeatt analysis-3

### 未来价值预测与选择器优化

- 减少计算的未来感知淘汰机制
- 可以跨样本迁移的可学习淘汰机制
- 把indexer打分器加入新指标
- 长CoT中对长期记忆的预测式prefetching

### 安全边界、保护与纠错

- 稀疏化的安全边界 decodeatt analysis-5
- 训练稀疏的token保护机制
- 轻量可训练保护模块/head
- 通过错误轨迹对比提升KVCache性能 
- 驱逐时的外部verifier

### 其他思路

- 做indexer cache
- 找几个数据集的failure case做一个定点修复
- kv eviction x looped transformer
- 分型方式 多任务 长期记忆 稀疏训练kv优化