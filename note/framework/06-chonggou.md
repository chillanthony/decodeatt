# KV 管理代码重构

## 目标

当前代码首先服务于 HuggingFace reference runner，后续还需要在 vLLM、PyTorch
SDPA/FlashAttention 类执行栈上验证新的 KV 管理算法。重构目标不是马上实现 vLLM
后端，而是先固定三个边界：

1. **策略层（Policy）**：只根据 observation、cache metadata 和预算返回保留位置或
   head-wise 选择，不直接调用模型，也不修改 cache。
2. **缓存层（Cache Manager）**：负责 cache 的视图、compact、分页/block 分配和
   有效长度统计；策略不依赖 tuple、`DynamicCache` 或 vLLM block 的具体表示。
3. **执行层（Execution Backend）**：负责 prefill/decode forward、attention 输出
   和模型后端差异；评估器只提交 workload 和策略配置。

评估器继续负责数据集、候选数、并发/分桶、结果落盘和吞吐指标，不把 serving
scheduler 逻辑塞进策略实现。

## 已完成的第一阶段

### 执行后端接口

新增 `kv_eviction/backends/`：

- `ExecutionBackend`：定义 `prefill()` 和 `decode()` 语义边界。
- `TransformersExecutionBackend`：包装当前 callable HuggingFace causal LM，保持
  现有 `past_key_values`、`attention_mask`、`position_ids` 参数约定。

`generate_token_evict()` 和 `generate_token_evict_batch()` 新增可选的
`execution_backend` 参数；不传时自动使用 HF adapter，所以现有 CLI 和结果格式不变。
`kvbench.evaluator.run_generation_eval()` 也暴露该参数，未来可以从 evaluator 注入
后端，而不修改策略代码。

每个结构化 arm 现在可以写 `engine: hf` 或 `engine: vllm`，默认值为 `hf`。旧的
字符串写法（例如 `rkv@1024`）自动使用 HF reference runner。当前 vLLM evaluator
只允许 `fullkv`，因为它把所有 prompt 交给 vLLM 的 serving scheduler；R-KV、SnapKV
等需要直接拥有 KV 的策略如果写成 `engine: vllm` 会立即报错，不会静默回退或伪装成
已经接入了 KV eviction。

### Cache manager 接口

新增 `kv_eviction/cache_manager.py`：

- `CacheManager`：定义单请求和 batch compact 操作。
- `TransformersCacheManager`：支持 tuple cache、HF `DynamicCache` 风格的
  `layers`、`key_cache/value_cache`，保留 global 和 head-wise selection。

生成循环的实际淘汰已经通过 manager 执行。`runner_token.py` 中旧的私有 compact
函数暂时保留，原因是诊断脚本和 parity 测试仍直接使用它们；下一阶段应把这些兼容
函数改成 manager 的薄包装，避免两份实现继续漂移。

### 策略层现状

`kv_eviction/strategies/base.py` 的 `SelectionContext` 和
`TokenEvictionPolicy` 已经是策略边界，现有 R-KV、SnapKV、H2O 等实现可以继续
复用。需要注意：`params` 中目前仍允许注入 `_model`、`_layers` 等 reference-runner
对象，这部分会阻碍后端移植，应逐步改成显式的 `PolicyRuntimeContext` 或 cache
metadata。

### 回归约束

新增 `tests/test_cache_manager.py`，覆盖 tuple cache 的单请求和 batch compact。
现有 `test_batch_generation.py`、策略 parity 测试和 evaluator 结果格式测试仍是
重构后的兼容性约束。

## 第二阶段：清理 reference runner

按以下顺序继续，避免一次性重写生成循环：

1. 将 `runner_token.py` 中的 `_cache_view`、`_compact_cache_update` 和
   `_compact_cache_update_batch` 改为 `TransformersCacheManager` 的兼容薄包装，删除
   重复的 tensor gather 实现。
2. 把 `_select_tokens()` 中的锚点保护、debug 结构和 policy selection 拆成
   `EvictionController`。controller 只接收 `SelectionContext`、观测结果和 manager
   metadata，返回 `EvictionDecision`（保留索引、救援索引、调试分数）。
3. 将 attention observation 从生成循环拆为 `ObservationProvider`；R-KV 的
   q/k attention-logit 观测和 SnapKV 的 attention window 观测分别实现 provider。
4. 将 timing/debug 统计改成可选 sink。性能基准默认只记录 wall-clock、prefill、
   decode、eviction 和 aggregate throughput，避免 debug tensor/token decode 干扰结果。
5. 为单请求、静态 batch、提前结束请求补齐相同的 `BackendResult` 和指标字段，确保
   `batch_tokens_per_sec` 的口径一致。

## 第三阶段：vLLM / fused attention 接入

### 先做的最小适配

当前已经实现一个只支持 FullKV 的 serving evaluator（见 `kvbench/vllm_evaluator.py`），
用同一组 prompt/output workload 记录 aggregate output tok/s，作为后续 TTFT、TPOT、
p95 latency 和峰值显存对齐的 baseline。确认 evaluator 协议和服务端统计口径一致后，
再接入淘汰策略。

### KV 管理适配

vLLM 使用 paged/block KV，不应在 Python 层对每个请求调用 dense
`index_select`。新的 manager 需要提供：

- block ownership 和有效 token 数；
- block-level 保留/释放或 token-to-block 映射；
- batch 中不同请求的独立预算；
- 请求结束后的 block 回收和新请求补入；
- cache compaction 事件的异步或 CUDA 实现。

如果算法必须保留任意 token，不能只把连续 token 复制进一个新 tensor，否则 gather
成本会抵消 attention 节省。优先研究 block-aware selection；无法 block 对齐时，先
做 dense reference 结果对齐，再单独评估 kernel 代价。

### FlashAttention / SDPA 注意事项

FlashAttention 只解决 attention 计算，不自动解决动态 KV 淘汰。要获得端到端收益，
需要同时保证：

- selector 输出能被 kernel 直接消费；
- cache layout 与 block/page 访问模式一致；
- observation 不在每个 decode step 打开完整 attention；
- compact、metadata 更新和调度不会阻塞主 decode stream。

## 当前仍需优化的部分

1. **重复实现**：旧 compact helper 和新 manager 目前并存，下一步必须收敛到单一实现。
2. **模型依赖泄漏**：策略参数里的 `_model`、`_layers` 需要替换成显式 metadata 或
   provider，才能真正支持 vLLM。
3. **执行接口粒度**：当前 backend 仍是逐步 HF-style forward，不能表达 continuous
   batching；后续需要加入 request admission、active batch 和异步完成事件。
4. **cache 语义**：目前以 dense slot index 为主，尚未定义 block、page、logical
   position、physical position 的统一协议。
5. **观测成本**：R-KV 的 observation 和 redundancy 计算仍在 Python/reference
   路径，必须先做 CUDA/torch.compile 或 kernel prototype 才能判断真实收益。
6. **指标口径**：静态 batch、continuous batching、提前 EOS 和不同输出长度需要统一
   wall-clock、有效输出 token、request/s、TTFT、TPOT、p50/p95/p99 的定义。
7. **正确性矩阵**：每个新算法要同时通过单请求、异构 prompt batch、不同 EOS 时间、
   global/head-wise selection，以及 FullKV teacher-forcing 对齐。

## 暂不做的事情

- 不在这一阶段直接修改 vLLM 内核或 fork vLLM。
- 不把 FlashAttention 当成 KV manager；它只是执行 backend 的一个可能实现。
- 不删除旧 runner，直到新 manager 与 parity 测试覆盖完整。
