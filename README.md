# decodeatt / KVBench

KV eviction policy evaluation framework for long-generation reasoning workloads.

The framework entrypoint is `scripts/eval.py`. It currently reuses the existing
token-level eviction implementation in `kv_eviction.runner_token`. Older RescueKV
research scripts have been removed from the main workflow.

## 环境（uv）

```bash
# 安装 uv（若没有）：curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                      # 按 uv.lock 装好全部依赖到 .venv
source .venv/bin/activate

# 可选：flash-attn（需 CUDA 工具链，单独装）
uv pip install -e ".[flash]" --no-build-isolation
```

主复现模型：deepseek-ai/DeepSeek-R1-Distill-Llama-8B。
Linux + CUDA 下 torch 默认 PyPI wheel 已含 CUDA，无需额外配置。

## 当前结构

```text
configs/eval.yaml                 # 通用 KV 驱逐评测配置
configs/strategies/               # 可选策略组配置，如 all_supported
kvbench/                          # 新框架：数据、模型加载、arm 解析、评测、指标
kv_eviction/                      # 底层 KV 驱逐 runner 与策略实现
kv_eviction/strategies/           # token 驱逐策略，一策略一文件
scripts/eval.py                   # 新主入口
scripts/gen_traces.py             # trace 生成工具
```

## 运行通用 KV 驱逐评测

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --config configs/eval.yaml \
  --dataset math500 \
  --n 1 \
  --arms fullkv,snapkv@1024,h2o@1024,streamingllm@1024,rkv@1024 \
  --out results/rkv_smoke.json
```

当前支持的 token 级驱逐策略：

- `fullkv`: 不触发驱逐的上界 baseline。
- `snapkv@B`: SnapKV-style baseline，按每层/每 KV head 的 observation-window
  pooled attention 保留到预算 `B`；默认 `window_size=32`、`pooling=avgpool`。
- `h2o@B`: 官方 HuggingFace H2O baseline，按最后一步 attention 的 head 平均
  分数保留到预算 `B`，并保留最后 1 个 token。
- `streamingllm@B`: 官方 HuggingFace StreamingLLM baseline，保留 first tokens
  和最近 `B - first_tokens` 个 token。
- `rkv@B`: 论文版 R-KV，`B` 与官方 `R1KV.budget` 一致，是压缩后的总
  cache 长度；每 128 tokens 压缩一次，保留 `B - alpha` 个候选 token 加最后
  `alpha=8` 个 observation tokens。
- `random@B` / `window@B`: diagnostic 策略，不属于官方 baseline 集合。

策略实现位于 `kv_eviction/strategies/`：`full.py`、`snapkv.py`、`h2o.py`、
`streamingllm.py`、`rkv.py`、`random.py`、`window.py`；策略注册表在
`kv_eviction/strategies/token.py`，评测 arm 解析在 `kvbench/policies.py`。

结果 JSON 会为每题每个 arm 记录：

- `elapsed_sec`: 该题该策略耗时。
- `tokens_per_sec`: 生成吞吐。
- `peak_memory_bytes`: CUDA 峰值显存；非 CUDA 为 null。
- `mean_compression_ratio`: 每次驱逐后 cache 长度 / 驱逐前 cache 长度的均值。
- `evict_events`: 每次驱逐的 step、驱逐前后 cache 长度、压缩比、驱逐 token 数。
- `mean_effective_cache_len` / `max_effective_cache_len`: 按每层/每 KV head
  统计的有效 KV 长度；当前矩形 cache 策略下等于物理 cache 长度。
- `total_evicted_tokens` / `mean_evicted_per_event` / `cache_len_curve`: 物理
  cache 压缩强度和长度曲线。
- `head_budget_mean/std/min/max/entropy` / `num_underfilled_heads`: head-wise
  保留长度分布；当前固定长度策略下主要用于 sanity check。
- `prefill_sec` / `decode_sec` / `decode_forward_sec` /
  `attention_observation_sec` / `eviction_sec_total`: 时间开销拆分。

策略超参可在 `configs/eval.yaml` 里配置，例如：

```yaml
policy_params:
  snapkv:
    window_size: 32
    kernel_size: 7
    pooling: avgpool
  rkv:
    lambda: 0.1
    alpha: 8
    retain_ratio: 0.1
    retain_direction: last
    similarity_threshold: 0.5
    pool_kernel: 7
```

也可用 CLI 覆盖：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --policy-param rkv.lambda=0.2 \
  --policy-param rkv.alpha=8
```

策略组也可以拆到独立 config 后由主评测脚本选择：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --config configs/eval.yaml \
  --strategy-config configs/strategies/all_supported.yaml \
  --out results/rkv_smoke.json
```

## R-KV parity test

```bash
uv run python tests/test_rkv_parity.py
uv run python tests/test_official_baseline_parity.py
```

这些测试用小张量 oracle 对齐上游 HuggingFace `update_kv` 语义，覆盖
SnapKV、H2O、StreamingLLM、R-KV 的保留 indices、head-wise cache gather、
candidate attention 重归一化、完整 cache similarity、官方 total-budget
保留长度和 debug score 对齐。

论文口径 smoke reproduction：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --config configs/eval.yaml \
  --dataset math500 \
  --n 1 \
  --arms fullkv,snapkv@1024,h2o@1024,streamingllm@1024,rkv@1024 \
  --max-new 16384 \
  --out results/rkv_smoke.json
```
