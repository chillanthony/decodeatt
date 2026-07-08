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
configs/strategies/               # 可选策略组配置，如 rkv_paper/all_supported
kvbench/                          # 新框架：数据、模型加载、arm 解析、评测、指标
kv_eviction/                      # 底层 KV 驱逐 runner 与策略实现
kv_eviction/strategies/           # token 驱逐策略，一策略一文件
rkv/                              # 兼容旧 import 的转发包
scripts/eval.py                   # 新主入口
scripts/gen_traces.py             # trace 生成工具
```

## 运行通用 KV 驱逐评测

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --config configs/eval.yaml \
  --dataset math500 \
  --n 1 \
  --arms full,snapkv@1024,rkv@1024 \
  --out results/rkv_paper_smoke.json
```

当前支持的 token 级驱逐策略：

- `full`: 不触发驱逐的上界 baseline。
- `random@B`: 保留 sink/recent 后，从中段随机保留到预算 `B`。
- `h2o@B`: 按累计注意力保留到预算 `B`。
- `window@B`: 按观察窗注意力保留到预算 `B`。
- `snapkv@B`: 按观察窗 max-pooled 重要性保留到预算 `B`。
- `rkv@B`: 论文版 R-KV，`B` 与官方 `R1KV.budget` 一致，是压缩后的总
  cache 长度；每 128 tokens 压缩一次，保留 `B - alpha` 个候选 token 加最后
  `alpha=8` 个 observation tokens。

策略实现位于 `kv_eviction/strategies/`：`random.py`、`h2o.py`、`window.py`、
`snapkv.py`、`rkv.py`、`rkv_paper.py`；策略注册表在
`kv_eviction/strategies/token.py`，评测 arm 解析在 `kvbench/policies.py`。

结果 JSON 会为每题每个 arm 记录：

- `elapsed_sec`: 该题该策略耗时。
- `tokens_per_sec`: 生成吞吐。
- `peak_memory_bytes`: CUDA 峰值显存；非 CUDA 为 null。
- `mean_compression_ratio`: 每次驱逐后 cache 长度 / 驱逐前 cache 长度的均值。
- `evict_events`: 每次驱逐的 step、驱逐前后 cache 长度、压缩比、驱逐 token 数。

策略超参可在 `configs/eval.yaml` 里配置，例如：

```yaml
policy_params:
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
  --strategy-config configs/strategies/rkv_paper.yaml \
  --out results/rkv_paper_smoke.json
```

## R-KV reference parity test

```bash
uv run python tests/test_rkv_parity.py
```

该测试用小张量 oracle 对齐上游 R-KV 的 `R1KV.update_kv` 语义，覆盖
candidate attention 重归一化、完整 cache similarity、官方 total-budget
保留长度和 debug score 对齐。

论文口径 smoke reproduction：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --config configs/eval.yaml \
  --dataset math500 \
  --n 1 \
  --arms full,snapkv@1024,rkv@1024 \
  --max-new 16384 \
  --out results/rkv_paper_smoke.json
```
