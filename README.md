# decodeatt / KVBench

KV eviction policy evaluation framework for long-generation reasoning workloads.

The framework entrypoint is `scripts/eval.py`. It currently reuses the existing
token-level eviction implementation in `rkv.runner_token`. Older RescueKV
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
kvbench/                          # 新框架：数据、模型加载、arm 解析、评测、指标
rkv/runner.py                     # 旧兼容模型加载工具，gen_traces.py 在用
rkv/runner_token.py               # 现有 token 级 KV 驱逐底层实现
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
- `rkv@B`: 论文版 R-KV，`B` 是 paper 的 `Bbudget`；每 128 tokens 压缩一次，保留
  `Bbudget` 个候选 token 加最后 `alpha=8` 个 observation tokens。

策略选择接口在 `rkv/policies.py`，评测 arm 解析在 `kvbench/policies.py`。

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
    beta: 8
    similarity_threshold: 0.9
    pool_kernel: 5
```

也可用 CLI 覆盖：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --policy-param rkv.lambda=0.2 \
  --policy-param rkv.alpha=8
```

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
