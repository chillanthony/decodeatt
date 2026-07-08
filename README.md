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

主模型：deepseek-ai/DeepSeek-R1-Distill-Qwen-7B。
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
  --model /path/to/model \
  --dataset math500 \
  --n 24 \
  --arms full,random@1024,h2o@1024,window@1024,snapkv@1024,rkv@1024 \
  --out results/eval.json
```

当前支持的 token 级驱逐策略：

- `full`: 不触发驱逐的上界 baseline。
- `random@B`: 保留 sink/recent 后，从中段随机保留到预算 `B`。
- `h2o@B`: 按累计注意力保留到预算 `B`。
- `window@B`: 按观察窗注意力保留到预算 `B`。
- `snapkv@B`: 按观察窗 max-pooled 重要性保留到预算 `B`。
- `rkv@B`: 在重要性基础上加入 key 冗余惩罚，保留到预算 `B`。

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
    redundancy_lambda: 0.5
    pool_factor: 3
    pool_extra: 128
```

也可用 CLI 覆盖：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --policy-param rkv.redundancy_lambda=0.7 \
  --policy-param rkv.pool_factor=4
```

快速烟雾测试可用更小模型和内置样例：

```bash
PYTHONPATH=. uv run python scripts/eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset sample \
  --n 1 \
  --arms full,random@128,h2o@128,window@128,snapkv@128,rkv@128 \
  --max-new 64 \
  --greedy \
  --out results/smoke_eval.json
```
