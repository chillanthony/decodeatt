# decodeatt / KVBench

KV eviction policy evaluation framework for long-generation reasoning workloads.

The framework entrypoint is `scripts/eval.py`. It currently reuses the existing
token-level eviction implementation in `kv_eviction.runner_token`. Older RescueKV
research scripts have been removed from the main workflow.

## 环境（conda + pip）

用 conda 只负责提供隔离的 Python 解释器，**所有包都走 pip 装** ——
vLLM、transformers 5.x 在 conda 上没有可用包，混用两套包管理器还会互相覆盖。

```bash
conda create -n decodeatt python=3.11 -y
conda activate decodeatt

# 核心依赖
pip install -r requirements.txt
```

CUDA 机器上 torch 需要 cu128 构建，**先装 torch 再装其余**：

```bash
pip install torch==2.11.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
pip install -r requirements.txt      # 已装好的 torch 会被跳过
```

主复现模型：deepseek-ai/DeepSeek-R1-Distill-Llama-8B。

### 可选依赖

```bash
# flash-attn：需 CUDA 工具链，编译安装，且必须在 torch 之后
pip install flash-attn>=2.6 --no-build-isolation

# vLLM：吞吐测速专用，必须与 torch 版本严格对齐（0.25.1 要求 torch==2.11.0）
pip install vllm==0.25.1
```

vLLM 走 cudagraph，与主库的 eager-attention 评测路径互斥（取不到逐步 attention），
只在 `efficiency/` 下的 R-KV 测速里用。

### 脚本里的 Python

`scripts/` 下的运行脚本都读 `PYTHON_BIN`，默认指向 `$HOME/.venvs/decodeatt`：

```bash
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
```

在提交脚本（SLURM 等）里这样传即可，不需要改脚本本身。

### 导入路径

仓库不再以可安装包形式装进环境，`kvbench` / `kv_eviction` 靠工作目录在
`sys.path` 上被导入。从仓库根目录运行脚本即可；若从别处调用，需显式指定：

```bash
export PYTHONPATH="/path/to/decodeatt"
```

运行测试：

```bash
PYTHONPATH="$PWD" python tests/test_rkv_parity.py
```

## 当前结构

```text
configs/experiments/              # 完整实验配置；正式运行优先用这里
configs/onestrategy/              # 单策略完整配置，用于补跑或调试
kvbench/                          # 新框架：数据、模型加载、arm 解析、评测、指标
kv_eviction/                      # 底层 KV 驱逐 runner 与策略实现
kv_eviction/strategies/           # token 驱逐策略，一策略一文件
scripts/eval.py                   # 新主入口
scripts/cluster/                  # 通用多卡运行与策略参数入口
scripts/experiments/              # AIME24、MATH-500 实验网格与预设
scripts/tideprobe/                # TideProbe 多卡诊断流水线
scripts/bootstrap/                # 模型和数据集下载工具
efficiency/scripts/               # R-KV vLLM 构建与性能测评
kvbench/diagnostics/gen_traces.py # trace 生成工具
```

## 运行通用 KV 驱逐评测

```bash
PYTHONPATH=. python scripts/eval.py \
  --config configs/experiments/math500_official_b1024.yaml
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

### 简要结果模式

正式跑大量候选时，可启用 `brief` 模式，让最终结果文件只保存各策略的汇总
指标，不保存逐题生成文本、候选列表、cache 曲线或逐次 eviction 事件。文件大小
基本只随策略数量增长，通常为几 KB 到几十 KB：

```bash
PYTHONPATH=. python scripts/eval.py \
  --config configs/experiments/math500_official_b1024.yaml \
  --log-mode brief
```

也可以写进 YAML：

```yaml
experiment:
  out: results/math500_official_b1024.json
  log_mode: brief
```

默认值为 `full`，保持原有完整结果格式。`brief` 模式仍会计算同样的 accuracy、
pass@1、速度、显存、压缩率和 eviction 汇总指标。

策略超参可在完整实验配置的 `policy_defaults` 或单个 arm 的 `params` 里配置，例如：

```yaml
policy_defaults:
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

arms:
  - id: rkv_b1024_lam02
    policy: rkv
    budget: 1024
    params:
      lambda: 0.2
```

也可用 CLI 覆盖：

```bash
PYTHONPATH=. python scripts/eval.py \
  --policy-param rkv.lambda=0.2 \
  --policy-param rkv.alpha=8
```

单策略补跑可以直接使用 `configs/onestrategy/` 里的完整配置：

```bash
PYTHONPATH=. python scripts/eval.py \
  --config configs/onestrategy/rkv.yaml
```

## 同题多候选 batch

正式 pass@1 评测可以在每张 GPU 上并行生成同一道题的多个独立候选：

```bash
PYTHONPATH=. python scripts/eval.py \
  --config configs/experiments/math500_official_b1024.yaml \
  --batch-size 16 \
  --num-return-sequences 64
```

`batch_size` 是单次 GPU micro-batch，`num_return_sequences` 是每题总候选数；
候选随机种子依次为 `seed + seed_offset + candidate_idx`。最后不足一个完整
micro-batch 的候选会自动使用较小 batch。`batch_size=1`、
`num_return_sequences=1` 保持原有行为。

## 跨题 batch

每道题只采样一次时，可以把不同题目按 prompt token 长度排序后共同生成：

```bash
PYTHONPATH=. python scripts/eval.py \
  --config configs/experiments/math500_official_b1024.yaml \
  --problem-batch-size 8 \
  --prompt-bucket-size 32
```

`problem_batch_size` 是一次并行的题目数；`prompt_bucket_size` 是每次预取并按
prompt 长度排序的题目数，设为 `0` 时单进程默认使用
`4 * problem_batch_size`，多卡动态调度默认使用 `problem_batch_size`，避免单个
rank 提前领取过多题目。不同长度 prompt 使用左侧 padding、独立 position ids
和有效 KV mask；提前 EOS 的请求继续占据 batch 槽位，但后续 filler KV 会被
mask 掉。

跨题 batch 可以与同题多候选组合。此时一次 forward 的最大请求数为：

$$
\text{problem\_batch\_size} \times \text{batch\_size}
$$

例如 `problem_batch_size=4`、`batch_size=8`、`num_return_sequences=64` 时，
每次最多并行 32 条轨迹，每组题目分 8 个候选 micro-batch 完成。

## R-KV parity test

```bash
PYTHONPATH="$PWD" python tests/test_rkv_parity.py
PYTHONPATH="$PWD" python tests/test_official_baseline_parity.py
```

这些测试用小张量 oracle 对齐上游 HuggingFace `update_kv` 语义，覆盖
SnapKV、H2O、StreamingLLM、R-KV 的保留 indices、head-wise cache gather、
candidate attention 重归一化、完整 cache similarity、官方 total-budget
保留长度和 debug score 对齐。

论文口径 smoke reproduction：

```bash
PYTHONPATH=. python scripts/eval.py \
  --config configs/experiments/math500_official_b1024.yaml
```
