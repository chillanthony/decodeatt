# R-KV AIME24 快速复现

运行脚本：[aime24_rkv_b1536_s4_8gpu.sh](../../scripts/jiqun/aime24_rkv_b1536_s4_8gpu.sh)

## 实验要点

- 模型：`DeepSeek-R1-Distill-Llama-8B`。
- 数据：AIME 2024，共 30 道题。
- 方法：R-KV，固定 KV budget 为 1536。
- 生成：每题采样 4 次，temperature 为 0.6、top-$p$ 为 0.95，最多生成 32768 token。
- 并行：单节点 8 张 A800；8 个独立 rank 动态领取题目，每个 rank 内将同题 4 条候选组成一个 batch。
- 预计耗时：约 50–90 分钟；若较多轨迹生成到 32K 上限，时间会更长。
- 输出：默认写入 `runs/aime24_rkv_b1536_s4_8gpu/result/`，日志写入同一运行目录的 `output/`。

## 运行

```bash
bash scripts/jiqun/aime24_rkv_b1536_s4_8gpu.sh
```

模型、缓存或输出目录不同时，可以通过环境变量覆盖：

```bash
MODEL=/path/to/DeepSeek-R1-Distill-Llama-8B \
HF_HOME=/path/to/hf_cache \
RUN_DIR=/path/to/run \
bash scripts/jiqun/aime24_rkv_b1536_s4_8gpu.sh
```

脚本只运行 `rkv@1536`，因此 8 张卡都会分配给 R-KV；没有同时运行 FullKV 或其他 baseline。
