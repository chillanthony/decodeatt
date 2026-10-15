# R-KV AIME24 复现

## 实验要点

- 模型：`DeepSeek-R1-Distill-Llama-8B`。
- 数据：AIME 2024，共 30 道题。
- 方法：FullKV，以及 budget 为 1024、1536、2048、2560 的 R-KV、SnapKV。
- 生成：每题采样 64 次，temperature 为 0.6、top-$p$ 为 0.95，最多生成 32768 token。
- 并行：单节点 8 张卡；8 个独立 rank 动态领取题目，每个 rank 内将同题 4 条候选组成一个 batch。
- R-KV 超参对齐论文：$B_{buffer}=128$、$\alpha=8$、$\lambda=0.1$、pool kernel 7，每 128 decoding step 压缩一次。

## 实验结果

论文: note/paper/r-kv.pdf 

Table 2 在 R1-Llama-8B / AIME24 上报告：FullKV 为 $49.79\%$；R-KV@1536 为 $51.56\%$；SnapKV@1536 为 $26.04\%$。论文每题生成 64 条 response，共 $30\times64=1920$ 条，并将所有 response 的正确率记为 pass@1。

九组 `s64` 结果均为 30 道题、每题 64 条 response，共 1920 条；因此这里的 `accuracy/pass_at_1` 是所有 response 的正确率，和论文 Table 2 的统计口径一致。

| 方法 | Budget | 本地 s64 | 正确/总数 | 论文 | 差值 |
|---|---:|---:|---:|---:|---:|
| FullKV | — | 48.13% | 924/1920 | 49.79% | -1.66 pp |
| R-KV | 1024 | 44.01% | 845/1920 | 45.26% | -1.25 pp |
| R-KV | 1536 | 48.59% | 933/1920 | 51.56% | -2.97 pp |
| R-KV | 2048 | 49.53% | 951/1920 | 52.29% | -2.76 pp |
| R-KV | 2560 | 50.10% | 962/1920 | 53.85% | -3.75 pp |
| SnapKV | 1024 | 18.54% | 356/1920 | 15.73% | +2.81 pp |
| SnapKV | 1536 | 27.34% | 525/1920 | 26.04% | +1.30 pp |
| SnapKV | 2048 | 33.91% | 651/1920 | 32.76% | +1.15 pp |
| SnapKV | 2560 | 39.58% | 760/1920 | 39.43% | +0.15 pp |

### 主要结论

1. **定性复现基本成立**：R-KV 和 SnapKV 都随 budget 增大而提升；R-KV 在 1536/2048 上分别为 48.59%/49.53%，显著高于 SnapKV 的 27.34%/33.91%。
2. **R-KV@1536 未超过 FullKV，但 R-KV@2048 已略高于 FullKV**：本地分别为 48.59%、49.53% 和 48.13%；相对论文仍低 2.97 pp、2.76 pp 和 1.66 pp。
3. **SnapKV 与论文非常接近**：1024/1536/2048/2560 与论文差值分别为 +2.81、+1.30、+1.15、+0.15 pp，说明该 baseline 的趋势和绝对水平复现得较好。

## 效率结果

下表使用各聚合 JSON 中的 `mean_tokens_per_sec`、`max_peak_memory_bytes`、`mean_effective_cache_len` 和 `mean_elapsed_sec`；吞吐是实验汇总口径，不是单个 rank 的峰值。

| 方法 | Budget | 吞吐 (tok/s) | 峰值显存 | 平均有效 cache | 平均生成长度 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|
| FullKV | — | 31.46 | 51.37 GiB | 15,901 | 15,798 | 767.68 s |
| R-KV | 1024 | 234.35 | 33.36 GiB | 1,019 | 20,129 | 84.62 s |
| R-KV | 1536 | 154.38 | 41.49 GiB | 1,530 | 18,201 | 115.58 s |
| R-KV | 2048 | 116.19 | 49.59 GiB | 2,042 | 17,343 | 148.61 s |
| SnapKV | 1024 | 224.77 | 25.86 GiB | 1,018 | 18,542 | 81.61 s |
| SnapKV | 1536 | 162.18 | 30.66 GiB | 1,530 | 18,210 | 112.23 s |
| SnapKV | 2048 | 129.62 | 35.44 GiB | 2,042 | 17,518 | 134.49 s |
| SnapKV | 2560 | 102.05 | 40.21 GiB | 2,552 | 16,693 | 165.27 s |

相对 FullKV，R-KV@1536 的峰值显存降低约 19.2%，平均有效 cache 降低约 90.4%，吞吐约为 4.91 倍；SnapKV@1536 的峰值显存降低约 40.3%，吞吐约为 5.15 倍。R-KV@1536 的平均 eviction 总耗时为 10.33 s，SnapKV@1536 为 1.13 s，说明 R-KV 的相似度/冗余计算开销明显更高。注意：这是自然生成长度、batch=4 的端到端汇总，不能直接等同于论文固定长度或最大 batch 的效率数字。

# R-KV MATH500 复现

## 实验要点

- 模型：`DeepSeek-R1-Distill-Llama-8B`；数据集：MATH500。
- 对比 FullKV、R-KV 和 SnapKV，budget 为 128、256、512、768、1024、1536、2048。
- 每个实验固定 batch=4，并使用 8 个并行 rank；结果汇总见 `local/runs_extracted/runs/math500_llama8b_main_grid_s4/summary.csv`。

## 实验结果

原文使用 `DeepSeek-R1-Distill-Llama-8B` 在 MATH-500 上评测 FullKV、R-KV 和 SnapKV，最大生成长度设为 16,384 token，并报告平均生成长度约 2,979.1 token；论文的 budget-ratio 图中将约 1,024 token budget 标为约 34% cache budget。原文结论是：R-KV 在 MATH-500 上约 34% 的 KV cache budget 下达到与 FullKV 相当的准确率，并显著优于 SnapKV；在固定 budget 分析中，R-KV@1024 被作为达到无损压缩的关键配置，R-KV@1536 则进一步提供更高的准确率裕量。论文实验使用 NVIDIA A100 80G。

| 方法 | Budget | 本地正确/总数 | 本地 Accuracy | 原文 Accuracy | 差值 | 平均生成长度 | 吞吐 (tok/s) | 平均耗时 (s) | 峰值显存 (GiB) | 最终 cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullKV | — | 1746/2000 | 87.30% | 82.38% | +4.92 pp | 3,829 | 87.94 | 80.65 | 24.29 | 5,581 |
| R-KV | 128 | 1053/2000 | 52.65% | 51.08% | +1.57 pp | 8,139 | 104.41 | 74.55 | 16.17 | 162 |
| R-KV | 256 | 1429/2020 | 70.74% | 67.39% | +3.35 pp | 6,311 | 104.13 | 58.84 | 16.17 | 299 |
| R-KV | 512 | 1584/2000 | 79.20% | 76.92% | +2.28 pp | 4,997 | 103.06 | 48.48 | 16.17 | 558 |
| R-KV | 768 | 1651/2008 | 82.22% | 80.21% | +2.01 pp | 4,446 | 104.34 | 44.07 | 16.17 | 808 |
| R-KV | 1024 | 1667/2016 | 82.69% | 81.34% | +1.35 pp | 4,250 | 103.06 | 42.77 | 16.17 | 1,049 |
| R-KV | 1536 | 1686/1988 | 84.81% | 82.34% | +2.47 pp | 4,069 | 96.76 | 46.84 | 16.60 | 1,503 |
| R-KV | 2048 | 1714/2000 | 85.70% | 82.65% | +3.05 pp | 3,897 | 92.95 | 48.34 | 17.14 | 1,913 |
| SnapKV | 128 | 481/2000 | 24.05% | 32.53% | -8.48 pp | 5,480 | 84.67 | 69.29 | 16.16 | 172 |
| SnapKV | 256 | 921/2000 | 46.05% | 50.07% | -4.02 pp | 5,101 | 94.34 | 58.73 | 16.15 | 304 |
| SnapKV | 512 | 1346/2036 | 66.11% | 64.03% | +2.08 pp | 4,484 | 100.79 | 48.01 | 16.17 | 555 |
| SnapKV | 768 | 1468/1976 | 74.29% | 70.81% | +3.48 pp | 4,255 | 102.35 | 45.10 | 16.03 | 802 |
| SnapKV | 1024 | 1587/2028 | 78.25% | 74.43% | +3.82 pp | 4,168 | 101.69 | 43.94 | 16.03 | 1,054 |
| SnapKV | 1536 | 1707/2048 | 83.35% | 78.43% | +4.92 pp | 3,814 | 98.72 | 42.01 | 16.17 | 1,518 |
| SnapKV | 2048 | 1773/2100 | 84.43% | 80.50% | +3.93 pp | 3,825 | 96.01 | 45.45 | 16.25 | 1,888 |


### 主要结论

1. R-KV 和 SnapKV 的准确率都随 budget 增大而提升，但 FullKV 仍然最高，为 87.30%。
2. R-KV@2048 达到 85.70%，最接近 FullKV；SnapKV@2048 为 84.43%，略低于 R-KV。
3. R-KV 在 budget 768–1024 时吞吐约 103–104 tok/s，明显高于 FullKV 的 87.94 tok/s；继续增大 budget 后吞吐逐步下降。
4. 压缩方法峰值显存约 16–17 GiB，低于 FullKV 的 24.29 GiB；但低 budget 会显著拉长生成长度并降低准确率。
5. R-KV@1024 在准确率、吞吐和显存之间表现出较好的折中，而 SnapKV 在相同 budget 下准确率整体低于 R-KV。

## 效率结果

相对 FullKV，R-KV@1024 的平均吞吐提升约 17.2%，平均耗时减少约 47.0%，峰值显存降低约 33.4%；其最终 cache 长度约为 FullKV 的 18.8%。但压缩方法的平均生成长度更长，说明端到端收益不能只由 cache 长度判断，还需同时关注生成长度和吞吐。

# 脚本生成

- FullKV 每个实验目录生成 1 个 `.log`、1 个聚合 `.json` 和 8 个 `rank*.sdpa.json` 分片文件。
- R-KV / SnapKV 每个实验目录生成 1 个 `.log`、1 个聚合 `.json` 和 8 个 `rank*.eager.json` 分片文件。
- 分片文件名中的 `rankN` 表示并行 rank，backend 后缀表示 attention backend：FullKV 为 `sdpa`，R-KV/SnapKV 为 `eager`。

- `log_mode`：表示日志输出模式，通常为 `brief`，说明采用简略日志。
- 顶层 `num_problems`：表示整个实验包含的题目数量。
- `summary`：存放按实验方法和 budget 汇总的结果。
- `rkv@1536` 等方法键：表示当前实验的方法名称及其 KV budget。
- `accuracy`：表示所有生成结果中的总体正确率。
- `pass_at_1`：表示 pass@1 指标，即按单次结果统计的正确率。
- `num_problems`：表示参与该方法统计的题目数。
- `correct`：表示判定为正确的 response 数量。
- `total`：表示 response 总数量。
- `mean_head_budget_std`：表示不同 attention head 的 budget 标准差，越小表示各 head 分配越均匀。
- `max_compression_ratio`：表示实验中出现的最大名义压缩比例。
- `mean_elapsed_sec`：表示平均端到端运行耗时，单位为秒。
- `mean_other_decode_sec`：表示除 attention forward 和 eviction 外其他 decoding 操作的平均耗时。
- `mean_eviction_sec_mean`：表示单次 eviction 操作的平均耗时。
- `mean_n_evict`：表示平均发生 eviction 的次数。
- `min_effective_compression_ratio`：表示实验中最低的有效压缩比例。
- `mean_final_cache_len`：表示生成结束时 KV cache 长度的平均值。
- `mean_effective_cache_len`：表示考虑各 attention head 实际使用情况后的平均有效 cache 长度。
- `max_effective_cache_len`：表示实验中出现的最大有效 cache 长度。
- `mean_total_effective_kv_tokens`：表示每个样本平均实际参与 KV 计算的 token 总量。
- `min_compression_ratio`：表示实验中最低的名义压缩比例。
- `max_effective_compression_ratio`：表示实验中最高的有效压缩比例。
- `mean_total_evicted_tokens`：表示每个样本平均被 eviction 删除的 token 数量。
- `mean_decode_sec`：表示平均 decoding 阶段总耗时，单位为秒。
- `mean_decode_forward_sec`：表示平均 decoding forward 计算耗时。
- `mean_gen_len`：表示平均生成 token 数量，可用于判断生成长度差异。
- `mean_num_underfilled_heads`：表示平均没有填满目标 budget 的 attention head 数量。
- `mean_prefill_sec`：表示平均 prefill 阶段耗时，单位为秒。
- `mean_tokens_per_sec`：表示平均生成吞吐，单位为 token/秒。
- `mean_head_budget_entropy`：表示各 attention head budget 分配的熵，可衡量分配是否均匀。
- `mean_effective_compression_ratio`：表示实际有效 cache 相对于原始 cache 的平均比例。
- `mean_attention_observation_sec`：表示收集 attention 观测信息的平均耗时。
- `mean_eviction_sec_total`：表示每个样本所有 eviction 操作耗时的平均总和。
- `mean_effective_evicted_per_event`：表示每次 eviction 平均删除的有效 KV token 数量。
- `mean_total_effective_evicted_tokens`：表示每个样本平均实际有效删除的 KV token 总量。
- `mean_evicted_per_event`：表示每次 eviction 平均删除的名义 token 数量。
- `max_peak_memory_bytes`：表示实验期间观测到的最大 GPU 显存占用，单位为字节。
- `max_head_budget_max`：表示实验中单个 attention head 获得的最大 budget。
- `mean_compression_ratio`：表示名义 cache 长度相对于原始 cache 长度的平均比例。
- `min_head_budget_min`：表示实验中单个 attention head 获得的最小 budget。
- `mean_head_budget_mean`：表示所有 attention head 的平均 budget。


# 测速实验


## 原文结果

论文在 **NVIDIA A100 80GB** 上，以 **Llama 3-8B** 为模型，比较 FullKV、SnapKV 与 R-KV 的端到端生成吞吐量。吞吐量单位为 token/s；`max` 表示在显存允许下的最大 batch size。

### 生成长度 8K

| 方法/设置 | KV budget | 显存节省 | Batch | 吞吐量 (token/s) |
|---|---:|---:|---:|---:|
| FullKV | — | — | 1 | 75.44 |
| FullKV | — | — | 62 (max) | 849.13 |
| R-KV | Fixed 1024 | 87.50% | 1 | 80.46 |
| R-KV | Fixed 1024 | 87.50% | 402 (max) | 3,251.52 |
| R-KV | Fixed 1536 | 81.25% | 287 (max) | 2,525.75 |
| R-KV | Fixed 3072 | 62.50% | 150 (max) | 1,520.99 |
| R-KV | Ratio 10% (819) | 90.00% | 479 (max) | 3,809.15 |
| R-KV | Ratio 34% (2,785) | 66.00% | 167 (max) | 1,608.01 |
| R-KV | Ratio 54% (4,423) | 46.00% | 105 (max) | 1,257.83 |

### 生成长度 16K

| 方法/设置 | KV budget | 显存节省 | Batch | 吞吐量 (token/s) |
|---|---:|---:|---:|---:|
| FullKV | — | — | 1 | 69.41 |
| FullKV | — | — | 30 (max) | 347.03 |
| R-KV | Fixed 1024 | 93.75% | 1 | 80.95 |
| R-KV | Fixed 1024 | 93.75% | 402 (max) | 3,188.82 |
| R-KV | Fixed 1536 | 90.63% | 287 (max) | 2,447.61 |
| R-KV | Fixed 3072 | 81.25% | 150 (max) | 1,406.28 |
| R-KV | Ratio 10% (1,638) | 90.00% | 271 (max) | 2,300.28 |
| R-KV | Ratio 34% (5,570) | 66.00% | 82 (max) | 797.43 |
| R-KV | Ratio 54% (8,847) | 46.00% | 46 (max) | 584.77 |

论文还报告：8K、固定 budget=1024 时约 **3.8×** 吞吐提升；16K、固定 budget=1024 时约 **9.19×**；比例 budget=10% 时，16K 约 **6.6×**（相对 FullKV 最大 batch）。

## 实现流程

参考仓库 https://github.com/zefan-cai/r-kv
以下流程用于在当前仓库中复现 R-KV 效率实验，推荐使用仓库现有的 vLLM benchmark；具体参数需根据本机 GPU 显存、CUDA、PyTorch 和 vLLM 版本调整。

### 1. 准备环境

安装匹配的 CUDA、PyTorch 和 vLLM 依赖，并应用 R-KV 修改：

```bash
bash scripts/apply_rkv.sh
pip install -e vllm-src
```

### 2. 准备数据

```bash
cd vLLM/benchmark
./prepare_data.sh
```

### 3. 测试 FullKV 基线

```bash
python eval.py --n 200 --label fullkv
```

### 4. 测试 R-KV

```bash
VLLM_V1_R_KV_BUDGET=256 \
VLLM_V1_R_KV_BUFFER=128 \
python eval.py --n 200 --label rkv_b256_buf128
```

### 5. 扫描不同配置

依次测试 `budget=128/256/512/1024`，并固定或扫描 `buffer=128/256`。记录吞吐量、准确率、compaction 次数和 GPU 显存占用。

### 6. 进行显存压力实验

在相同并发度下分别运行 FullKV 和 R-KV，并降低显存利用率参数，例如 vLLM 的 `--gpu-memory-utilization`；如果使用 SGLang，则调整 `--mem-fraction-static`。显存受限时更容易观察到 R-KV 的吞吐优势。
