# TideProbe Step 1：诊断探测

## 目标

Step 1 只验证两个假设，不实现新的在线淘汰机制：

1. **分层假设**：不同 Transformer 层对 KV 压缩的敏感度是否有稳定差异。
2. **过渡段假设**：淘汰命中过渡、回溯或纠错段时，后续 NLL 是否比命中普通推理段上升更多。

Sparse cliff 只作为附属诊断指标，不单独立项。

## 总体流程

```text
FullKV 固定轨迹
  → 过渡点检测
  → 逐层反事实 sparse
  → 淘汰位置与过渡段对齐
  → sparse-cliff 曲线
  → Go / No-go
```

所有实验使用同一条 FullKV 轨迹做 teacher forcing，避免不同策略的生成分叉干扰对比。

## 公共准备

1. 固定模型、AIME 题目、seed、生成参数和输出格式。
2. 统一 `gen_traces.py` 与当前 KVBench 的模型、数据和 prompt 配置。
3. 生成 8–10 条 AIME FullKV 轨迹，并保存 token、logit entropy 和 FullKV 逐 token NLL。

产物默认保存到 `/home/ma-user/work/trace/`（可通过 `TRACE_DIR` 环境变量覆盖）。预计开发 0.5 天，GPU 运行 2–4 小时。

### 完成情况

代码侧公共准备已完成，实际 trace 仍需在 8 张 A100 上运行生成：

- 新建 `tideprobe` 分支，并将 trace 诊断代码放入 `kvbench/diagnostics/`。
- `scripts/gen_traces.py` 已统一复用 KVBench 的模型加载、AIME 数据加载和 prompt 构造逻辑。
- 新增固定配置 `configs/experiments/tideprobe_step1.yaml`：DeepSeek-R1-Distill-Llama-8B、前 10 道 AIME、seed 0、temperature 0.6、top-p 0.95、最多生成 32768 tokens。
- 每条 `.pt` trace 保存 `prompt_ids`、`gen_ids`、逐 token 文本、原始 logit entropy、FullKV 逐 token NLL、完整生成文本及模型/数据/生成参数元数据。
- 新增 `scripts/jiqun/tideprobe_gen_traces_8xa100.sh`，通过 `torchrun` 将 10 道题静态分配到 8 张 A100；已有完整 trace 默认跳过，文件采用原子写入，支持任务断点续跑。
- 全量测试通过，共 39 项；Python 编译、CLI 和 shell 语法检查通过。

运行命令：

```bash
bash scripts/jiqun/tideprobe_gen_traces_8xa100.sh
```

## 实验 1：过渡点检测

对 FullKV 轨迹提取三类信号：

- logit entropy 突变
- attention entropy 突变
- 相邻 token 的 K/V 表示变化（KV delta）

对三个信号做滚动标准化，用局部峰值定位候选过渡点。用以下信息作为弱标签：

- `wait`、`but wait`、`I made a mistake` 等强回溯短语
- 中途结论或 boxed answer 翻转
- 少量人工复核

评价：

- event recall@$\pm16$ tokens
- AUPRC
- 每 1K token 误报数
- 融合信号与三个单信号的消融对比

在全部 8–10 条轨迹上运行，并人工检查约 30–50 个候选点。输出 `transition_events.jsonl`、检测案例图和评价指标。

进入后续实验的条件：融合检测结果明显优于只用 logit entropy。预计开发 0.5–1 天，GPU 运行 2–4 小时。

### 完成情况

实验 1 的代码与运行链路已完成，真实事件和评价指标需在公共准备的 trace 生成后运行得到：

- 新增 `kvbench/diagnostics/transition_signals.py`，沿固定 `gen_ids` 做 FullKV teacher forcing，逐 token 提取各层 attention entropy 和相邻 K/V cosine delta，同时重算 FullKV NLL 用于检查 replay 是否与原轨迹一致。信号保存到 `runs/tideprobe_step1/signals/`。
- 新增 `scripts/jiqun/tideprobe_extract_signals_8xa100.sh`，使用 8 张 A100 按 trace 静态分片提取信号，已有完整信号文件默认跳过，支持断点续跑。
- 新增 `kvbench/diagnostics/transition_detection.py` 和 `scripts/detect_transitions.py`：对 logit entropy 变化、attention entropy 变化和 KV delta 做滚动标准化、局部峰值检测与等权融合。
- 自动弱标签覆盖强回溯短语、中途数字答案翻转和 boxed answer 翻转；另支持传入人工标注 JSONL，格式为 `{"id":"aime-0","token_index":123,"type":"manual"}`。
- 输出 `transition_events.jsonl`、`transition_metrics.json` 和逐 trace 案例图；指标包含 event recall@$\pm16$ tokens、AUPRC、每 1K token 未匹配候选数，以及融合与三个单信号的消融对比。默认保留分数最高的 50 个候选供人工检查。
- 全量测试共 43 项通过；另使用随机初始化的微型 Llama 验证了真实 DynamicCache、eager attention 和 teacher-forcing 接口。

运行顺序：

```bash
# 1. 如尚未生成固定轨迹，先运行公共准备
bash scripts/jiqun/tideprobe_gen_traces_8xa100.sh

# 2. 在 8 张 A100 上提取 attention entropy 和 KV delta
bash scripts/jiqun/tideprobe_extract_signals_8xa100.sh

# 3. 在 CPU 上检测、评估并画图
uv run python scripts/detect_transitions.py \
  --config configs/experiments/tideprobe_step1.yaml
```

如需合并人工标注，在第 3 步增加 `--manual-labels path/to/manual_labels.jsonl`。

## 实验 2：逐层敏感度扫描

当前 cache 不支持各层不等长度，因此不物理删除单层 KV，而是在目标层 attention 上屏蔽指定历史 token。其他层保持 FullKV。

设置：

- 预算：$256$ 和 $512$；有区分度后再扩到 $128$ 和 $1024$
- 先按连续 4 层一组扫描，再在敏感组内逐层扫描
- 主对照使用相同的随机 keep mask，每个 mask 使用 3 个 seed，用 Window/R-KV mask 做辅助验证
- 在均匀位置、过渡点和匹配的普通位置各测探一组

层敏感度定义为接下来 32 token 的 NLL 增量：

$$
S_l(B)=\operatorname{NLL}_{l,B}-\operatorname{NLL}_{\mathrm{FullKV}}
$$

最终输出 `layer_sensitivity.csv`、layer $\times$ budget 热力图和层敏感曲线。

进入后续机制开发的条件：敏感层排序跨轨迹、跨 seed 基本稳定。预计开发 1–2 天，多卡 GPU 运行 3–8 小时。

### 完成情况

实验 2 的代码与运行链路已完成，真实层敏感度和稳定性结论需在实验 1 产物生成后运行得到：

- 新增 `kvbench/diagnostics/layer_sensitivity.py`：在固定 FullKV 轨迹上，以目标 token 前一位置为 query 起点，对目标层或连续层组注入专属 attention mask，其他层和所有层的 KV cache 保持 FullKV；以未来 32 token 的平均 NLL 增量计算敏感度。
- 同一 trace/probe 只做一次长上下文 prefill，之后复用并裁回 DynamicCache 扫描不同层、预算和 mask，避免重复构建长上下文；每次同时检查 block replay NLL 与轨迹中的 FullKV NLL 是否对齐。
- 主实验支持连续 4 层一组扫描、预算 $256/512$、共享 Random mask 的 3 个 seed；辅助支持 Window mask，并直接复用仓库现有 R-KV 选择逻辑生成 R-KV mask。
- probe 默认覆盖每条轨迹一个均匀位置、分数最高的一个过渡点，以及在附近且避开过渡窗口的一个匹配普通位置；也可通过 CLI 单独选择 probe 类型、题目和层范围。
- 新增 `scripts/jiqun/tideprobe_layer_sensitivity_8xa100.sh`，按 trace/probe 在 8 张 A100 上静态分片；不同主实验、细扫和辅助实验使用独立 run tag，不会互相覆盖 shard。
- 新增 `kvbench/diagnostics/layer_analysis.py` 和 `scripts/summarize_layer_sensitivity.py`，输出 `layer_sensitivity.csv`、聚合 CSV、layer $\times$ budget 热力图、敏感曲线和 `layer_sensitivity_metrics.json`；指标包含跨 trace 和跨 random seed 的层排序 Spearman 稳定性。
- 全量测试共 46 项通过；随机初始化的微型 Llama 已验证 SDPA、DynamicCache、目标层 mask、Random/Window/R-KV mask 和 NLL 对齐。

运行顺序：

```bash
# 1. 连续 4 层一组的 Pilot：Random 3 seeds + Window，预算 256/512
bash scripts/jiqun/tideprobe_layer_sensitivity_8xa100.sh

# 2. 根据 layer_sensitivity_summary.csv 选出敏感层组后逐层细扫
SCAN_MODE=layer LAYERS=8-11 RUN_TAG=fine_8_11 \
  bash scripts/jiqun/tideprobe_layer_sensitivity_8xa100.sh

# 3. 可选：对过渡点和匹配普通位置补 R-KV mask 辅助验证
MASK_POLICIES=rkv PROBE_TYPES=transition,ordinary RUN_TAG=rkv_aux \
  bash scripts/jiqun/tideprobe_layer_sensitivity_8xa100.sh
```

第 2 步的 `LAYERS=8-11` 只是命令示例，应替换为第 1 步实际发现的敏感层组；每次 8 卡任务结束后会自动重新合并全部 shard 并生成 CSV、指标和图。

## 实验 3：淘汰与过渡段对齐

扩展现有 `score_trace_token()`，让它在可选 debug 模式下返回：

- 逐 token NLL
- eviction step
- 被淘汰和保留的真实 token 位置

在固定轨迹上对比 FullKV、R-KV、SnapKV、Window 和 Random。Pilot 对 R-KV、SnapKV、Window 和 Random 使用预算 $512,1024,1536$，8–10 条轨迹共约 $4\times3\times10=120$ 次 teacher-forced 运行，FullKV 作为共享对照。

对每次驱逐计算它命中过渡窗口的比例，再统计驱逐后 32 token 相对 FullKV 的 NLL 增量。对比：

- 高 transition-hit 事件
- 低 transition-hit 事件
- 淘汰时刻、淘汰量和 token 年龄匹配的普通事件

输出 `eviction_alignment.csv` 和 transition-hit–NLL 关系图。如果命中过渡段后的 NLL 损失稳定更大，则支持思路 B。

预计开发 1–1.5 天，多卡 GPU 运行 2–6 小时。

### 完成情况

实验 3 的代码与运行链路已完成，真实 transition-hit 与 NLL 关系需在实验 1 的固定轨迹和过渡事件生成后运行得到：

- 扩展 `kv_eviction/runner_token.py::score_trace_token()` 的可选 debug 模式，返回逐 token NLL、eviction step、cache 长度，以及每次驱逐中被淘汰和保留的真实 token 位置。
- R-KV 和 SnapKV 会在不同层、不同 KV head 保留不同位置，因此 debug 额外维护 per-layer/per-head 真实位置映射，并以 `[真实位置, 层头槽位计数]` 压缩保存；这比只记录代表 head 的 `slot_pos` 更准确，也避免输出文件过度膨胀。
- 新增 `kvbench/diagnostics/eviction_scoring.py` 和 `scripts/score_eviction_alignment.py`，在固定轨迹上 teacher forcing 扫描 R-KV、SnapKV、Window、Random 的预算 $512/1024/1536$；FullKV 逐 token NLL 直接复用公共轨迹作为共享对照。
- 新增 `scripts/jiqun/tideprobe_eviction_alignment_8xa100.sh`：8 个 rank 按四种策略分为四个子组，每种策略由两个 rank 分摊 trace $\times$ budget 任务；每个原始 `.pt` 结果独立原子写入并默认跳过已有文件，可断点续跑。
- 新增 `kvbench/diagnostics/eviction_alignment.py` 和 `scripts/analyze_eviction_alignment.py`：按过渡点 $\pm16$ token 窗口计算每次驱逐的 transition-hit 比例，再计算驱逐后 32 token 相对 FullKV 的 NLL 增量。
- 将 transition-hit 至少为 1% 的事件记为 high-hit、完全未命中的事件记为 low-hit；在同一 trace、策略和预算内，按驱逐时刻、有效驱逐量和被淘汰 token 平均年龄为 high-hit 事件选择普通匹配对照。
- 输出 `eviction_alignment.csv`、`eviction_alignment_metrics.json` 和 `figures/transition_hit_nll.png`；指标包含各策略/预算/high-low 组的 NLL 增量、transition-hit–NLL Spearman 相关性，以及 high-hit 与匹配对照的成对 NLL 差。
- 全量测试共 49 项通过；微型 Llama 已验证 Random/R-KV 的逐头真实位置计数，合成实验已验证 transition-hit、未来 NLL 和匹配对照分析。

运行命令：

```bash
bash scripts/jiqun/tideprobe_eviction_alignment_8xa100.sh
```

脚本会先在 8 张 A100 上生成四策略三预算的原始评分文件，再自动在 CPU 上合并生成 CSV、指标和关系图。如只需重新分析已有原始结果，可运行：

```bash
uv run python scripts/analyze_eviction_alignment.py \
  --config configs/experiments/tideprobe_step1.yaml
```

## 实验 4：Sparse cliff

先直接复用实验 3 的 $512,1024,1536$ 三个预算点；如果出现明显拐点迹象，再补 $2048,2560$。计算过渡/纠错窗口的 Correction Fidelity：

$$
\operatorname{CF}(B)
=
1-
\frac{
\operatorname{NLL}_{B}^{\mathrm{transition}}
-
\operatorname{NLL}_{\mathrm{FullKV}}^{\mathrm{transition}}
}{
\operatorname{NLL}_{\mathrm{FullKV}}^{\mathrm{transition}}
}
$$

输出 `sparse_cliff.csv` 和 budget–CF 曲线。Pilot 阶段不强求精确定位 cliff；该结果只用来比较策略安全边界，不作为独立贡献。预计 0.5 天，主要为 CPU 分析。

### 完成情况

实验 4 的代码与运行链路已完成，真实 Sparse Cliff 结论需在实验 3 的原始逐 token NLL 生成后运行得到：

- 新增 `kvbench/diagnostics/sparse_cliff.py` 和 `scripts/analyze_sparse_cliff.py`，直接复用实验 3 的逐 token sparse NLL 与公共 FullKV NLL，不增加新的 GPU forward。
- 对每条 trace、策略和预算构造过渡点 $\pm16$ token 窗口并集，分别计算过渡段与普通段的 FullKV/sparse NLL、NLL 增量和 fidelity；过渡段指标按定义输出 Correction Fidelity。
- `sparse_cliff.csv` 保存逐 trace 结果，`sparse_cliff_summary.csv` 按策略和预算用 token 数加权汇总，避免不同轨迹的过渡窗口数量差异造成偏置。
- 新增 pilot 诊断：相邻预算的 CF 增益及每 512 tokens 归一化增益、最大增益区间、三点以上的最强斜率变化位置、CF 单调性违反次数，以及首次达到默认 0.9 安全阈值的预算。该诊断只报告迹象，不自动宣称存在精确 cliff。
- 分析会严格检查每条 trace $\times$ 策略 $\times$ 预算的实验 3 原始文件是否齐全，缺失时直接报错，避免用不完整矩阵画曲线。
- 输出 `sparse_cliff_metrics.json` 和 `figures/budget_cf.png`，图中包含四种策略的 budget–CF 曲线、跨 trace 标准差、FullKV 基准线和安全阈值线。
- 新增 `scripts/jiqun/tideprobe_sparse_cliff.sh` 作为 CPU 分析入口；全量测试共 50 项通过，合成三预算曲线已验证 CF、阈值预算和单调性诊断。

标准 Pilot 运行：

```bash
bash scripts/jiqun/tideprobe_sparse_cliff.sh
```

如果 Pilot 出现明显拐点迹象，再补 $2048,2560$ 两个 GPU 预算点并重新分析五点曲线：

```bash
# 复用实验 3 的 8 卡评分器，只补缺少的两个预算
BUDGETS=2048,2560 \
  bash scripts/jiqun/tideprobe_eviction_alignment_8xa100.sh

# 合并全部五个预算点重新计算 CF
BUDGETS=512,1024,1536,2048,2560 \
  bash scripts/jiqun/tideprobe_sparse_cliff.sh
```

## 最小实验矩阵

### Pilot

- 模型：DeepSeek-R1-Distill-Llama-8B
- 数据：8–10 条 AIME FullKV 轨迹
- 层扫描预算：$256,512$
- 策略扫描预算：$512,1024,1536$

Pilot 有清晰信号后，再扩展到 AIME 30 题和约 50 条 MATH-500 轨迹。

## 开发与运行排期

| 阶段 | 开发与分析 | 多卡 GPU 运行 |
|---|---:|---:|
| 公共准备 | 0.5 天 | 2–4 小时 |
| 过渡点检测 | 0.5–1 天 | 2–4 小时 |
| 逐层敏感度扫描 | 1–2 天 | 3–8 小时 |
| 淘汰与过渡段对齐 | 1–1.5 天 | 2–6 小时 |
| Sparse cliff 与报告 | 0.5 天 | 主要为 CPU |

总计约 3.5–5.5 个工作日。实验 2 按层组/轨迹分卡，实验 3 按策略/预算/轨迹分卡。

## 需要的代码改动

1. 统一 `gen_traces.py` 与当前 `kvbench` 的模型、数据和生成配置。
2. 增加过渡信号提取与事件检测脚本。
3. 增加单层 attention mask 反事实探针。
4. 为 `score_trace_token()` 增加可选的逐 token NLL 和 eviction-position 输出。
5. 增加一个分析脚本，统一生成曲线、CSV 和汇总报告。

诊断代码放在 `kvbench/diagnostics/`，策略选择仍复用 `kv_eviction/strategies/`，不另写一套 R-KV/SnapKV。

## 产物

```text
/home/ma-user/work/trace/
  aime-*.pt

runs/tideprobe_step1/
  transition_events.jsonl
  layer_sensitivity.csv
  eviction_alignment.csv
  sparse_cliff.csv
  figures/
  report.md
```

## Go/No-go

### 思路 A

Go：至少两个预算下出现稳定的敏感层和不敏感层，且结论不随 mask seed 或选择策略反转。

No-go：层敏感曲线基本平坦，或者跨轨迹、跨 mask 不稳定。

### 思路 B

Go：融合检测器优于单独 entropy，且命中过渡段的驱逐事件在匹配对照后仍产生更大的后续 NLL 损失。

No-go：过渡点不可靠，或 transition-hit 与 NLL 损失没有稳定关系。

| A | B | 后续路线 |
|---|---|---|
| Go | Go | 先做过渡点分段淘汰，再叠加分层预算 |
| Go | No-go | 只做分层预算 |
| No-go | Go | 只做过渡点分段淘汰 |
| No-go | No-go | 停止 TideProbe 机制开发 |
