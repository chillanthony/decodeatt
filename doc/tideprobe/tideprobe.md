# TideProbe（潮汐探针）

> 实验代号 TideProbe：`Tide` 对应分层预算的 Reasoning Wave，`Probe` 对应在引入新机制前先做诊断探测。该计划由原 EntropyGuard 与 SparseSafe 两条 idea 归并而来。

结合现有 notes、R-KV 复现结果(attention 打分保留的关键 token 在 AIME 上不涨 acc、NLL 反而变差),以及 2026 年新工作,判断最值得做的两个方向如下。

## 思路 A:分层预算分配(Layer/Head-wise Budget Allocation)

R-KV 类方法按统一全局预算打分淘汰,但 2026 年多个工作(ReasonAlloc、Lethe、Crystal-KV)证实:各层对 KV 的需求是非线性的、且主要由模型架构决定,小预算(128–512)下收益最大。这与你现有 rkv 打分完全正交,可以直接叠加。

- **路径 A1(offline)**:小样本 probe 逐层扫描淘汰敏感性,拟合 "Reasoning Wave" 层需求曲线,把总预算按层重分配后再跑 R-KV 打分。
- **路径 A2(online)**:解码时按 head 的注意力集中度/信息效用动态把预算从"死头"转给高活跃头,替代静态均匀分配。

## 思路 B:过渡点感知的分段渐进淘汰(Transition-Aware Progressive Aging)

ThinKV 发现 CoT 中存在"过渡 thought"(不确定性/回溯 token):每次过渡出现,前面所有推理段的重要性整体衰减。这个机制直接解释了 RescueKV 盲区现象——关键 token 不是靠静态打分找的,而是由推理轨迹结构决定的。检测信号(logit/attention entropy 突变、回溯词)是 forward 的免费副产品。

- **路径 B1(aging)**:在线检测过渡点,每次触发只对当前段豁免保护、对历史段执行递减保留调度(如 64→32→16),替代均匀 token 级淘汰。
- **路径 B2(rescue)**:检测到回溯(自纠错)事件时,临时恢复刚被淘汰的旧段 KV(查询时间局部性高,恢复代价可接受)。

## 实现顺序

**Step 1 — 诊断探测,不写任何新机制**

用现有框架跑探测,验证两个思路的假设成不成立:

- **过渡点检测器**:由原来的 `entropyguard 回滚点检测` + `靠 entropy/nll 检查关键分支点` 合并而来,用 `kvbench.diagnostics.transition_detection` 根据 logit/attention entropy 突变 + kvdelta 定位过渡/回溯点。
- **模拟 sparse**:用模拟淘汰逐层扫描敏感性,画"层敏感曲线"(决定思路 A 有无区分度);把淘汰位置和过渡点对齐,看命中过渡段时 NLL 掉得最多吗(决定思路 B 值不值得做)。
- **sparsecliff 降为诊断指标**:用多种策略跑,检测纠正悬崖,作为验证"保护机制是否真的推迟了悬崖"的度量,不单独立项。

**Step 2 — 过渡点感知的分段渐进淘汰(思路 B)**

在线检测过渡点 → 当前段豁免保护 + 历史段递减保留(如 64→32→16),把 R-KV 打分套进去。

**Step 3 — 叠加分层预算分配(思路 A)**

在 Step 2 的分段淘汰之上,用 Step 1 的层敏感曲线把预算按层重分配,验证是否 1+1>2。

**Step 4 — 可选增强(主线验证后才做)**

`Dual-Pool selection`、`持久分数预测器`、`blockwise/tokenwise 两层打分`、`query-adaptive rescue`、`spanning 滑窗扩展打分器` 都属于"锦上添花"的变体,主线机制验证有效后再逐个加,避免一开始陷入变体矩阵。
