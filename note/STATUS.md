# 当前阶段

当前重点是整理 note、scripts 工作流、环境和 memory，并推进 TideProbe 实验。

# 最近完成几步

RKV 复现：
- AIME 全量：整理已有的结果。
- Math500：整理实验结果。

已提交并推送当前工作区变更，`main` 与 `origin/main` 均为 `7466223`。

已将 cu128 环境所需的 vLLM 和 flash-attn 纳入 `requirements.txt`，PyTorch 安装说明使用 SJTU cu128 源。

已将 GitHub 仓库 `chillanthony/decodeatt` 的可见性从 private 改为 public，并回查确认生效。

修改全局readme，升级框架依赖版本。

清理收尾：
- `note/framework/` 重新编号为 00-05（数据集 / 评估协议 / 脚本 / 加速 / evaluationsum / 升级）。
- 删除 `scripts/__pycache__/`（11 个已迁至 `kvbench/diagnostics/` 的孤儿 .pyc），并清掉 `strategies/`、`tests/` 下 3 个对应已删模块的 .pyc。

将环境变量、测试入口、评测配置目录和输出大小约束写入 `note/AGENTS.md`；版本与实验细节继续保留在 `note/MEMORY.md`。

补充 `note/repo.md`，按目录记录仓库各部分的职责，并区分代码、配置、实验产物、研究笔记和缓存目录。

审阅 `note/MEMORY.md`：确认环境变量、测试入口、配置目录和输出大小约束应提升为 `AGENTS.md` 的长期规则；版本与实验细节继续留在 MEMORY。

所有 shell launcher 已改为 source `scripts/env.sh`，清除了重复的环境探测和 TideProbe 硬编码仓库路径，并通过 dry-run 验证网格与单策略入口。

整理 `configs/` 与 `scripts/`：保留仍被实验、诊断或测试使用的入口，未发现可安全删除的无引用配置；新增 `scripts/env.sh` 统一仓库根目录、Python、模型、HF 缓存、runs 和 TideProbe 输出路径。

汇总 `note/framework/` 下 00-05 六份文档的核心内容，新增 `note/framework/sum.md`。

新增吞吐量 smoke test：`kvbench/evaluator.py` 统一记录 batch 级吞吐，新增
`configs/experiments/throughput_smoke.yaml` 和
`scripts/experiments/throughput_smoke.sh`，默认本地跑 `fullkv`/`rkv@256`，并发档位为 1/2/4。
当前机器没有项目 conda/torch 环境，未完成 GPU 实测。

完成第一阶段 KV 管理重构：新增 `kv_eviction/backends/` 执行后端协议、
`kv_eviction/cache_manager.py` dense HF cache manager；reference generation、batch
generation、teacher-forced trace 和 evaluator 均可注入 backend/manager。具体迁移边界
和后续 vLLM/block cache 工作记录在 `note/framework/06-chonggou.md`。

完成 per-arm execution engine 选择：结构化 arm 支持 `engine: hf|vllm`，默认 HF；新增
`kvbench/vllm_evaluator.py` 和 `configs/onestrategy/fullkv_vllm.yaml`。当前 vLLM 路径
只支持 FullKV，R-KV/SnapKV 等显式要求 vLLM 时会报错，避免误报已完成 KV 接入。

按本次要求，仅将 `note/AGENTS.md`、`note/MEMORY.md`、`note/STATUS.md` 纳入 Git 提交并推送。

# 之后几步的规划


TideProbe：
- 两级分层预算控制器：offline 层敏感性 probe + 定预算，online head 级重分配。
- 过渡点感知的分段渐进淘汰：entropy 过渡检测 -> 历史段递减 + 当前段豁免，回溯时 rescue 召回。
- 看 `step.md`，整理实验脚本与所有路径。
- 看差异定位，升级叙事。

- 在远端回退commit
- 看一下lhr的环境是否能正常运行
  - 如果不行，求助构建当前requirements的环境依赖，更新，以及把知识沉淀到笔记里面。
  - 如果可以，直接修改当前的requirements，然后把知识沉淀到笔记里面
- 验证版本，修改路径，验证 smoke test，保存镜像。
- 写脚本运行吞吐量smoke test实验

# 已完成归档

rescuekv
通过四维签名定位复发页面。
KV 驱逐机制有盲区（nll kill），证伪 MRI 等。
模拟淘汰之后 nll 不升，acc 无明显增益 AIME，因为保留的可能是错的，签名机制不筛选。

搭建 KV eviction 框架。
挂到 A800 上面跑通 smoke test。
思考挂实验 pipeline。

完成 KV eviction 的 sparse kernel 适配探索。
生成这个主题的 idea 在 ideas 里面。
看 idea 并且迭代。
最后搞明白 A800 的代码运行环境。
写华为明天的汇报 721。
整理华为项目之前的知识。
整理当前的项目现状。
改 batch 化并 smoke test 测吞吐。
调研 RKV、ThinkV、SparseKV 后续工作。
精读 RKV、ThinkV、EPIKV、DeepSeek-V4-FlashMemory、LookaheadKV。
复现 RKV 结果 `aime24_rkv_b1536_s4_8gpu.sh`，整理日志。
看所有 ICLR 26 KV 论文，迭代两个实验方案，提出新的并继续实验计划。
从 RKV 后续进行调研，820 讨论调研，迭代了新 idea TideProbe。
整理 git 仓库分支。
整理当前仓库里的冗余文件（scripts）。
整理 reframing0907。
把 RKV、ThinkV 后续认真看一下。
思考缺口，整理 idea list。

已将归档 README 中的长期有效信息提炼并写入 `note/MEMORY.md`。
