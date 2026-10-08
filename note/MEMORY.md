# 环境与依赖

- 项目用于长文本推理场景的 KV Cache 淘汰策略评测，主入口是 `scripts/eval.py`，主复现模型是 `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`。
- 使用 conda 提供 Python 3.11 隔离环境，依赖统一通过 pip 安装；核心依赖由 `requirements.txt` 管理。
- CUDA 环境使用 cu128，驱动至少为 550；官方源不可用时，先从 SJTU cu128 索引单独安装 `torch==2.11.0`，再安装其余依赖。
- `requirements.txt` 包含 `vllm==0.25.1` 和 `flash-attn>=2.6`；安装前需准备 CUDA 工具链，安装 requirements 时使用 `--no-build-isolation`。
- 脚本通过 `PYTHON_BIN` 指定解释器；从仓库外运行时必须设置 `PYTHONPATH`。多卡脚本还需要按机器覆盖 `HF_HOME`、`HF_ENDPOINT` 和 `HF_DATASETS_CACHE`。

# 评测结构

- `configs/experiments/` 用于正式实验，`configs/onestrategy/` 用于单策略补跑或调试；`kvbench/` 负责评测流程与指标，`kv_eviction/` 负责 runner 和策略实现。
- 支持 `fullkv`、`snapkv`、`h2o`、`streamingllm`、`rkv`，以及诊断用的 `random`、`window`；策略实现位于 `kv_eviction/strategies/`，注册和 arm 解析分别在 `token.py`、`kvbench/policies.py`。
- R-KV 默认每 128 tokens 压缩一次，保留 `B - alpha` 个候选 token 和最后 `alpha=8` 个 observation tokens；`B` 表示压缩后的总 cache 长度。
- 结果记录吞吐、耗时、显存、压缩率、淘汰事件、cache 长度曲线、head budget 分布及 prefill/decode/eviction 时间拆分。
- 大规模实验可使用 `log_mode=brief`，只保留汇总指标；默认 `full` 会保留逐题文本、候选、cache 曲线和淘汰事件。

# 运行与验证

- 所有 shell 入口统一 source `scripts/env.sh`；`PYTHON_BIN` 默认取已激活 conda 环境的解释器，也可显式覆盖。`HF_HOME`、`HF_ENDPOINT`、`HF_DATASETS_CACHE`、`RUNS_ROOT`、`TIDEPROBE_ROOT` 和 `TRACE_DIR` 可按机器覆盖。

- 支持同题多候选 batch（`batch-size`、`num-return-sequences`）和跨题 batch（`problem-batch-size`、`prompt-bucket-size`）；两者组合时最大并行请求数为 `problem_batch_size * batch_size`。
- `tests/test_rkv_parity.py` 和 `tests/test_official_baseline_parity.py` 用小张量 oracle 对齐 SnapKV、H2O、StreamingLLM、R-KV 的 cache 更新语义。
- 常规验证方式是从仓库根目录设置 `PYTHONPATH` 后运行 `tests/test_*.py`；正式评测使用 `configs/experiments/` 下的 YAML 配置。

# 研究结论

- 长推理 decoding 不同于 prefilling，KV 重要性具有 milestone、recurrence、reflection 等时间变化，不能只依赖一次 attention 分数。
- 本地 AIME24 复现显示：R-KV 随 budget 增大稳定提升，R-KV@2048 略高于 FullKV；SnapKV 明显较弱。
- KV 重要性依赖未来 query，单次不可逆淘汰可能误删关键 token，应关注分层压缩、可恢复 rescue 和自我纠错安全边界。
- 正式评测必须报告绝对 budget、cache ratio、平均/最终 cache 长度、精度和端到端吞吐；当前主要瓶颈是 Python 调度、eager attention 和 cache compact。
- TideProbe 诊断必须基于同一条冻结 FullKV 轨迹做 teacher forcing；当前代码和合成测试完成，但真实层敏感度、过渡命中损失和 sparse cliff 仍待 GPU 实验验证。
- TideProbe 应先验证层敏感度和过渡段脆弱性，再实现分层预算、渐进淘汰和 rescue。
- KV 管理重构第一阶段已完成：`runner_token`/`evaluator` 支持注入 execution backend 和 cache manager；当前默认仍是 HuggingFace dense cache，vLLM 接入需要独立实现 paged/block cache manager，不能只替换 attention kernel。
- 每个结构化评测 arm 可通过 `engine: hf|vllm` 选择执行栈；旧字符串 arm 默认 HF。当前 vLLM evaluator 仅支持 FullKV serving baseline，动态 KV eviction 策略必须等 paged/block CacheManager 完成后再接入。
