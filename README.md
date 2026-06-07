# decodeatt / RescueKV

Training-free self-correction protection primitive layered over KV eviction backends.
研究笔记见 iCloud 的 RescueKV/（本仓库不追踪）；实现计划见 04-0607代码实现.md。

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
