"""Step ①：用 R1-Distill-Qwen-7B 生成 CoT trace 并落盘。

落盘：prompt_ids / gen_ids / 每步熵 / 题目 / 标准答案。
用法：uv run python scripts/gen_traces.py --config configs/default.yaml --n 1
"""
# TODO(step1): load cfg -> load_model -> generate(output_scores=True) -> 算熵 -> 落盘
