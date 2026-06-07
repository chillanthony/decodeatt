"""信号 2 触发器：reflection 步标注。

按 token 解码匹配 reflection 词表；答案翻转检测先留桩。
"""

REFLECTION_WORDS = [
    "wait", "but", "actually", "hold on", "let me recheck", "let me reconsider",
    "on second thought", "重新", "等等", "再想想", "不对",
]

# TODO(step2): def mark_reflection_steps(gen_ids, tokenizer) -> list[int]
# TODO(step2): def detect_answer_flips(trace) -> list[int]  # 先返回 []
