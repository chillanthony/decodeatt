"""信号 2 触发器：reflection 步标注。

- mark_reflection_steps：按 reflection 词表在生成 token 序列上定位"自我纠正"步。
- detect_answer_flips：基于 \\boxed{} 的"中途答案 vs 最终答案不一致"轻量检测（v1）。

step 下标 = gen_ids 中的位置（第 i 个解码步产出第 i 个 token）。
字符位置 -> step 用"累积解码"的偏移表映射，兼容 CJK 的字节回退 token。
"""
from __future__ import annotations

import bisect
import re

# reflection / 回退标志词（收紧版）：只留强信号短语，去掉满天飞的单字 but/wait/hmm
# 避免 R_t 退化成"几乎所有早期 page"。
REFLECTION_PATTERNS = [
    r"\bwait,", r"\bbut wait\b", r"\bhold on\b",
    r"\blet me (?:re)?check\b", r"\blet me reconsider\b",
    r"\blet me re-?examine\b", r"\blet me try again\b",
    r"\bon second thought\b", r"\bthat'?s (?:wrong|not right)\b",
    r"\bi made a mistake\b", r"\bactually,? wait\b",
    "重新", "再想想", "不对", "等一下", "搞错",
]
_REFLECTION_RE = re.compile("|".join(REFLECTION_PATTERNS), re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _to_list(gen_ids):
    if hasattr(gen_ids, "tolist"):
        return gen_ids.tolist()
    return list(gen_ids)


def _text_and_offsets(gen_ids, tokenizer):
    """返回 (full_text, offsets, ids)。

    offsets[i] = token i 文本在 full_text 中的起始字符位置。
    用累积解码保证 CJK/字节回退 token 也能正确对齐。
    """
    ids = _to_list(gen_ids)
    offsets = []
    prev_len = 0
    for i in range(len(ids)):
        offsets.append(prev_len)
        prev_len = len(tokenizer.decode(ids[: i + 1]))
    full = tokenizer.decode(ids) if ids else ""
    return full, offsets, ids


def _char_to_step(char_pos: int, offsets: list[int]) -> int:
    i = bisect.bisect_right(offsets, char_pos) - 1
    return max(0, i)


def mark_reflection_steps(
    gen_ids, tokenizer, step_entropy=None, entropy_quantile: float | None = None
) -> list[int]:
    """返回命中 reflection 词的解码步下标（升序去重）。

    若给 step_entropy + entropy_quantile，则再叠加"高熵步"过滤：只保留熵 >= 分位阈值的步
    （纠错往往发生在高不确定的决策点，进一步收紧 R_t）。
    """
    text, offsets, _ = _text_and_offsets(gen_ids, tokenizer)
    steps = {_char_to_step(m.start(), offsets) for m in _REFLECTION_RE.finditer(text)}

    if step_entropy is not None and entropy_quantile is not None:
        import torch

        e = step_entropy if isinstance(step_entropy, torch.Tensor) else torch.tensor(step_entropy)
        thr = torch.quantile(e.float(), entropy_quantile).item()
        steps = {s for s in steps if s < e.numel() and float(e[s]) >= thr}

    return sorted(steps)


def detect_answer_flips(gen_ids, tokenizer) -> list[int]:
    """v1：若出现 >=2 个 \\boxed{}，把与最终答案不一致的中途 boxed 步标为翻转点。

    (更强的"中途结论 vs 最终答案"检测留待 Step 2 升级。)
    """
    text, offsets, _ = _text_and_offsets(gen_ids, tokenizer)
    boxed = [(m.start(), m.group(1).strip()) for m in _BOXED_RE.finditer(text)]
    if len(boxed) < 2:
        return []
    final = boxed[-1][1]
    flips = {_char_to_step(pos, offsets) for pos, val in boxed[:-1] if val != final}
    return sorted(flips)


def _selftest_trace(path: str):
    """自测入口：在一条 trace .pt 上跑并打印命中步与上下文。"""
    import torch
    from transformers import AutoTokenizer

    trace = torch.load(path, weights_only=False)
    tok = AutoTokenizer.from_pretrained(trace["model"])
    refl = mark_reflection_steps(trace["gen_ids"], tok)
    flips = detect_answer_flips(trace["gen_ids"], tok)
    ids = _to_list(trace["gen_ids"])
    print(f"[trace] {path}  gen_len={len(ids)}")
    print(f"reflection steps ({len(refl)}): {refl}")
    print(f"answer-flip steps ({len(flips)}): {flips}")
    for s in refl[:10]:
        ctx = tok.decode(ids[max(0, s - 8): s + 8])
        print(f"  step {s}: ...{ctx!r}...")


if __name__ == "__main__":
    import sys

    _selftest_trace(sys.argv[1])
