"""Dataset loading helpers for KV eviction evaluations."""
from __future__ import annotations

SAMPLE_PROBLEMS = [
    {
        "id": "sample-aime-1",
        "question": (
            "Find the number of ordered pairs of positive integers (a, b) such that "
            "a + b = 1000 and neither a nor b has a zero digit."
        ),
        "answer": "738",
    },
    {
        "id": "sample-math-1",
        "question": "What is the remainder when 7^100 is divided by 13?",
        "answer": "9",
    },
]

_DATASETS = {
    "aime": ("Maxwell-Jia/AIME_2024", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}
_Q_FIELDS = ["problem", "Problem", "question", "Question"]
_A_FIELDS = ["answer", "Answer", "solution", "Solution"]


def _pick(row: dict, fields: list[str]):
    for field in fields:
        if field in row and row[field] is not None:
            return row[field]
    raise KeyError(f"missing fields {fields}; columns={list(row.keys())}")


def load_problems(name: str, n: int) -> list[dict]:
    """Load evaluation problems.

    Supported names: sample, aime, math500, mix.
    """
    if name == "sample":
        return (SAMPLE_PROBLEMS * ((n // len(SAMPLE_PROBLEMS)) + 1))[:n]
    if name == "mix":
        half = (n + 1) // 2
        return load_problems("aime", half) + load_problems("math500", n - half)
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected sample/aime/math500/mix")

    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    repo, split = _DATASETS[name]
    ds = load_dataset(repo, split=split)
    problems = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        problems.append(
            {
                "id": f"{name}-{i}",
                "question": str(_pick(row, _Q_FIELDS)),
                "answer": str(_pick(row, _A_FIELDS)),
            }
        )
    return problems

