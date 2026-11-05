"""Answer extraction and grading on common MATH-500 / AIME answer formats."""
from __future__ import annotations

import pytest

from kvbench.metrics import extract_answer, is_correct

pytest.importorskip("math_verify")


@pytest.mark.parametrize("gold, text", [
    (r"\frac{14}{3}", r"so the answer is \boxed{\dfrac{14}{3}}"),
    (r"\frac{14}{3}", r"\boxed{14/3}"),
    (r"\sqrt{2}", r"\boxed{\sqrt 2}"),
    (r"0.5", r"\boxed{\frac12}"),
    (r"\text{(C)}", r"\boxed{C}"),
    (r"\$18.90", r"\boxed{18.90}"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3, \frac{\pi}{2})}"),
    (r"\{1,2\}", r"\boxed{\{2,1\}}"),
    (r"x \in [2,5)", r"\boxed{[2,5)}"),
    (r"3\sqrt{3}", r"\boxed{\sqrt{27}}"),
    (r"10\%", r"\boxed{10}"),
    (r"90^\circ", r"\boxed{90}"),
    (r"x^2+7x+10", r"\boxed{x^2 + 7x + 10}"),
    (r"-2, 1", r"\boxed{1, -2}"),
    (r"3", r"\boxed{3.0}"),
    ("738", r"Therefore \boxed{738}."),
    ("9", "The answer is 9."),
])
def test_equivalent_answers_are_accepted(gold, text):
    assert is_correct(extract_answer(text), gold)


@pytest.mark.parametrize("gold, text", [
    ("5", r"\boxed{6}"),
    ("2", r"\boxed{-2}"),
    (r"\frac{1}{2}", r"\boxed{\frac{1}{3}}"),
    (r"\pi", r"\boxed{3.14159}"),
    (r"\text{(C)}", r"\boxed{D}"),
    (r"\sqrt{2}", r"\boxed{2}"),
    ("738", "no final answer here"),
])
def test_wrong_answers_are_rejected(gold, text):
    assert not is_correct(extract_answer(text), gold)
