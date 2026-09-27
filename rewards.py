"""Safe, exact rewards for the Countdown arithmetic task."""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction
from typing import Any


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_EOS_RE = re.compile(r"(?:(?:<\|im_end\|>|<\|endoftext\|>)\s*)+\Z")
_EQUATION_RE = re.compile(
    r"(?P<expression>\d[\d\s()+\-*/]{0,255})\s*=\s*(?P<result>-?\d+(?:\s*/\s*\d+)?)"
)
_FORMAT_RE = re.compile(r"\A.*?</think>\s*<answer>\s*.+?\s*</answer>\s*\Z", re.DOTALL)
_DIRECT_FORMAT_RE = re.compile(r"\A<answer>\s*(.*?)\s*</answer>\Z", re.DOTALL)
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def completion_text(completion: Any) -> str:
    """Normalize the plain and conversational completion shapes used by TRL."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        return completion_text(completion[-1])
    return ""


def visible_text(completion: Any) -> str:
    """Remove terminal Qwen EOS markers before checking the response format."""
    return _EOS_RE.sub("", completion_text(completion)).strip()


def extract_answer(completion: Any) -> str | None:
    """Extract exactly one answer element from a model completion."""
    matches = _ANSWER_RE.findall(completion_text(completion))
    if len(matches) != 1:
        return None
    expression = matches[0].strip()
    return expression if expression else None


def final_answer_expression(completion: Any) -> str | None:
    """Return the sole answer only if it follows a closed thinking block."""
    text = visible_text(completion)
    if not _FORMAT_RE.fullmatch(text):
        return None
    matches = _ANSWER_RE.findall(text)
    if len(matches) != 1:
        return None
    return matches[0].strip() or None


def direct_answer_expression(completion: Any) -> str | None:
    """Return an answer only when it is the entire visible non-thinking reply."""
    match = _DIRECT_FORMAT_RE.fullmatch(visible_text(completion))
    if match is None:
        return None
    return match.group(1).strip() or None


def candidate_expressions(completion: Any) -> list[str]:
    """Find tagged answers and arithmetic equations in a reasoning trace.

    Base models often discover a valid equation before learning to close their
    thinking block. Rewarding that discovery supplies a sparse bootstrap signal;
    each candidate still goes through the exact AST and operand verifier.
    """
    candidates: list[str] = []
    answer = extract_answer(completion)
    if answer is not None:
        candidates.append(answer)

    for line in completion_text(completion).splitlines():
        for match in _EQUATION_RE.finditer(line):
            # Evaluate every left-hand side ourselves. In a chain such as
            # ``69 + (22 - 29) = 69 - 7 = 62``, the first written RHS is an
            # intermediate expression even though the first LHS is the answer.
            candidates.append(match.group("expression").strip())
    return candidates


def evaluate_expression(expression: str, numbers: list[int]) -> Fraction:
    """Evaluate an arithmetic expression exactly and verify its input numbers.

    This intentionally does not call ``eval``. Only integer literals and the four
    basic binary operators are accepted. The literals must be the same multiset
    as ``numbers``, preventing rewards for simply printing the target.
    """
    if len(expression) > 256:
        raise ValueError("expression is too long")

    tree = ast.parse(expression, mode="eval")
    used_numbers: list[int] = []

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ValueError("only integer literals are allowed")
            used_numbers.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    value = visit(tree)
    if Counter(used_numbers) != Counter(int(number) for number in numbers):
        raise ValueError("the expression must use every supplied number exactly once")
    return value


def answer_reward(
    completions: list[Any], nums: list[list[int]], target: list[int], **_: Any
) -> list[float]:
    """Return 1 only for a valid, usable final answer that reaches the target."""
    rewards: list[float] = []
    for completion, row_numbers, row_target in zip(
        completions, nums, target, strict=True
    ):
        expression = final_answer_expression(completion)
        try:
            solved = expression is not None and evaluate_expression(expression, list(row_numbers)) == int(row_target)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
            solved = False
        rewards.append(float(solved))
    return rewards


def direct_answer_reward(
    completions: list[Any], nums: list[list[int]], target: list[int], **_: Any
) -> list[float]:
    """Strict exact reward for a sole tagged answer without a thinking block."""
    rewards: list[float] = []
    for completion, row_numbers, row_target in zip(completions, nums, target, strict=True):
        expression = direct_answer_expression(completion)
        try:
            solved = expression is not None and evaluate_expression(expression, list(row_numbers)) == int(row_target)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
            solved = False
        rewards.append(float(solved))
    return rewards


def trace_reward(
    completions: list[Any], nums: list[list[int]], target: list[int], **_: Any
) -> list[float]:
    """Small bootstrap signal for a correct equation in the reasoning trace."""
    rewards: list[float] = []
    for completion, row_numbers, row_target in zip(completions, nums, target, strict=True):
        solved = False
        for expression in candidate_expressions(completion):
            try:
                solved = evaluate_expression(expression, list(row_numbers)) == int(row_target)
            except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
                continue
            if solved:
                break
        rewards.append(0.1 if solved else 0.0)
    return rewards


def proximity_reward(
    completions: list[Any], nums: list[list[int]], target: list[int], **_: Any
) -> list[float]:
    """Small dense signal for legal all-number expressions near the target.

    This never substitutes for the exact final-answer reward. It is intended
    for harder four-number tasks where exact successes are initially rare.
    """
    rewards: list[float] = []
    for completion, row_numbers, row_target in zip(completions, nums, target, strict=True):
        distances: list[Fraction] = []
        for expression in candidate_expressions(completion):
            try:
                distances.append(abs(evaluate_expression(expression, list(row_numbers)) - int(row_target)))
            except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
                continue
        rewards.append(0.15 / (1.0 + float(min(distances)) / 10.0) if distances else 0.0)
    return rewards


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Reward a closed Qwen thinking block and one tagged final expression."""
    return [0.2 if final_answer_expression(item) is not None else 0.0 for item in completions]


def direct_format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Reward one answer tag with no visible thinking or extra text."""
    return [0.2 if direct_answer_expression(item) is not None else 0.0 for item in completions]
