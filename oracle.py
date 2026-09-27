"""Exact exhaustive Countdown solver for dataset audits and optional fallback."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache


def find_solution(numbers: list[int], target: int) -> str | None:
    """Return one legal expression, or None if the target is unreachable.

    Each input position is used exactly once. Subset dynamic programming keeps
    one expression per exact rational value, so repeated numbers remain distinct.
    """
    if len(numbers) not in (3, 4):
        raise ValueError("expected three or four numbers")

    @lru_cache(maxsize=None)
    def values(mask: int) -> dict[Fraction, str]:
        if mask.bit_count() == 1:
            index = mask.bit_length() - 1
            return {Fraction(numbers[index]): str(numbers[index])}

        found: dict[Fraction, str] = {}
        left_mask = (mask - 1) & mask
        while left_mask:
            right_mask = mask ^ left_mask
            if left_mask < right_mask:
                for left, left_expr in values(left_mask).items():
                    for right, right_expr in values(right_mask).items():
                        a, b = f"({left_expr} + {right_expr})", f"({left_expr} * {right_expr})"
                        found.setdefault(left + right, a)
                        found.setdefault(left * right, b)
                        found.setdefault(left - right, f"({left_expr} - {right_expr})")
                        found.setdefault(right - left, f"({right_expr} - {left_expr})")
                        if right:
                            found.setdefault(left / right, f"({left_expr} / {right_expr})")
                        if left:
                            found.setdefault(right / left, f"({right_expr} / {left_expr})")
            left_mask = (left_mask - 1) & mask
        return found

    return values((1 << len(numbers)) - 1).get(Fraction(target))
