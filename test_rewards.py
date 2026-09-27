from fractions import Fraction

import pytest

from rewards import answer_reward, direct_answer_reward, direct_format_reward, evaluate_expression, final_answer_expression, format_reward, proximity_reward, trace_reward


def test_exact_fraction_arithmetic() -> None:
    assert evaluate_expression("(3 / 2) * 4", [3, 2, 4]) == Fraction(6)


def test_duplicate_numbers_are_counted() -> None:
    assert evaluate_expression("3 * 3 + 1", [3, 3, 1]) == 10
    with pytest.raises(ValueError):
        evaluate_expression("3 * 1 + 1", [3, 3, 1])


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo nope')",
        "2 ** 3",
        "-2 + 3",
        "2.0 + 3",
        "target",
    ],
)
def test_disallowed_syntax(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate_expression(expression, [2, 3])


def test_reward_requires_target_and_all_numbers() -> None:
    completions = [
        "multiply, then add</think><answer>3 * 3 + 1</answer>",
        "just echo it</think><answer>10</answer>",
        "wrong</think><answer>3 + 3 + 1</answer>",
    ]
    assert answer_reward(
        completions,
        nums=[[3, 3, 1]] * 3,
        target=[10] * 3,
    ) == [1.0, 0.0, 0.0]


def test_truncated_reasoning_can_supply_bootstrap_reward() -> None:
    completion = "Still searching. Try 69 + (22 - 29) = 62! This works, so"
    assert answer_reward([completion], nums=[[22, 29, 69]], target=[62]) == [0.0]
    assert trace_reward([completion], nums=[[22, 29, 69]], target=[62]) == [0.1]


def test_claimed_target_does_not_bypass_expression_check() -> None:
    completion = "A bad calculation says 69 + 22 + 29 = 62, but it is false."
    assert answer_reward([completion], nums=[[22, 29, 69]], target=[62]) == [0.0]
    assert trace_reward([completion], nums=[[22, 29, 69]], target=[62]) == [0.0]


def test_proximity_reward_requires_legal_all_number_expression() -> None:
    completions = [
        "Try 1 + 2 + 3 + 4 = 10",
        "Try 1 + 2 + 3 * 4 = 15",
        "Try 1 + 2 + 3 = 6",
    ]
    scores = proximity_reward(completions, nums=[[1, 2, 3, 4]] * 3, target=[10] * 3)
    assert scores[0] == pytest.approx(0.15)
    assert 0 < scores[1] < scores[0]
    assert scores[2] == 0


def test_conversational_completion_shape() -> None:
    completion = [
        {"role": "assistant", "content": "x</think><answer>8 - 3</answer>"}
    ]
    assert final_answer_expression(completion) == "8 - 3"
    assert format_reward([completion]) == [0.2]


def test_final_answer_accepts_terminal_eos_but_rejects_extra_text() -> None:
    good = "69 + (22 - 29) = 62</think><answer>69 + (22 - 29)</answer><|im_end|>\n<|endoftext|><|endoftext|>"
    bad = "69 + (22 - 29) = 62</think><answer>69 + (22 - 29)</answer> extra text"
    assert answer_reward([good, bad], nums=[[22, 29, 69]] * 2, target=[62] * 2) == [1.0, 0.0]


def test_direct_answer_requires_only_tagged_expression() -> None:
    completions = [
        "<answer>82 + 25 - 8</answer><|im_end|>",
        "Thinking... <answer>82 + 25 - 8</answer>",
        "<answer>99</answer>",
    ]
    assert direct_answer_reward(
        completions, nums=[[82, 8, 25]] * 3, target=[99] * 3
    ) == [1.0, 0.0, 0.0]
    assert direct_format_reward(completions) == [0.2, 0.0, 0.2]
