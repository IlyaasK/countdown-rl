from oracle import find_solution
from rewards import evaluate_expression


def test_oracle_finds_three_and_four_number_solutions() -> None:
    for numbers, target in [([82, 8, 25], 99), ([15, 66, 98, 49], 100), ([3, 3, 1], 10)]:
        expression = find_solution(numbers, target)
        assert expression is not None
        assert evaluate_expression(expression, numbers) == target


def test_oracle_reports_unsolvable_task() -> None:
    assert find_solution([1, 1, 1], 100) is None
