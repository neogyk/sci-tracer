from typing import List, Tuple


def decimal_digits(n: int) -> List[int]:
    return [int(ch) for ch in str(n)]


def f_value(n: int) -> int:
    digits: List[int] = decimal_digits(n)
    base: int = max(digits) + 1
    value: int = 0
    for d in digits:
        value = value * base + d
    return value


def analytic_count_upto_999() -> int:
    # Count numbers in 1..999 containing at least one 9
    without_9: int = (9 ** 3) - 1  # 9 choices per digit (0..8), subtract 000 which is 0
    with_9: int = 999 - without_9  # 271
    single_digits_1_to_8: int = 8
    return with_9 + single_digits_1_to_8  # 271 + 8 = 279


def brute_force_count_upto_999() -> Tuple[int, List[int]]:
    solutions: List[int] = [n for n in range(1, 1000) if f_value(n) == n]
    return len(solutions), solutions


if __name__ == "__main__":
    analytic: int = analytic_count_upto_999()
    brute_count, brute_solutions = brute_force_count_upto_999()
    print(f"Analytic count (271 + 8): {analytic}")
    print(f"Brute-force count: {brute_count}")
    assert analytic == brute_count == 279, "Counts do not match expected value 279."