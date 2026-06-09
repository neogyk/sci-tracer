from typing import List, Sequence


def decimal_digits(n: int) -> List[int]:
    """
    Return the base-10 digits of n as a list [a_k, ..., a_0] with most significant first.
    Preconditions: n >= 0
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    return [int(ch) for ch in str(n)]


def interpret_in_base(digits: Sequence[int], base: int) -> int:
    """
    Interpret the given sequence of nonnegative digits (most significant first) in the specified base.
    Preconditions:
      - base >= 2
      - all digits d satisfy 0 <= d < base
    """
    if base < 2:
        raise ValueError("Base must be at least 2")
    value: int = 0
    for d in digits:
        if not (0 <= d < base):
            raise ValueError(f"Digit {d} is not valid for base {base}")
        value = value * base + d
    return value


def f_value(n: int) -> int:
    """
    Let m be the maximum decimal digit of n, and set b = m + 1.
    Define f(n) as the value obtained by interpreting the decimal digits of n as a base-b numeral.
    Preconditions: n >= 1 (so b >= 2 is guaranteed)
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    digs: List[int] = decimal_digits(n)
    b: int = max(digs) + 1
    return interpret_in_base(digs, b)


def is_fixed_point_characterization(n: int) -> bool:
    """
    Characterization of when f(n) = n:
      - If n contains a digit 9, then max digit m = 9, so b = 10 and f(n) = n.
      - If n has no digit 9 (so b < 10), then f(n) = n holds iff n is a single digit in {1, ..., 8}.
    Preconditions: n >= 1
    """
    if n <= 0:
        return False
    digs: List[int] = decimal_digits(n)
    if 9 in digs:
        return True
    # No digit 9 -> b < 10. Only single-digit numbers 1..8 satisfy f(n) = n.
    return n < 10 and 1 <= n <= 8


def is_fixed_point_by_computation(n: int) -> bool:
    """
    Directly check whether f(n) = n by computing f(n).
    Preconditions: n >= 1
    """
    return f_value(n) == n


def count_fixed_points(limit: int) -> int:
    """
    Count the integers n in [1, limit] such that f(n) = n, using the characterization.
    Preconditions: limit >= 1
    """
    if limit < 1:
        return 0
    return sum(1 for n in range(1, limit + 1) if is_fixed_point_characterization(n))


if __name__ == "__main__":
    # Verify the characterization matches direct computation on 1..999 and yields 279.
    LIM: int = 999
    count_char: int = sum(1 for n in range(1, LIM + 1) if is_fixed_point_characterization(n))
    count_comp: int = sum(1 for n in range(1, LIM + 1) if is_fixed_point_by_computation(n))
    assert count_char == count_comp == 279

    # Additional sanity checks mentioned in the research plan:
    # - Numbers 1..999 containing at least one digit 9: 271.
    with9: int = sum(1 for n in range(1, LIM + 1) if 9 in decimal_digits(n))
    assert with9 == 271

    # - Single-digit solutions without a 9 are 1..8: 8 of them.
    singles_1_to_8: int = sum(1 for n in range(1, 10) if is_fixed_point_characterization(n))
    assert singles_1_to_8 == 9  # includes 9 itself via the 'contains 9' case

    # Final print (optional): demonstrate the total count.
    print("Total fixed points in 1..999:", count_char)