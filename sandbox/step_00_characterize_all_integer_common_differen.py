from __future__ import annotations

from functools import reduce
from math import gcd, isqrt
from typing import Iterable, List, Sequence


def gcd_of_list(values: Iterable[int]) -> int:
    """
    Compute the greatest common divisor (GCD) of a non-empty iterable of integers.
    """
    it = iter(values)
    first = next(it)
    return reduce(gcd, it, abs(first))


def positive_divisors(n: int) -> List[int]:
    """
    Return all positive divisors of a positive integer n, sorted increasingly.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    small: List[int] = []
    large: List[int] = []
    limit: int = isqrt(n)
    for d in range(1, limit + 1):
        if n % d == 0:
            small.append(d)
            q = n // d
            if q != d:
                large.append(q)
    return small + large[::-1]


def valid_positive_differences(a1: int, targets: Sequence[int]) -> List[int]:
    """
    For an arithmetic sequence a_n = a1 + (n-1)d with integer d and n >= 1,
    return all positive integer d such that every value in `targets` is a term.
    This occurs iff for each t in targets, (t - a1) is divisible by d, i.e.,
    d divides gcd(|t - a1| for t in targets).
    """
    if not targets:
        return []
    diffs: List[int] = [abs(t - a1) for t in targets]
    g: int = gcd_of_list(diffs)
    if g == 0:
        # All targets equal to a1; any positive d would work. Conventionally return empty (no restriction).
        return []
    return positive_divisors(g)


if __name__ == "__main__":
    a1: int = 4
    targets: List[int] = [24, 34]
    D: List[int] = valid_positive_differences(a1, targets)
    print(D)  # Expected: [1, 2, 5, 10]