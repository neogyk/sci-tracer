from __future__ import annotations
from typing import Iterable, Final


def a10(d: int) -> int:
    """Return the 10th term a_10 = 4 + 9*d for an arithmetic sequence starting at 4.

    Args:
        d: Common difference (integer).
    Returns:
        The 10th term value as an integer.
    """
    if not isinstance(d, int):
        raise TypeError("d must be an int")
    return 4 + 9 * d


def sum_of_tenths(ds: Iterable[int]) -> int:
    """Sum the 10th terms for a collection of valid common differences.

    Args:
        ds: Iterable of integer common differences.
    Returns:
        Sum of the corresponding 10th terms.
    """
    total: int = 0
    for d in ds:
        total += a10(int(d))
    return total


if __name__ == "__main__":
    D: Final[tuple[int, ...]] = (1, 2, 5, 10)
    S: int = sum_of_tenths(D)
    print(S)