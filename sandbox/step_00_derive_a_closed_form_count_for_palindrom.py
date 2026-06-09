from __future__ import annotations

import itertools
import math
from typing import Final, Iterable, Tuple


TARGET_SUM: Final[int] = 13
DIGITS: Final[Tuple[int, ...]] = tuple(range(1, 10))


def count_palindromes_closed_form() -> int:
    """
    Count palindromic numbers (base-10) with no zeros and digit sum 13
    using parity and stars-and-bars. Only odd lengths contribute.
    For length 2n+1 (n >= 1), with first-half sum S and middle digit m:
        2S + m = 13, m in {1,3,5,7,9} => S in {2,3,4,5,6}.
    Number of ordered n-tuples of positive integers summing to S is C(S-1, n-1).
    Total = sum_{n=1..6} sum_{S=max(n,2)..6} C(S-1, n-1).
    """
    total: int = 0
    for n in range(1, 7):  # half-length (excluding middle digit)
        for S in range(max(n, 2), 7):  # S = 2..6 and S >= n
            total += math.comb(S - 1, n - 1)
    return total


def count_palindromes_bruteforce() -> int:
    """
    Empirically verify the analytical count by constructing halves and checking middle digit.
    For each n in 1..6 and each half tuple in {1..9}^n, set m = 13 - 2*sum(half).
    Accept if m is an odd digit in 1..9. Each such (half, m) yields a unique palindrome.
    """
    count: int = 0
    for n in range(1, 7):
        for half in itertools.product(DIGITS, repeat=n):
            S: int = sum(half)
            if S < 2 or S > 6:
                continue
            m: int = TARGET_SUM - 2 * S
            if 1 <= m <= 9 and (m % 2 == 1):
                count += 1
    return count


if __name__ == "__main__":
    closed_form: int = count_palindromes_closed_form()
    brute_force: int = count_palindromes_bruteforce()
    assert closed_form == 62, f"Closed-form mismatch: expected 62, got {closed_form}"
    assert brute_force == 62, f"Brute-force mismatch: expected 62, got {brute_force}"
    print(closed_form)