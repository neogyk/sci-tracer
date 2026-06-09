from __future__ import annotations

import math
from fractions import Fraction
from typing import Final


def tau(n: int) -> int:
    """Return the number of positive divisors of n (tau function)."""
    cnt: int = 1
    x: int = n
    p: int = 2
    while p * p <= x:
        if x % p == 0:
            e: int = 0
            while x % p == 0:
                x //= p
                e += 1
            cnt *= (e + 1)
        p = 3 if p == 2 else p + 2
    if x > 1:
        cnt *= 2
    return cnt


def partial_sum_tau_over_10m(N: int, k: int) -> Fraction:
    """Compute S_k = sum_{m=1..k} tau(N+m) / 10^m exactly as a Fraction."""
    s: Fraction = Fraction(0, 1)
    for m in range(1, k + 1):
        s += Fraction(tau(N + m), 10 ** m)
    return s


def tail_bound_sqrt(N: int, start_m: int) -> float:
    """
    Bound T = sum_{m=start_m..inf} tau(N+m)/10^m using tau(n) <= 2*sqrt(n) and
    sqrt(N+m) <= sqrt(N) + sqrt(m).

    Then T <= 2 * [ sum sqrt(m)/10^m + sqrt(N) * sum 1/10^m ] over m >= start_m.
    For S1 = sum sqrt(m)/10^m, use a geometric majorant with ratio
    r = (sqrt(start_m+1)/sqrt(start_m)) / 10, and first term t_sm = sqrt(start_m)/10^{start_m}:
      S1 <= t_sm / (1 - r).
    For S2 = sum 1/10^m, S2 = 10^{-start_m} / (1 - 1/10) = 10^{-start_m} / 0.9.
    """
    sm: int = start_m
    # First term and common-ratio upper bound for S1
    t_sm: float = math.sqrt(sm) / (10 ** sm)
    r: float = (math.sqrt(sm + 1.0) / math.sqrt(sm)) / 10.0
    S1: float = t_sm / (1.0 - r)
    S2: float = (10 ** (-sm)) / (1.0 - 0.1)  # = 10^{-sm} / 0.9
    return 2.0 * (S1 + math.sqrt(N) * S2)


def main() -> None:
    N: Final[int] = 100
    # Exact partial sum for m=1..9
    partial_9: Fraction = partial_sum_tau_over_10m(N, 9)
    # Rigorous tail bound from m=10 to infinity
    tail_bound: float = tail_bound_sqrt(N, 10)
    R100_upper: float = float(partial_9) + tail_bound

    print("Exact partial (m=1..9) as Fraction:", partial_9)
    print("Exact partial (m=1..9) as decimal:", float(partial_9))
    print("Tail bound from m>=10:", tail_bound)
    print("R_100 upper bound:", R100_upper)
    print("Is R_100 < 1?", R100_upper < 1.0)


if __name__ == "__main__":
    main()