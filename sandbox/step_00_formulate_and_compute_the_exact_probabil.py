from __future__ import annotations
from fractions import Fraction
from typing import Final


def compute_probability() -> Fraction:
    """Compute exact P that A and B each reach 2 before first C.

    Uses the decomposition:
      P = sum_{n=4}^∞ (1/3)(2/3)^n [1 - 2(1+n)/2^n]
        = S1 - S2, where
      S1 = (1/3) * sum_{n=4}^∞ (2/3)^n
      S2 = (2/3) * sum_{n=4}^∞ (n+1) * (1/3)^n
    """
    two_thirds: Final[Fraction] = Fraction(2, 3)
    one_third: Final[Fraction] = Fraction(1, 3)

    # S1 = (1/3) * ((2/3)^4) / (1 - 2/3)
    S1: Fraction = one_third * (two_thirds ** 4) / (Fraction(1, 1) - two_thirds)

    # Tail of sum_{n=0}^∞ (n+1) r^n = 1/(1-r)^2, with r = 1/3, from n=4 to ∞.
    r: Final[Fraction] = one_third
    total_series: Fraction = Fraction(1, 1) / (Fraction(1, 1) - r) ** 2
    partial_sum_up_to_3: Fraction = sum(Fraction(n + 1, 1) * (r ** n) for n in range(4))
    tail_from_4: Fraction = total_series - partial_sum_up_to_3

    # S2 = (2/3) * tail_from_4
    S2: Fraction = two_thirds * tail_from_4

    P: Fraction = S1 - S2
    return P


if __name__ == "__main__":
    P_exact: Fraction = compute_probability()
    print("Exact probability P =", P_exact)           # Expected 7/54
    print("Decimal approximation =", float(P_exact))
    # Sanity check against the known closed-form value
    assert P_exact == Fraction(7, 54)