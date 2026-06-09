from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Final, Tuple


def compute_distance_and_sum(p: Fraction, T: Fraction) -> Tuple[Fraction, int]:
    """
    Given Patrick's speed p (mph) and travel time T (hours) as Fractions,
    compute the distance D = p * T (in miles) reduced to lowest terms,
    and return (D, m_plus_n) where m_plus_n is the sum of D's numerator and denominator.
    """
    D: Fraction = p * T  # Fraction auto-reduces to lowest terms
    m: int = D.numerator
    n: int = D.denominator
    return D, m + n


def main() -> None:
    # Known values from prior step:
    p: Final[Fraction] = Fraction(18, 5)  # Patrick's speed in mph
    T: Final[Fraction] = Fraction(14, 5)  # Time in hours

    D, m_plus_n = compute_distance_and_sum(p, T)

    # Verify D is in lowest terms
    m: int = D.numerator
    n: int = D.denominator
    assert gcd(m, n) == 1, "Distance fraction is not in lowest terms."

    print(f"D = {m}/{n} miles")
    print(f"m + n = {m_plus_n}")


if __name__ == "__main__":
    main()