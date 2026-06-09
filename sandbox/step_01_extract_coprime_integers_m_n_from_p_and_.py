from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Tuple


def reduce_and_compute_target(p: Fraction) -> Tuple[int, int, int]:
    """
    Given a probability p as a Fraction, reduce it to lowest terms to obtain coprime integers (m, n),
    then compute the target value 100*m + n.

    Returns:
        (m, n, target) where target = 100*m + n.
    """
    # Fraction automatically reduces to lowest terms
    p_reduced = Fraction(p.numerator, p.denominator)
    m, n = p_reduced.numerator, p_reduced.denominator

    # Ensure coprime
    assert gcd(m, n) == 1, "Fraction was not in lowest terms."

    target = 100 * m + n
    return m, n, target


def main() -> None:
    # Given exact rational probability P = 7/54 from prior step
    P = Fraction(7, 54)
    m, n, target = reduce_and_compute_target(P)
    print(m, n, target)  # Expected: 7 54 754


if __name__ == "__main__":
    main()