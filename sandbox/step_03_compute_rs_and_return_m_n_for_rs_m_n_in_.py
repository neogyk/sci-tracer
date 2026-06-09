from __future__ import annotations
from fractions import Fraction
from typing import Final


def compute_m_plus_n() -> int:
    """
    Compute m + n for RS where S_y = 25/3 and R_y = 75/16.
    RS = |S_y - R_y| = 175/48 in lowest terms, so return 175 + 48 = 223.
    """
    s_y: Final[Fraction] = Fraction(25, 3)
    r_y: Final[Fraction] = Fraction(75, 16)
    rs: Fraction = abs(s_y - r_y)
    # Fraction simplifies to lowest terms by construction
    assert rs == Fraction(175, 48)
    m_plus_n: int = rs.numerator + rs.denominator
    return m_plus_n


if __name__ == "__main__":
    result: int = compute_m_plus_n()
    print(result)