from __future__ import annotations

from fractions import Fraction
from typing import Tuple


def p_from_eq1(T: Fraction) -> Fraction:
    """
    From pT = (p + 2)(T - 1) => p = 2T - 2
    """
    return Fraction(2, 1) * T - Fraction(2, 1)


def p_from_eq2(T: Fraction) -> Fraction:
    """
    From pT = (p + 9)(T - 2) => 2p = 9T - 18 => p = (9/2)T - 9
    """
    return Fraction(9, 2) * T - Fraction(9, 1)


def solve_T_p() -> Tuple[Fraction, Fraction]:
    """
    Solve the simultaneous linear equations:
      p = 2T - 2
      p = (9/2)T - 9
    to find T and p as exact rational numbers.
    """
    lhs_coeff: Fraction = Fraction(2, 1) - Fraction(9, 2)  # (2 - 9/2) = -5/2
    rhs: Fraction = Fraction(-7, 1)  # -9 + 2 = -7
    T: Fraction = rhs / lhs_coeff  # (-7)/(-5/2) = 14/5
    p: Fraction = p_from_eq1(T)  # 2T - 2 = 18/5

    # Internal consistency checks
    assert p == p_from_eq2(T)
    # Optional: verify equal distances
    D1: Fraction = p * T
    D2: Fraction = (p + Fraction(2, 1)) * (T - Fraction(1, 1))
    D3: Fraction = (p + Fraction(9, 1)) * (T - Fraction(2, 1))
    assert D1 == D2 == D3

    return T, p


if __name__ == "__main__":
    T_sol, p_sol = solve_T_p()
    print(f"T = {T_sol} hours, p = {p_sol} mph")