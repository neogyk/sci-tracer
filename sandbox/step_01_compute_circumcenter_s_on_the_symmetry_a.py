from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple


@dataclass(frozen=True)
class Point3D:
    x: Fraction
    y: Fraction
    z: Fraction


def solve_for_s() -> Fraction:
    """
    Solve 25 + s^2 = (s - 12)^2 + 81 for s using exact rational arithmetic.
    Simplifies to -24s = -200 -> s = 25/3.
    """
    numerator: int = 225 - 25  # from rearranging: 25 = -24s + 225 -> -24s = -200
    denominator: int = 24
    s: Fraction = Fraction(numerator, denominator)  # 200/24 = 25/3
    return s


def circumradius_squared(s: Fraction) -> Fraction:
    """
    Circumradius squared R^2 = |SA|^2 where A = (-5,0,0) and S = (0,s,0).
    |SA|^2 = 25 + s^2
    """
    return Fraction(25, 1) + s * s


def verify_equation(s: Fraction) -> None:
    """
    Verify the original equation 25 + s^2 = (s - 12)^2 + 81 holds exactly.
    """
    lhs: Fraction = Fraction(25, 1) + s * s
    rhs: Fraction = (s - Fraction(12, 1)) ** 2 + Fraction(81, 1)
    assert lhs == rhs, f"Equation verification failed: LHS={lhs}, RHS={rhs}"


def main() -> Tuple[Point3D, Fraction]:
    s: Fraction = solve_for_s()
    verify_equation(s)

    S: Point3D = Point3D(Fraction(0, 1), s, Fraction(0, 1))
    R2: Fraction = circumradius_squared(s)

    # Expected S = (0, 25/3, 0) and R^2 = 850/9
    assert s == Fraction(25, 3), f"Unexpected s: {s}"
    assert R2 == Fraction(850, 9), f"Unexpected circumradius squared: {R2}"

    # Output (could be consumed by other tools)
    print(f"Circumcenter S: (x={S.x}, y={S.y}, z={S.z})")
    print(f"Circumradius squared R^2: {R2} (~ {float(R2):.6f})")
    return S, R2


if __name__ == "__main__":
    main()