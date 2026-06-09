from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TriangleSolution:
    # Sides: AB = BC = x, AC = y; incenter distances: AI = CI = t
    x: int
    y: int
    t: int
    # Parameters from reduction
    m: int
    d: int


def diophantine_holds(m: int, d: int) -> bool:
    """
    Check the reduced Diophantine constraint:
      125 m (3m + d)^2 = 2 (3m - d)^2 (122m + d)
    """
    if m <= 0 or d <= 0:
        return False
    lhs: int = 125 * m * (3 * m + d) ** 2
    rhs: int = 2 * (3 * m - d) ** 2 * (122 * m + d)
    return lhs == rhs


def candidate_from(m: int, d: int) -> TriangleSolution:
    """
    Given parameters (m, d), construct the candidate:
      x = 122m + d, y = 6m - 2d, t = 3m + d
    """
    x: int = 122 * m + d
    y: int = 6 * m - 2 * d
    t: int = 3 * m + d
    return TriangleSolution(x=x, y=y, t=t, m=m, d=d)


def verify_solution(sol: TriangleSolution) -> bool:
    """
    Verify all constraints:
      - Positivity
      - Triangle inequalities for isosceles triangle (y < 2x)
      - Additional bound y < 2t
      - Perimeter ratio (2x + y) / (y + 2t) = 125 / 6
      - Incenter relation t^2 = y^2 * x / (2x + y)  (checked by cross-multiplication)
      - Derived structural relation s = 2x + y = 250 m
    """
    x, y, t, m = sol.x, sol.y, sol.t, sol.m

    if not (x > 0 and y > 0 and t > 0):
        return False

    # Triangle inequalities for sides x, x, y: need y < 2x (and trivially y > |x-x| = 0 already)
    if not (y < 2 * x):
        return False

    # Additional condition specified in the plan
    if not (y < 2 * t):
        return False

    # Perimeter ratio exact check: (2x + y) / (y + 2t) = 125 / 6
    if (2 * x + y) * 6 != (y + 2 * t) * 125:
        return False

    # Incenter relation exact check: t^2 = y^2 * x / (2x + y)
    if t * t * (2 * x + y) != y * y * x:
        return False

    # Structural relation from the reduction: s = 2x + y = 250 m
    if 2 * x + y != 250 * m:
        return False

    return True


def search_minimal_ab(limit_m: int = 1000) -> Optional[TriangleSolution]:
    """
    Enumerate m >= 1, and for each m enumerate 1 <= d < 3m,
    test the Diophantine condition. On the first match, construct
    (x, y, t) and verify all constraints. Return the first valid solution,
    which is minimal in x due to enumeration order (x = 122m + d).
    """
    for m in range(1, limit_m + 1):
        for d in range(1, 3 * m):
            if not diophantine_holds(m, d):
                continue
            sol: TriangleSolution = candidate_from(m, d)
            if verify_solution(sol):
                return sol
    return None


def main() -> None:
    sol: Optional[TriangleSolution] = search_minimal_ab()
    if sol is None:
        raise RuntimeError("No valid solution found within the search bound.")
    # Output the minimal AB = x
    print(sol.x)


if __name__ == "__main__":
    main()