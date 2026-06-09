from typing import List, Set
import math


def solve_quadratic_integer_roots(a: int, b: int, c: int) -> List[int]:
    """
    Solve a quadratic equation a*x^2 + b*x + c = 0 and return integer roots (if any).
    Raises a ValueError if the discriminant is not a perfect square or roots are non-integer.
    """
    if a == 0:
        raise ValueError("Coefficient 'a' must be nonzero for a quadratic equation.")

    discriminant: int = b * b - 4 * a * c
    if discriminant < 0:
        raise ValueError("Discriminant is negative; no real roots.")

    sqrt_disc: int = math.isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        raise ValueError("Discriminant is not a perfect square; integer roots do not exist.")

    denom: int = 2 * a
    roots: List[int] = []

    for sign in (1, -1):
        numerator: int = -b + sign * sqrt_disc
        if numerator % denom != 0:
            raise ValueError("Quadratic does not have integer roots.")
        roots.append(numerator // denom)

    return roots


if __name__ == "__main__":
    # Solve n^2 - 144 n + 1584 = 0
    candidates: Set[int] = set(solve_quadratic_integer_roots(1, -144, 1584))
    print(f"N_candidates = {sorted(candidates)}")