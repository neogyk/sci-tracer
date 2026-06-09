from __future__ import annotations

from fractions import Fraction
from typing import Final


def solve_cos_theta_from_abprime(ab_prime: Fraction) -> Fraction:
    """
    Given AB' = ab_prime and the relation AB'^2 = 5 - 4 cos(theta),
    solve exactly for cos(theta) as a Fraction.
    """
    return (Fraction(5, 1) - ab_prime * ab_prime) / 4


def main() -> None:
    ab_prime: Final[Fraction] = Fraction(4, 3)
    cos_theta: Fraction = solve_cos_theta_from_abprime(ab_prime)

    # Verify the derived value and the defining relation
    assert cos_theta == Fraction(29, 36)
    lhs: Fraction = ab_prime * ab_prime
    rhs: Fraction = Fraction(5, 1) - 4 * cos_theta
    assert lhs == rhs

    print(f"cos(theta) = {cos_theta}  (~ {float(cos_theta):.6f})")


if __name__ == "__main__":
    main()