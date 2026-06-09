from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Tuple


@dataclass(frozen=True)
class DiophantineResult:
    # Reduced ratio BD:CD = p:q
    ratio_bd_cd: Tuple[int, int]
    # BC = (p + q) * k
    bc_coefficient: int
    # Left-hand side coefficients in the linear Diophantine equation: (coeff_AE, coeff_AF)
    left_coeffs: Tuple[int, int]
    # Right-hand side constants: constant_term + coeff_k2 * k^2
    right_constants: Tuple[int, int]
    # Pretty-printed equation for AB, AC
    equation_pretty: str
    # Pretty-printed BC relation
    bc_relation_pretty: str


def angle_bisector_ratio(ab: int, ac: int) -> Tuple[int, int]:
    """
    Given sides AB and AC (positive integers), return the reduced ratio BD:CD = AB:AC = p:q.
    """
    if ab <= 0 or ac <= 0:
        raise ValueError("Side lengths AB and AC must be positive integers.")
    g: int = gcd(ab, ac)
    p: int = ab // g
    q: int = ac // g
    return p, q


def derive_diophantine(ab: int, ac: int) -> DiophantineResult:
    """
    Derive the linear Diophantine relation from:
      - Angle Bisector Theorem: BD:CD = AB:AC = p:q -> let BD = p*k, CD = q*k, so BC = (p+q)*k.
      - Power of a Point (B and C lie on the tangent line at D to the circle ω through A):
            BA * BE = BD^2 - R^2
            CA * CF = CD^2 - R^2
        With E, F the second intersections of lines AB, AC with ω, and AE, AF the chord segments from A,
        we have BE = AB - AE and CF = AC - AF.
      - Subtracting the two power identities eliminates R^2 and yields:
            AB*(AB - AE) - AC*(AC - AF) = (p^2 - q^2) * k^2
        Rearranged to:
            AB*AE - AC*AF = (AB^2 - AC^2) + (q^2 - p^2) * k^2

    For AB=200, AC=225 (so p=8, q=9), this becomes:
        200*AE - 225*AF = -10625 + 17*k^2, and BC = 17*k.
    """
    p, q = angle_bisector_ratio(ab, ac)

    # Coefficients on LHS: AB*AE - AC*AF
    coeff_ae: int = ab
    coeff_af: int = -ac

    # RHS constant term: AB^2 - AC^2
    rhs_const: int = ab * ab - ac * ac
    # RHS k^2 coefficient: q^2 - p^2
    rhs_k2_coeff: int = q * q - p * p

    # BC = (p + q) * k
    bc_coeff: int = p + q

    # Pretty-printed results specialized to AB, AC
    equation_pretty: str = f"{coeff_ae}*AE {('+ ' if coeff_af >= 0 else '- ')}{abs(coeff_af)}*AF = {rhs_const} + {rhs_k2_coeff}*k^2"
    bc_relation_pretty: str = f"BC = {bc_coeff}*k"

    return DiophantineResult(
        ratio_bd_cd=(p, q),
        bc_coefficient=bc_coeff,
        left_coeffs=(coeff_ae, coeff_af),
        right_constants=(rhs_const, rhs_k2_coeff),
        equation_pretty=equation_pretty,
        bc_relation_pretty=bc_relation_pretty,
    )


if __name__ == "__main__":
    # Given AB=200, AC=225
    AB: int = 200
    AC: int = 225
    result: DiophantineResult = derive_diophantine(AB, AC)

    # Display the derivation outcome
    print(f"Angle Bisector ratio BD:CD = {result.ratio_bd_cd[0]}:{result.ratio_bd_cd[1]}")
    print(result.bc_relation_pretty)  # Expect: BC = 17*k
    print("Derived linear Diophantine relation (eliminating R^2):")
    # Expect: 200*AE - 225*AF = -10625 + 17*k^2
    # Note: right_constants = (AB^2 - AC^2, q^2 - p^2) = (-10625, 17)
    print(result.equation_pretty)