from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple


class VerificationResult(NamedTuple):
    p: Fraction
    T: Fraction
    D: Fraction
    D_tanya: Fraction
    D_jose: Fraction
    matches_tanya: bool
    matches_jose: bool
    m: int
    n: int
    m_plus_n: int


def verify_consistency() -> VerificationResult:
    # Given values from prior computation
    p: Fraction = Fraction(18, 5)   # Patrick's speed (mph)
    T: Fraction = Fraction(14, 5)   # Time from Patrick's start to arrival (hours)
    D: Fraction = Fraction(252, 25) # Distance to school (miles)

    # Cross-check using Tanya's and Jose's expressions
    D_tanya: Fraction = (p + Fraction(2, 1)) * (T - Fraction(1, 1))
    D_jose: Fraction = (p + Fraction(9, 1)) * (T - Fraction(2, 1))

    matches_tanya: bool = D_tanya == D
    matches_jose: bool = D_jose == D

    # Distance in lowest terms and m + n
    m: int = D.numerator
    n: int = D.denominator
    m_plus_n: int = m + n

    # Ensure internal consistency
    assert matches_tanya and matches_jose, "Inconsistency detected among distances."

    return VerificationResult(
        p=p,
        T=T,
        D=D,
        D_tanya=D_tanya,
        D_jose=D_jose,
        matches_tanya=matches_tanya,
        matches_jose=matches_jose,
        m=m,
        n=n,
        m_plus_n=m_plus_n,
    )


def main() -> None:
    result: VerificationResult = verify_consistency()
    print(f"Patrick's speed p = {result.p} mph")
    print(f"Common arrival time T = {result.T} hours")
    print(f"Distance D = {result.D} miles")
    print(f"Tanya's computed distance = {result.D_tanya} (matches D: {result.matches_tanya})")
    print(f"Jose's computed distance = {result.D_jose} (matches D: {result.matches_jose})")
    print(f"m + n = {result.m_plus_n}")


if __name__ == "__main__":
    main()