from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class Qsqrt5:
    """
    Represents numbers of the form a + b*sqrt(5) with exact rational coefficients.
    """
    a: Fraction
    b: Fraction

    def __add__(self, other: "Qsqrt5") -> "Qsqrt5":
        return Qsqrt5(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "Qsqrt5") -> "Qsqrt5":
        return Qsqrt5(self.a - other.a, self.b - other.b)

    def __mul__(self, other: "Qsqrt5") -> "Qsqrt5":
        # (a1 + b1√5)(a2 + b2√5) = (a1a2 + 5 b1b2) + (a1b2 + a2b1)√5
        a = self.a * other.a + Fraction(5) * self.b * other.b
        b = self.a * other.b + self.b * other.a
        return Qsqrt5(a, b)

    def conj(self) -> "Qsqrt5":
        return Qsqrt5(self.a, -self.b)

    def norm(self) -> Fraction:
        # (a + b√5)(a - b√5) = a^2 - 5 b^2
        return self.a * self.a - Fraction(5) * self.b * self.b

    def inv(self) -> "Qsqrt5":
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot invert zero in Q(√5).")
        return Qsqrt5(self.a / n, -self.b / n)

    def __truediv__(self, other: "Qsqrt5") -> "Qsqrt5":
        return self * other.inv()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Qsqrt5):
            return False
        return self.a == other.a and self.b == other.b

    def __str__(self) -> str:
        return format_qsqrt5(self)


def q(a: int | Fraction, b: int | Fraction = 0) -> Qsqrt5:
    return Qsqrt5(Fraction(a), Fraction(b))


def format_fraction(fr: Fraction) -> str:
    if fr.denominator == 1:
        return f"{fr.numerator}"
    return f"{fr.numerator}/{fr.denominator}"


def format_qsqrt5(x: Qsqrt5) -> str:
    a_str = format_fraction(x.a)
    b_str = format_fraction(abs(x.b))
    if x.b == 0:
        return a_str
    if x.a == 0:
        sign = "-" if x.b < 0 else ""
        return f"{sign}{b_str}√5"
    sign = " - " if x.b < 0 else " + "
    return f"{a_str}{sign}{b_str}√5"


def compute_S1() -> Qsqrt5:
    # Given data
    S0: Fraction = Fraction(308)
    sum_diag_sq: Fraction = Fraction(800)

    # cos 72° = (√5 − 1)/4
    c1: Qsqrt5 = q(-1, 0) / q(4, 0) + q(0, 1) / q(4, 0)  # (-1)/4 + (1/4)√5
    # Or directly: c1 = q(Fraction(-1, 4), Fraction(1, 4))

    # Identity: Σ|e_i + e_{i+1}|^2 = 2*S0 + 2*c1*S1
    # Given Σ diagonal squares = 800, solve for S1:
    # 800 = 2*308 + 2*c1*S1  =>  c1*S1 = 92  =>  S1 = 92 / c1
    rhs: Fraction = (sum_diag_sq - 2 * S0) / 2  # 92
    S1: Qsqrt5 = q(rhs, 0) / c1

    # Sanity check: c1 * S1 == 92
    assert c1 * S1 == q(rhs, 0), "Consistency check failed for c1 * S1 = 92"

    return S1


if __name__ == "__main__":
    S1_val = compute_S1()
    print("S1 =", S1_val)
    # Optional: show factored form when a == b
    if S1_val.a == S1_val.b:
        k = S1_val.a
        print(f"S1 = {format_fraction(k)}(√5 + 1)")