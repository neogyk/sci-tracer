from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LinearForm:
    """
    Represents a linear form L(h1, h2) = a1*h1 + a2*h2 + c with integer coefficients.
    """
    a1: int  # coefficient of h1
    a2: int  # coefficient of h2
    c: int   # constant term

    def add(self, other: "LinearForm") -> "LinearForm":
        return LinearForm(self.a1 + other.a1, self.a2 + other.a2, self.c + other.c)

    def sub(self, other: "LinearForm") -> "LinearForm":
        return LinearForm(self.a1 - other.a1, self.a2 - other.a2, self.c - other.c)

    def scale(self, k: int) -> "LinearForm":
        return LinearForm(k * self.a1, k * self.a2, k * self.c)

    def __str__(self) -> str:
        parts = []
        if self.a1 != 0:
            parts.append(f"{self.a1}*h1")
        if self.a2 != 0:
            parts.append(f"{self.a2}*h2")
        if self.c != 0 or not parts:
            parts.append(f"{self.c}")
        return " + ".join(parts).replace("+ -", "- ")


def shoelace_area_linear_form() -> LinearForm:
    """
    Compute the polygon area (without absolute value sign, using orientation)
    via the shoelace formula for A(0,0), B(0,h1), C(14,h1-14), D(20,h2), E(20,0).

    With the chosen orientation and the geometric constraints (h1 >= 15, h2 = h1 - 8 >= 7),
    sum2 - sum1 > 0, so Area = (sum2 - sum1)/2 is valid (no need for absolute value).

    Returns:
        LinearForm representing Area(h1,h2) = 17*h1 + 3*h2 - 140
    """
    # y-values as linear forms
    yA = LinearForm(0, 0, 0)
    yB = LinearForm(1, 0, 0)       # h1
    yC = LinearForm(1, 0, -14)     # h1 - 14
    yD = LinearForm(0, 1, 0)       # h2
    yE = LinearForm(0, 0, 0)       # 0

    # x-values (constants)
    xA, xB, xC, xD, xE = 0, 0, 14, 20, 20

    # sum1 = Σ x_i * y_{i+1}
    sum1 = (
        yB.scale(xA)
        .add(yC.scale(xB))
        .add(yD.scale(xC))
        .add(yE.scale(xD))
        .add(yA.scale(xE))
    )

    # sum2 = Σ y_i * x_{i+1}
    sum2 = (
        yA.scale(xB)
        .add(yB.scale(xC))
        .add(yC.scale(xD))
        .add(yD.scale(xE))
        .add(yE.scale(xA))
    )

    # Area = (sum2 - sum1)/2 (orientation chosen so it's positive on the valid domain)
    area_form = sum2.sub(sum1)
    assert area_form.a1 % 2 == 0 and area_form.a2 % 2 == 0 and area_form.c % 2 == 0, "Coefficients must be even."
    area_form = LinearForm(area_form.a1 // 2, area_form.a2 // 2, area_form.c // 2)

    return area_form  # Expected: 17*h1 + 3*h2 - 140


def substitute_h2_with_h1(area: LinearForm) -> Tuple[int, int]:
    """
    Substitute h2 = h1 - 8 into Area(h1,h2) = a1*h1 + a2*h2 + c.

    Returns:
        (a, c) such that Area(h1) = a*h1 + c
    """
    a = area.a1 + area.a2          # coefficient on h1 after substitution
    c = area.c - 8 * area.a2       # constant term after substitution
    return a, c


def divisibility_condition_for_16(a: int, c: int) -> Tuple[int, int]:
    """
    For Area(h1) = a*h1 + c, derive the condition Area ≡ 0 (mod 16).
    Solve a*h1 + c ≡ 0 (mod 16) -> h1 ≡ r (mod m), reduced to smallest modulus.

    Returns:
        (m, r) meaning h1 ≡ r (mod m)
    """
    mod = 16
    a_mod = a % mod
    c_mod = c % mod

    # We need a_mod * h1 ≡ (-c_mod) (mod 16)
    rhs = (-c_mod) % mod

    # Reduce by gcd if possible
    from math import gcd

    g = gcd(a_mod, mod)
    assert rhs % g == 0, "No solutions exist to the congruence (should not happen here)."
    a_reduced = a_mod // g
    mod_reduced = mod // g
    rhs_reduced = rhs // g

    # Find modular inverse of a_reduced modulo mod_reduced (a_reduced and mod_reduced are coprime)
    inv = pow(a_reduced, -1, mod_reduced)
    r = (inv * rhs_reduced) % mod_reduced

    return mod_reduced, r


def main() -> None:
    # 1) Compute area linear form via shoelace
    area_form = shoelace_area_linear_form()
    print(f"Area(h1,h2) = {area_form}")  # Expect: 17*h1 + 3*h2 - 140

    # 2) Substitute h2 = h1 - 8
    a, c = substitute_h2_with_h1(area_form)
    print(f"After h2 = h1 - 8: Area(h1) = {a}*h1 + {c}")  # Expect: 20*h1 - 164

    # 3) Derive divisibility condition: 16 | Area
    m, r = divisibility_condition_for_16(a, c)
    print(f"Divisibility by 16 condition: h1 ≡ {r} (mod {m})")  # Expect: h1 ≡ 1 (mod 4)

    # Optional quick sanity check with a sample h1 that satisfies the congruence
    sample_h1: int = 17  # 17 ≡ 1 (mod 4)
    area_val: int = a * sample_h1 + c
    print(f"Sample check: h1 = {sample_h1} -> Area = {area_val}, Area % 16 = {area_val % 16}")


if __name__ == "__main__":
    main()