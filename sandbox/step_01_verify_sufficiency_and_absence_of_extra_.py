from __future__ import annotations

import cmath
import itertools
from typing import Iterable, List, Tuple


def cbrt(z: complex) -> complex:
    """Principal complex cube root with real-handling for stability."""
    # Handle (near-)real inputs to ensure real cube roots remain real
    if abs(z.imag) < 1e-18:
        x: float = z.real
        if x >= 0:
            return complex(x ** (1.0 / 3.0), 0.0)
        else:
            return complex(-((-x) ** (1.0 / 3.0)), 0.0)
    # Principal branch for complex values
    return cmath.exp(cmath.log(z) / 3.0)


def depressed_cubic_roots(p: float, q: float) -> Tuple[complex, complex, complex]:
    """
    Solve t^3 + p t + q = 0 via Cardano's method, returning all three (possibly complex) roots.
    """
    delta_c: complex = (q / 2.0) ** 2 + (p / 3.0) ** 3
    sqrt_delta: complex = cmath.sqrt(delta_c)
    u: complex = cbrt(-q / 2.0 + sqrt_delta)
    v: complex = cbrt(-q / 2.0 - sqrt_delta)
    # Cube roots of unity
    omega: complex = complex(-0.5, cmath.sqrt(3) / 2.0)  # exp(2πi/3)
    omega2: complex = omega * omega
    # For depressed cubic, roots are u*ω^k + v*ω^{-k} for k=0,1,2
    r0: complex = u + v
    r1: complex = u * omega + v * omega2
    r2: complex = u * omega2 + v * omega
    return (r0, r1, r2)


def unique_with_tol(values: Iterable[complex], tol: float = 1e-10) -> List[complex]:
    """Deduplicate complex numbers under absolute tolerance tol."""
    uniq: List[complex] = []
    for val in values:
        if not any(abs(val - u) <= tol for u in uniq):
            uniq.append(val)
    return uniq


def verify_n_equals_132() -> None:
    # Given n = 132, we have s2 = -n/12 = -11 and s3 = 20.
    # Construct a, b, c as roots of t^3 + s2 t - s3 = 0, i.e., t^3 - 11 t - 20 = 0.
    p: float = -11.0
    q: float = -20.0

    # Discriminant of depressed cubic: Δ = -4 p^3 - 27 q^2
    discriminant: float = -4.0 * (p ** 3) - 27.0 * (q ** 2)

    # Solve for roots a, b, c
    a, b, c = depressed_cubic_roots(p, q)

    # Basic Vieta consistency checks
    tol: float = 1e-8
    s1_num: complex = a + b + c
    s2_num: complex = a * b + b * c + c * a
    s3_num: complex = a * b * c

    assert abs(s1_num) <= 1e-6, f"Sum of roots should be 0, got {s1_num}"
    assert abs(s2_num - p) <= 1e-6, f"ab+bc+ca should be p={p}, got {s2_num}"
    assert abs(s3_num + q) <= 1e-6, f"abc should be -q={-q}, got {s3_num}"

    # Enumerate all 8 signed sums ±a ± b ± c
    signs = list(itertools.product([-1, 1], repeat=3))
    sums: List[complex] = [s[0] * a + s[1] * b + s[2] * c for s in signs]
    uniq_sums: List[complex] = unique_with_tol(sums, tol=1e-8)

    # The condition requires exactly 7 distinct sums (only collision from a + b + c = 0 and -a - b - c = 0)
    assert len(uniq_sums) == 7, f"Expected 7 distinct ±-sums, got {len(uniq_sums)}"

    # Further checks: discriminant nonzero and abc ≠ 0
    assert abs(s3_num) > 1e-12, "abc should be nonzero"
    assert abs(discriminant) > 1e-12, f"Discriminant should be nonzero, got {discriminant}"

    # Output a concise verification summary
    print("Verification for n = 132:")
    print(f"Roots a, b, c: {a}, {b}, {c}")
    print(f"Sum a+b+c ≈ {s1_num}")
    print(f"ab+bc+ca ≈ {s2_num} (target p = {p})")
    print(f"abc ≈ {s3_num} (target -q = {-q})")
    print(f"Discriminant Δ = {discriminant}")
    print(f"Number of distinct ±-sums = {len(uniq_sums)} (expected 7)")


if __name__ == "__main__":
    verify_n_equals_132()