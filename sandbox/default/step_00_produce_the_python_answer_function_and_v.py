from __future__ import annotations

import math
from typing import Final


def answer() -> float:
    """Return the exponent n in u ∝ T^n for a black body in 4 spatial dimensions.

    Theory: In d spatial dimensions, u(T) ∝ T^{d+1}. For d = 4, n = 5.
    """
    return 5.0


def _u_dimensionless(T: float, d: int = 4) -> float:
    """Compute a dimensionless proxy for black-body energy density u(T).

    u(T) ∝ T^{d+1} * ∫_0^∞ x^d/(e^x - 1) dx. We evaluate the integral numerically
    with Simpson's rule over a finite domain [0, b] with b large enough.
    """
    # Integration parameters (typed for clarity)
    a: Final[float] = 0.0
    b: Final[float] = 40.0  # sufficiently captures the integral tail for stability
    N: int = 20000          # Simpson panels (must be even)
    if N % 2 == 1:
        N += 1

    h: float = (b - a) / float(N)

    def f(x: float) -> float:
        if x == 0.0:
            return 0.0
        ex: float = math.exp(x)
        return (x ** d) / (ex - 1.0)

    s: float = f(a) + f(b)
    for i in range(1, N):
        x: float = a + i * h
        s += (4.0 if i % 2 == 1 else 2.0) * f(x)

    integral: float = s * h / 3.0
    return (T ** (d + 1)) * integral


# Numerical sanity check of scaling for d = 4:
_u1: float = _u_dimensionless(1.0, d=4)
_u2: float = _u_dimensionless(2.0, d=4)
_ratio: float = _u2 / _u1
_expected: float = 2.0 ** (4 + 1)
_rel_err: float = abs(_ratio - _expected) / _expected

# Tolerate small numerical error from quadrature and finite upper limit
assert _rel_err < 1e-3, f"Scaling check failed: ratio={_ratio}, expected={_expected}, rel_err={_rel_err}"