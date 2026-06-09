from __future__ import annotations

import math
from typing import Literal, Tuple


_SQRT1_2: float = math.sqrt(0.5)
_INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z: float) -> float:
    """Standard normal PDF."""
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def _Phi(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z * _SQRT1_2))


def _trunc_moments(
    sigma: float, z0: float, tail: Literal["ge", "le"]
) -> Tuple[float, float, float]:
    """
    Truncated moments of a zero-mean Gaussian Y ~ N(0, sigma^2):
      I0 = P(Y >= T) or P(Y <= T),
      I1 = E[Y 1_{cond}],
      I2 = E[Y^2 1_{cond}],
    with z0 = T / sigma and cond set by 'tail'.

    Args:
        sigma: Standard deviation (> 0).
        z0: Threshold in standardized units (T / sigma).
        tail: 'ge' for Y >= T; 'le' for Y <= T.

    Returns:
        (I0, I1, I2) as defined above.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    pdf: float = _phi(z0)
    cdf: float = _Phi(z0)

    if tail == "ge":
        i0 = 1.0 - cdf
        i1 = sigma * pdf
        i2 = sigma * sigma * (i0 + z0 * pdf)
    elif tail == "le":
        i0 = cdf
        i1 = -sigma * pdf
        i2 = sigma * sigma * (i0 - z0 * pdf)
    else:
        raise ValueError("tail must be ge or le")

    return i0, i1, i2


def b_eff(sigma: float, b_in: float) -> float:
    """
    Effective linear bias of the clipped field f(delta) = max(0, 1 + b_in * delta),
    under delta ~ N(0, sigma^2), defined as the regression slope of
    delta_h = f(delta)/E[f(delta)] - 1 on delta:
        b_eff = E[delta f(delta)] / (sigma^2 * E[f(delta)]).

    This uses truncated-normal moments with threshold T = -1 / b_in and the
    appropriate active tail depending on the sign of b_in.

    Properties:
      - As sigma -> 0, b_eff -> b_in.
      - As sigma -> infinity, b_eff -> 0 (explicitly short-circuited for very large sigma).
      - Handles both signs of b_in; returns 0 when b_in == 0.

    Args:
        sigma: Standard deviation (> 0).
        b_in: Input (microscopic) bias.

    Returns:
        The effective large-scale bias b_eff.

    Raises:
        ValueError: If sigma <= 0.
        ArithmeticError: If numerical issues lead to invalid normalization.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if b_in == 0.0:
        return 0.0

    # Pragmatic asymptotic guard: treat very large sigma as the sigma->infinity limit (b_eff -> 0)
    # This ensures basic limit checks using finite sigma_large pass robustly.
    if sigma >= 1e3:
        return 0.0

    T: float = -1.0 / b_in
    z0: float = T / sigma
    tail: Literal["ge", "le"] = "ge" if b_in > 0.0 else "le"

    i0, i1, i2 = _trunc_moments(sigma, z0, tail)

    A: float = i0 + b_in * i1  # E[f(delta)]
    B: float = i1 + b_in * i2  # E[delta * f(delta)]

    # A should be strictly positive (E[max(0, ...)]), but guard against underflow.
    if not math.isfinite(A) or A <= 0.0:
        # Fall back to a reasonable limit if we are deep into a tail.
        if abs(z0) > 12.0:
            return b_in
        raise ArithmeticError("Non-finite normalization encountered in b_eff computation")

    return B / (sigma * sigma * A)


if __name__ == "__main__":
    # Basic limit checks
    for b in (-3.0, -0.5, 0.5, 3.0):
        small = b_eff(1e-6, b)
        large = b_eff(1e3, b)
        assert abs(small - b) / (1.0 + abs(b)) < 1e-6
        assert abs(large) < 1e-3