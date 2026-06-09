from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


SQRT2: Final[float] = math.sqrt(2.0)
SQRT2PI: Final[float] = math.sqrt(2.0 * math.pi)


def _phi(z: float) -> float:
    """
    Standard normal PDF φ(z).
    """
    return math.exp(-0.5 * z * z) / SQRT2PI


def _Phi(z: float) -> float:
    """
    Standard normal CDF Φ(z).
    """
    return 0.5 * (1.0 + math.erf(z / SQRT2))


@dataclass(frozen=True)
class TruncMoments:
    """
    Truncated-Gaussian raw moments over the 'active' region where 1 + b_in * δ > 0:
      I0 = E[1_active], I1 = E[δ 1_active], I2 = E[δ^2 1_active].
    """
    I0: float
    I1: float
    I2: float


def truncated_moments(sigma: float, b_in: float) -> TruncMoments:
    """
    Compute I0, I1, I2 for δ ~ N(0, σ^2) over the active region 1 + b_in δ > 0.

    Definitions:
      - Threshold T = -1 / b_in (for b_in ≠ 0), z0 = T / σ = -1 / (b_in σ).
      - If b_in > 0 (upper tail, δ > T):
          I0 = 1 - Φ(z0)
          I1 = σ φ(z0)
          I2 = σ^2[(1 - Φ(z0)) + z0 φ(z0)]
        If b_in < 0 (lower tail, δ < T):
          I0 = Φ(z0)
          I1 = -σ φ(z0)
          I2 = σ^2[Φ(z0) - z0 φ(z0)]
        If b_in = 0: the active region is all δ, so I0 = 1, I1 = 0, I2 = σ^2.

    Parameters
    ----------
    sigma : float
        Standard deviation σ (> 0).
    b_in : float
        Input (linear) bias parameter.

    Returns
    -------
    TruncMoments
        Moments (I0, I1, I2) over the active region.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")
    if b_in == 0.0:
        return TruncMoments(I0=1.0, I1=0.0, I2=sigma * sigma)

    z0: float = -1.0 / (b_in * sigma)
    pdf: float = _phi(z0)

    if b_in > 0.0:
        I0 = 1.0 - _Phi(z0)
        I1 = sigma * pdf
        I2 = (sigma * sigma) * (I0 + z0 * pdf)
    else:
        I0 = _Phi(z0)
        I1 = -sigma * pdf
        I2 = (sigma * sigma) * (I0 - z0 * pdf)

    return TruncMoments(I0=I0, I1=I1, I2=I2)


def b_eff(sigma: float, b_in: float) -> float:
    """
    Effective large-scale bias b_eff obtained from regressing the (mean-normalized)
    clipped tracer on δ:

      n(δ) = n̄ * max(0, 1 + b_in δ)
      A = E[max(0, 1 + b_in δ)] = I0 + b_in I1
      B = E[δ max(0, 1 + b_in δ)] = I1 + b_in I2
      b_eff = (B / A) / σ^2 = (I1 + b_in I2) / [σ^2 (I0 + b_in I1)]

    Limits:
      - As σ → 0, b_eff → b_in.
      - As σ → ∞, b_eff → 0.

    Parameters
    ----------
    sigma : float
        Standard deviation σ (> 0).
    b_in : float
        Input linear bias parameter.

    Returns
    -------
    float
        Effective bias b_eff(σ, b_in).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")
    if b_in == 0.0:
        return 0.0

    moms = truncated_moments(sigma, b_in)
    A: float = moms.I0 + b_in * moms.I1
    B: float = moms.I1 + b_in * moms.I2

    # Guard against pathological near-zero A (should not occur for this transform).
    if A <= 0.0:
        # Numerically impossible under exact arithmetic; fall back to limit behavior.
        return 0.0

    return B / (sigma * sigma * A)


if __name__ == "__main__":
    # Minimal smoke tests illustrating behavior.
    for sigma in (0.1, 0.5, 1.0, 3.0):
        for b in (-2.0, -0.5, 0.0, 0.5, 2.0):
            be = b_eff(sigma, b)
            print(f"sigma={sigma:4.1f}, b_in={b:5.1f} -> b_eff={be: .6f}")