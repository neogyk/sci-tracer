from math import erf, sqrt, exp, pi
from typing import Final

# Constants with typing
SQRT2: Final[float] = sqrt(2.0)
SQRT2PI: Final[float] = sqrt(2.0 * pi)
INV_SQRT2PI: Final[float] = 1.0 / SQRT2PI


def _phi(x: float) -> float:
    """
    Standard normal PDF.
    """
    return INV_SQRT2PI * exp(-0.5 * x * x)


def _Phi(x: float) -> float:
    """
    Standard normal CDF via the error function.
    """
    return 0.5 * (1.0 + erf(x / SQRT2))


def b_eff(sigma: float, b_in: float) -> float:
    """
    Effective linear bias for a clipped-linear mapping:
      n = nbar * max(0, 1 + b_in * delta),  with delta ~ N(0, sigma^2)

    We define delta_h = (n - <n>)/<n>, and return
      b_eff = <delta_h * delta> / <delta^2>.

    Closed-form result using truncated-normal moments:
      Let gamma = b_in * sigma, t = -1/gamma.
      For gamma > 0 (keep z > t):
        I0 = (1 - Phi(t)) + gamma * phi(t)
        I1 = phi(t) + gamma * ((1 - Phi(t)) + t * phi(t))
      For gamma < 0 (keep z < t):
        I0 = Phi(t) - gamma * phi(t)
        I1 = -phi(t) + gamma * (Phi(t) - t * phi(t))
      b_eff = I1 / (sigma * I0)

    Edge case: for |gamma| very small (clipping inactive), b_eff -> b_in.

    Parameters
    ----------
    sigma : float
        Standard deviation of matter overdensity (must be > 0).
    b_in : float
        Bare linear bias coefficient.

    Returns
    -------
    float
        Effective linear bias.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    gamma: float = b_in * sigma

    # If clipping is effectively inactive, return the linear (bare) bias
    if abs(gamma) < 1e-14:
        return b_in

    t: float = -1.0 / gamma

    if gamma > 0.0:
        I0: float = (1.0 - _Phi(t)) + gamma * _phi(t)
        I1: float = _phi(t) + gamma * ((1.0 - _Phi(t)) + t * _phi(t))
    else:
        I0 = _Phi(t) - gamma * _phi(t)
        I1 = -_phi(t) + gamma * (_Phi(t) - t * _phi(t))

    return I1 / (sigma * I0)


# Quick smoke tests for sanity in regimes where clipping is negligible
# Very small sigma: clipping is vanishingly rare; b_eff ~ b_in
assert abs(b_eff(1e-3, 2.0) - 2.0) < 1e-8
assert abs(b_eff(1e-3, -1.5) - (-1.5)) < 1e-8

print("b_eff implementation loaded.")