from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple


SQRT1_2: float = 1.0 / math.sqrt(2.0)
SQRT_2PI: float = math.sqrt(2.0 * math.pi)


def _phi(z: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * z * z) / SQRT_2PI


def _Phi(z: float) -> float:
    """Standard normal CDF via erfc for numerical stability."""
    return 0.5 * math.erfc(-z * SQRT1_2)


def b_eff(sigma: float, b_in: float) -> float:
    """
    Effective large-scale bias after clipping: n = nbar * max(0, 1 + b_in * delta),
    delta ~ N(0, sigma^2).
    Returns slope = <delta_h delta> / <delta^2>, where delta_h = n/<n> - 1.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    # Small-sigma limit: essentially linear response if clipping is negligible
    if sigma == 0.0:
        # Use limit behavior
        return float(b_in)
    if b_in == 0.0:
        return 0.0

    z0: float = -1.0 / (b_in * sigma)

    if b_in > 0.0:
        # Active region: delta > T, T = -1/b
        I0: float = 0.5 * math.erfc(z0 * SQRT1_2)  # 1 - Phi(z0)
        ph: float = _phi(z0)
        I1: float = sigma * ph
        I2: float = sigma * sigma * (I0 + z0 * ph)
    else:
        # b_in < 0: Active region: delta < T
        I0 = _Phi(z0)
        ph = _phi(z0)
        I1 = -sigma * ph
        I2 = sigma * sigma * (I0 - z0 * ph)

    A: float = I0 + b_in * I1
    B: float = I1 + b_in * I2

    # A is expectation of a nonnegative function; protect against underflow
    if A <= 0.0:
        # In extreme numeric underflow, fall back to limit behavior
        # For very large |z0| where clipping is negligible, return ~ b_in
        return float(b_in)

    return B / (A * sigma * sigma)


def _ndtri_acklam(p: float) -> float:
    """
    Acklam's inverse-normal (probit) approximation.
    Valid for p in (0,1). Maximum error ~1e-9.
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0,1)")

    # Coefficients from Peter J. Acklam's algorithm
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    )


def build_z_grid(n: int) -> List[float]:
    """
    Deterministic quantile grid: z_i = Phi^{-1}((i+0.5)/n), i=0..n-1
    """
    if n <= 0:
        raise ValueError("n must be positive")
    step = 1.0 / n
    half = 0.5 * step
    return [_ndtri_acklam((i * step) + half) for i in range(n)]


def slope_qmc(sigma: float, b_in: float, z_grid: List[float]) -> float:
    """
    Deterministic quasi-MC estimate of slope = <delta_h delta>/sigma^2
    using quantiles z ~ N(0,1), delta = sigma * z.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0.0:
        return float(b_in)
    n = len(z_grid)
    s_f: float = 0.0
    s_df: float = 0.0
    for z in z_grid:
        d = sigma * z
        f = 1.0 + b_in * d
        if f < 0.0:
            f = 0.0
        s_f += f
        s_df += d * f
    mean_f = s_f / n
    mean_df = s_df / n
    if mean_f <= 0.0:
        # Degenerate (should not happen); return limiting behavior
        return float(b_in)
    return (mean_df / mean_f) / (sigma * sigma)


def validate() -> Dict[str, float]:
    """
    Validate analytical b_eff against deterministic QMC across a grid.
    Asserts:
      - relative error <= 1% where |b_eff|>1e-3
      - absolute error <= 1e-4 where |b_eff|<=1e-3
    """
    # Precompute z-grid once
    N: int = 32768  # balance accuracy and runtime
    z_grid = build_z_grid(N)

    sigmas: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    bvals: List[float] = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

    max_rel_err: float = 0.0
    max_abs_small: float = 0.0
    worst_rel_case: Tuple[float, float] | None = None
    worst_abs_case: Tuple[float, float] | None = None

    for sigma in sigmas:
        for b in bvals:
            be = b_eff(sigma, b)
            est = slope_qmc(sigma, b, z_grid)
            err = abs(est - be)
            if abs(be) > 1e-3:
                rel = err / abs(be)
                if rel > max_rel_err:
                    max_rel_err = rel
                    worst_rel_case = (sigma, b)
            else:
                if err > max_abs_small:
                    max_abs_small = err
                    worst_abs_case = (sigma, b)

    # Assertions per spec
    assert max_rel_err <= 0.01 + 1e-12, f"Relative error too high: {max_rel_err:.6g} at {worst_rel_case}"
    assert max_abs_small <= 1e-4 + 1e-12, f"Absolute error (small) too high: {max_abs_small:.6g} at {worst_abs_case}"

    return {
        "max_rel_err": max_rel_err,
        "max_abs_small": max_abs_small,
        "N": float(N),
    }


# Produce validation metrics at import time so the harness can access them
metrics: Dict[str, float] = validate()

if __name__ == "__main__":
    # Quick sanity checks on limits
    assert abs(b_eff(1e-9, 1.5) - 1.5) < 1e-6
    # Large-sigma asymptotic: b_eff ~ sqrt(pi/2)/sigma (for b_in=+1)
    sigma_large = 1e2
    val = b_eff(sigma_large, 1.0)
    expected_asym = math.sqrt(0.5 * math.pi) / sigma_large
    # Be conservative: value should be below 2x the asymptotic estimate
    assert val < 2.0 * expected_asym

    # Use precomputed metrics
    print("Validation passed.")
    print(f"Max relative error (|b_eff|>1e-3): {metrics['max_rel_err']:.6g}")
    print(f"Max absolute error (|b_eff|<=1e-3): {metrics['max_abs_small']:.6g}")