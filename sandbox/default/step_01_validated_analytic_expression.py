import numpy as np
from typing import Tuple, Optional
from math import erf, sqrt, exp, pi

SQRT2: float = sqrt(2.0)
SQRT2PI: float = sqrt(2.0 * pi)
INV_SQRT2PI: float = 1.0 / SQRT2PI

def _phi(x: float) -> float:
    return INV_SQRT2PI * exp(-0.5 * x * x)

def _Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT2))

def b_eff(sigma: float, b_in: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    gamma: float = b_in * sigma
    if abs(gamma) < 1e-14:
        return 0.0
    t: float = -1.0 / gamma
    if gamma > 0.0:
        I0: float = (1.0 - _Phi(t)) + gamma * _phi(t)
        I1: float = _phi(t) + gamma * ((1.0 - _Phi(t)) + t * _phi(t))
    else:
        I0 = _Phi(t) - gamma * _phi(t)
        I1 = -_phi(t) + gamma * (_Phi(t) - t * _phi(t))
    return I1 / (sigma * I0)

def monte_carlo_validation(seed: int = 12345) -> Tuple[float, Optional[Tuple[float, float, float, float, float]]]:
    rng = np.random.default_rng(seed)
    N: int = 1_200_000
    Z = rng.standard_normal(N)
    nb: float = 1.0
    sigmas = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0], dtype=float)
    bvals  = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
    max_rel_err: float = 0.0
    worst: Optional[Tuple[float, float, float, float, float]] = None
    for sigma in sigmas:
        delta = sigma * Z
        var_delta = float(sigma * sigma)
        for b_in in bvals:
            n = nb * np.maximum(0.0, 1.0 + float(b_in) * delta)
            nbar_emp = float(n.mean())
            delta_h = n / nbar_emp - 1.0
            be_emp = float(np.mean(delta_h * delta) / var_delta)
            be_th = float(b_eff(float(sigma), float(b_in)))
            denom = max(1e-12, abs(be_th))
            rel_err = abs(be_emp - be_th) / denom
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                worst = (float(sigma), float(b_in), be_emp, be_th, rel_err)
            assert rel_err <= 0.01, f"Rel error >1% at sigma={sigma}, b_in={b_in}: emp={be_emp}, th={be_th}, rel={rel_err}"
    return max_rel_err, worst

if __name__ == "__main__":
    max_rel_err, worst = monte_carlo_validation()
    print(f"Monte Carlo validation passed. Max relative error = {max_rel_err:.5f}. Worst case (sigma, b_in, emp, th, rel) = {worst}")