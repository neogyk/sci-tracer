import numpy as np
from scipy.integrate import solve_ivp, OdeSolution
from typing import List

def eos_press_from_rho(rho: float, eos_Gamma: float, eos_kappa: float) -> float:
    """
    Polytropic EOS: P = kappa * rho^Gamma.
    """
    return eos_kappa * (rho ** eos_Gamma)

def tov_integrate(rhoc: float, Gamma: float, kappa: float, npoints: int, rmax: float) -> OdeSolution:
    """
    Integrate the TOV equations from the centre to the surface of a neutron star.
    
    Parameters
    ----------
    rhoc : float
        Central density.
    Gamma : float
        Polytropic exponent.
    kappa : float
        Polytropic constant.
    npoints : int
        Number of integration points (controls max_step).
    rmax : float
        Maximum radius for integration (should be larger than star radius).
    
    Returns
    -------
    OdeSolution
        The solution object containing arrays for r, P, m, and phi.
    """
    def tov_odes(r: float, y: List[float]) -> List[float]:
        P, m, phi = y
        if P <= 1e-15:
            return [0.0, 0.0, 0.0]
        rho = (P / kappa) ** (1.0 / Gamma)
        # Avoid division by zero at r=0 by adding a small constant
        denom = r * (r - 2.0 * m) + 1e-30
        dPdr = -(rho + P) * (m + 4.0 * np.pi * r**3 * P) / denom
        dmdr = 4.0 * np.pi * r**2 * rho
        dphidr = (m + 4.0 * np.pi * r**3 * P) / denom
        return [dPdr, dmdr, dphidr]

    def pressure_zero(r: float, y: List[float]) -> float:
        return y[0] - 1e-15

    pressure_zero.terminal = True
    pressure_zero.direction = -1

    P_c = eos_press_from_rho(rhoc, Gamma, kappa)
    y0 = [P_c, 0.0, 0.0]
    sol = solve_ivp(tov_odes, [0.0, rmax], y0,
                    max_step=rmax / npoints, method='RK45',
                    events=pressure_zero, rtol=1e-8, atol=1e-10)
    return sol