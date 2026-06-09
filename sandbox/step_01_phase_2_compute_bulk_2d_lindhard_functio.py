import numpy as np
import cmath

def D0(q, omega, gamma, e_F, k_F, v_F, bg_eps):
    """
    Compute the bulk 2D Lindhard function (polarization) at complex frequency.
    Formula: D0 = (k_F**2/(2π e_F)) * [1 - 0.5*(sqrt(1 - xi_-^2) + sqrt(1 - xi_+^2))]
    where xi_± = (ħω ± ħ v_F q)/(2E_F) with ħ=1.
    """
    # In units where ħ = 1
    prefactor = k_F**2 / (2 * np.pi * e_F)
    omega_c = omega + 1j * gamma
    xi_minus = (omega_c - v_F * q) / (2 * e_F)
    xi_plus  = (omega_c + v_F * q) / (2 * e_F)
    # Use cmath.sqrt for the principal branch (Im>0)
    sqrt_minus = cmath.sqrt(1 - xi_minus**2)
    sqrt_plus  = cmath.sqrt(1 - xi_plus**2)
    D0_val = prefactor * (1 - 0.5 * (sqrt_minus + sqrt_plus))
    return D0_val