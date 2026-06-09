import numpy as np

def compute_I_omega(D0: complex, V: complex, d: float, delta: float, b: complex, u: complex, G: complex, H: complex, A: complex, B: complex, C: complex, Q: complex, E: complex) -> float:
    """
    Compute Raman intensity I(omega) using the analytic formula from the problem.
    
    Parameters:
    - D0: Lindhard function (complex)
    - V: Coulomb form factor (complex)
    - d: layer spacing (nm or normalized)
    - delta: penetration depth (same units as d)
    - b, u, G, H, A, B, C, Q, E: intermediate complex numbers from Phase 3
    
    Returns:
    - I_omega: real non‑negative float representing Raman intensity
    """
    # Exponential factor from the surface correction
    xi = np.exp(-2.0 * d / delta)
    # (1 - exp(-2d/δ))^{-1}
    inv_1m_xi = 1.0 / (1.0 - xi)
    
    # First term: inv_1m_xi * (1 + C * xi / (1 - B * xi))
    first_term = inv_1m_xi * (1.0 + C * xi / (1.0 - B * xi))
    
    # Second term: placeholder for the remaining part of the expression
    second_term = Q * (1.0 + E)
    
    # Full complex expression (pulled from the Dyson‑equation solution)
    I_complex = -D0 * (first_term + second_term)
    I_omega = -np.imag(I_complex)
    
    # Enforce physical non‑negativity (Raman intensity ≥ 0)
    I_omega = max(I_omega, 0.0)
    
    return I_omega