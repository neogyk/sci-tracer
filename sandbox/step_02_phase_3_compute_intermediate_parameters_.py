import numpy as np

def compute_intermediates(D0: complex, q: float, d: float, delta: float, V: float) -> tuple:
    """
    Compute the intermediate dimensionless quantities b, u, G, H, A, B, C, Q, E
    as defined in the problem statement (Jain & Allen 1985, Das Sarma & Quinn 1982).
    
    Parameters:
    - D0  : complex, 2D Lindhard polarizability at the given q and complex frequency
    - q   : float, in-plane wavevector (inverse length)
    - d   : float, layer spacing
    - delta: float, penetration depth of the vacuum interface
    - V   : float, 2D Coulomb potential V(q) = 2π e^2 / (ε q) (appropriate units)
    
    Returns:
    Tuple of nine complex numbers: b, u, G, H, A, B, C, Q, E
    """
    # --- b and u ---
    b = np.cosh(q * d) - D0 * V * np.sinh(q * d)
    # Square root with branch such that Im(sqrt(b^2-1)) ≥ 0 (causality)
    disc = b**2 - 1.0
    sqrt_disc = np.sqrt(disc, dtype=complex)
    # NumPy sqrt principal branch has Re ≥ 0; we need Im ≥ 0 for retarded response
    # (if Im(sqrt_disc) < 0, choose the other branch)
    if np.imag(sqrt_disc) < 0:
        sqrt_disc = -sqrt_disc
    u = b + sqrt_disc

    # --- Penetration factor (note: phase 4 uses exp(-2d/δ) ) ---
    gamma = np.exp(-2.0 * d / delta)   # = exp(-2d/δ)

    # --- Derived intermediates (common form from semi‑infinite RPA) ---
    # A = (1 - u^2)/(2u)
    A = (1.0 - u**2) / (2.0 * u)
    # B = (1 - u^{-2})/(2 u^{-1}) = (u^2 - 1)/(2u) = -A
    B = (u**2 - 1.0) / (2.0 * u)      # = -A
    # G = 1/(1 - u * gamma)
    G = 1.0 / (1.0 - u * gamma)
    # H = 1/(1 - u^{-1} * gamma)
    H = 1.0 / (1.0 - u**(-1) * gamma)
    # C = sqrt(b^2 - 1)  (the numerator in the surface terms)
    C = sqrt_disc.copy()
    # Q = u  (the larger eigenvalue)
    Q = u.copy()
    # E = b  (the trace parameter)
    E = b.copy()

    # All values are complex; match the postcondition.
    return b, u, G, H, A, B, C, Q, E