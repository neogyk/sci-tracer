import numpy as np
from typing import List, Tuple, Optional

def chi_2d_Lindhard(q: float, omega: float, gamma: float, kF: float, m_eff: float) -> complex:
    """
    Compute the 2D Lindhard polarizability (spin-degenerate) at T=0.
    Parameters:
        q: wavevector magnitude (a.u.)
        omega: frequency (a.u.)
        gamma: broadening (a.u.)
        kF: Fermi wavevector (a.u.)
        m_eff: effective mass (a.u., with m_e = 1)
    Returns:
        complex polarizability (a.u.)
    """
    # Use atomic units: hbar = 1, mass = m_eff (if m_eff = 1, it's free electron)
    # Integration limits
    k_max = 5 * kF
    nk = 400
    ntheta = 200
    k_vals = np.linspace(0, k_max, nk)
    theta_vals = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    dk = k_max / (nk-1)
    dtheta = 2*np.pi / ntheta

    # Precompute factors
    # For each (k, theta) we compute denominator and numerator
    # Use broadcasting: k (nk), theta (ntheta)
    kk = k_vals[:, np.newaxis]   # shape (nk,1)
    tt = theta_vals[np.newaxis, :]  # shape (1,ntheta)

    # Energy difference: ε(k+q) - ε(k) = k*q*cosθ + q^2/2  (since ε = k^2/(2*m_eff)? Wait: energy = k^2/(2m_eff). In a.u., mass is m_eff.
    # But we set hbar=1. So ε = k^2/(2*m_eff). So difference = (|k+q|^2 - k^2)/(2*m_eff) = (2k q cosθ + q^2)/(2*m_eff) = (k q cosθ + q^2/2)/m_eff.
    # Let's include m_eff factor.
    energy_diff = (kk * q * np.cos(tt) + q**2 / 2.0) / m_eff
    # Denominator: omega - energy_diff + 1j*gamma
    denom = omega - energy_diff + 1j*gamma

    # Numerator: f(k) - f(k+q)
    # f(k) = 1 if k < kF else 0
    k_norm = kk
    kq_norm = np.sqrt(kk**2 + q**2 + 2*kk*q*np.cos(tt))
    mask_k_inside = k_norm < kF
    mask_kq_outside = kq_norm > kF
    mask_k_outside = k_norm > kF
    mask_kq_inside = kq_norm < kF
    numerator = np.zeros_like(denom, dtype=float)
    numerator[mask_k_inside & mask_kq_outside] = 1.0
    numerator[mask_k_outside & mask_kq_inside] = -1.0

    # Integrand: (1/(2π)^2) * 2 (spin) * k * numerator / denom
    # Factor: 2 for spin, and dk dθ k/(2π)^2
    integrand = 2.0 * numerator * kk / denom   # shape (nk, ntheta)
    # Integrate using simple trapezoidal rule in 2D
    integral = np.trapz(np.trapz(integrand, theta_vals, axis=1), k_vals)
    # Multiply by dk and dtheta? Actually np.trapz already includes the spacing.
    # But we need to normalize by (2π)^2. The integration over dk and dθ: we have k dk dθ. The factor for the full integral is 1/(2π)^2.
    # So final result = integral / (2π)^2
    chi = integral / (2*np.pi)**2
    return chi

def build_Coulomb_matrix(q: float, L: int, d: float, bg_eps: float) -> np.ndarray:
    """
    Build the Coulomb interaction matrix V_{ll'} for layered electron gas
    in a semi-infinite dielectric medium with vacuum interface.
    V_{ll'} = (2π / bg_eps) * [ e^{-q|z_l - z_{l'}|} + r * e^{-q(z_l + z_{l'})} ]
    where r = (bg_eps - 1)/(bg_eps + 1)
    Parameters:
        q: in-plane wavevector magnitude (a.u.)
        L: number of layers
        d: interlayer spacing (a.u.)
        bg_eps: background dielectric constant
    Returns:
        LxL matrix (a.u.)
    """
    r = (bg_eps - 1) / (bg_eps + 1)
    z = np.arange(L) * d   # positions of layers, assume first layer at z=0
    # Use broadcasting to compute |z_l - z_{l'}|
    dz = np.abs(z[:, np.newaxis] - z[np.newaxis, :])
    sum_z = z[:, np.newaxis] + z[np.newaxis, :]
    V = (2*np.pi / bg_eps) * (np.exp(-q*dz) + r * np.exp(-q*sum_z))
    return V

def I_Raman(q: float, d: float, omega: float, gamma: float, n_eff: float, e_F: float,
            k_F: float, v_F: float, bg_eps: float, delta_E: float, kd: float) -> float:
    """
    Compute the Raman intensity for a semi-infinite layered electron gas within RPA.
    Parameters:
        q: in-plane wavevector magnitude (a.u.)
        d: interlayer spacing (a.u.)
        omega: frequency (a.u.)
        gamma: broadening (a.u.)
        n_eff: effective 2D electron density (a.u.^-2) - not used directly, but maybe for consistency.
        e_F: Fermi energy (a.u.)
        k_F: Fermi wavevector (a.u.)
        v_F: Fermi velocity (a.u.)
        bg_eps: background dielectric constant
        delta_E: penetration depth of light (in units of layer spacing? Actually used as δ in exponential decay)
        kd: perpendicular wavevector times d (dimensionless), appears in phase factor e^{-2i kd (l-l')}
    Returns:
        Raman intensity (arbitrary units)
    """
    # Compute effective mass from Fermi velocity and Fermi wavevector: v_F = ℏ k_F / m_eff
    # In a.u., ℏ=1, so m_eff = k_F / v_F
    m_eff = k_F / v_F   # effective mass in a.u.

    # Number of layers: choose a sufficient number, say L = 30 (maybe more if d/δ is large)
    # We need to ensure that the exponential decay factor e^{-(l+l')d/δ} is small for large l.
    # Let's set L such that L*d > 3*delta_E (to be safe)
    L = max(int(np.ceil(5 * delta_E / d)), 20)  # at least 20 layers
    # But also limit to avoid large matrices: L up to 100 maybe.
    L = min(L, 100)

    # Compute the 2D Lindhard polarizability D0 (complex)
    D0 = chi_2d_Lindhard(q, omega, gamma, k_F, m_eff)

    # Build Coulomb interaction matrix V
    V = build_Coulomb_matrix(q, L, d, bg_eps)

    # Solve Dyson equation: D = (I - D0 V)^{-1} * D0
    # D0 is a scalar, so we can write D = D0 * inv(I - D0 V)
    I_mat = np.eye(L)
    inv_mat = np.linalg.inv(I_mat - D0 * V)
    D = D0 * inv_mat   # L x L complex

    # Compute the exponential factors for Raman sum
    l_indices = np.arange(L)
    # Factor1: exp(-(l+l') * d / delta_E)
    factor1 = np.exp( - (l_indices[:, None] + l_indices[None, :]) * d / delta_E )
    # Factor2: exp(-2i * kd * (l - l'))
    factor2 = np.exp( -2j * kd * (l_indices[:, None] - l_indices[None, :]) )
    # Sum over all l,l'
    total = np.sum( D * factor1 * factor2 )
    # Intensity: I(ω) = - Im(total)
    I_omega = -np.imag(total)

    return float(I_omega)

# Example usage (for testing, not part of final output)
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     # Test parameters (approximate for GaAs 2DEG)
#     q = 0.01  # a.u.
#     d = 1.0   # a.u.
#     omega = 0.1
#     gamma = 0.01
#     n_eff = 1e11  # cm^-2, convert to a.u.: 1 a.u. = 1/(π a0^2) ~ 1.6e15 cm^-2, so very small
#     # Instead use typical a.u. values: e_F = 0.01 a.u., k_F = 0.1, v_F = 0.1
#     e_F = 0.01
#     k_F = 0.1
#     v_F = 0.1
#     bg_eps = 13.0
#     delta_E = 100.0  # large penetration depth
#     kd = 0.0
#     I = I_Raman(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd)
#     print("Raman intensity:", I)