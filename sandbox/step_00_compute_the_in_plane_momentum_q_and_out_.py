import numpy as np

def q_cal(th, gamma, E0, omega):
    """
    Compute in-plane momentum transfer q and out-of-plane momenta k_i_z and k_s_z.
    
    Parameters
    ----------
    th : list or array of floats
        Incident angle(s) from the surface (in degrees).
    gamma : list or array of floats
        Total scattering angle(s) (in degrees), measured from the incident direction.
    E0 : float
        Incident electron energy (eV).
    omega : list or array of floats
        Energy loss (eV). Must have same length as th and gamma.
    
    Returns
    -------
    q : numpy.ndarray
        In-plane momentum transfer (Å⁻¹).
    k_i_z : numpy.ndarray
        Out-of-plane component of incident wavevector (Å⁻¹, positive into sample).
    k_s_z : numpy.ndarray
        Out-of-plane component of scattered wavevector (Å⁻¹, negative outward).
    
    Notes
    -----
    Nonrelativistic kinematics: k = sqrt(2mE)/ħ.
    Constants: m_e_c2 = 510998.95 eV, hc = 1239.84193 eV·nm, ħc = hc/(2π) ≈ 197.3269788 eV·nm.
    Output in Å⁻¹: 1 nm = 10 Å.
    Coordinate system: x is in-plane along the surface, z is perpendicular (positive into sample).
    Incident beam direction: angle th from surface, so k_i_x = k_i cos(th), k_i_z = k_i sin(th).
    Scattered beam: total scattering angle γ, direction relative to incident; β = γ - th is the
    exit angle from the surface (measured on the opposite side). Then k_s_x = k_s cos(β),
    k_s_z = -k_s sin(β) (negative because outgoing).
    """
    # Convert to numpy arrays
    th_rad = np.deg2rad(np.asarray(th))
    gamma_rad = np.deg2rad(np.asarray(gamma))
    omega_arr = np.asarray(omega)
    
    # Constants
    m_e_c2 = 510998.95  # eV
    hc = 1239.84193     # eV·nm
    hbar_c = hc / (2 * np.pi)  # eV·nm, 197.3269788
    
    # Wavevector factor (nm⁻¹ per sqrt(eV))
    factor_nm = np.sqrt(2 * m_e_c2) / hbar_c
    # Convert to Å⁻¹ (1 nm = 10 Å)
    factor = factor_nm / 10.0  # Å⁻¹ per sqrt(eV)
    
    # Incident wavevector magnitude
    k_i = factor * np.sqrt(E0)
    
    # Scattered wavevector magnitude (ensure non-negative radicand)
    radicand = E0 - omega_arr
    if np.any(radicand < 0):
        # For unphysical losses, set k_s to NaN
        k_s = np.where(radicand >= 0, factor * np.sqrt(radicand), np.nan)
    else:
        k_s = factor * np.sqrt(radicand)
    
    # Components
    k_i_x = k_i * np.cos(th_rad)
    k_i_z = k_i * np.sin(th_rad)   # positive into sample
    
    # Exit angle β = γ - θ
    beta_rad = gamma_rad - th_rad
    k_s_x = k_s * np.cos(beta_rad)
    k_s_z = -k_s * np.sin(beta_rad)  # negative outward
    
    # In-plane momentum transfer
    q = k_s_x - k_i_x
    
    return q, k_i_z, k_s_z