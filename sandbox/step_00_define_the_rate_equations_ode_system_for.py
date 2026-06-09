import numpy as np

def ode(z: float, y: np.ndarray,
        Ppsat: float, Pssat: float, N: float,
        sigma_ap: float, sigma_ep: float,
        sigma_as: float, sigma_es: float,
        gamma_p: float, gamma_s: float,
        alpha_p: float, alpha_s: float) -> np.ndarray:
    """
    Rate equations for forward/backward pump and signal in a doped fiber laser.
    
    State vector y: [Pp_fwd, Pp_bwd, Ps_fwd, Ps_bwd]
    Returns dydz as numpy array.
    """
    # Extract components
    Pp_fwd, Pp_bwd, Ps_fwd, Ps_bwd = y
    
    # Total pump and signal powers at point z
    Pp_total = Pp_fwd + Pp_bwd
    Ps_total = Ps_fwd + Ps_bwd
    
    # Inversion (normalized to total doping N) using standard formula:
    # n2 = N * ( (sigma_ap/(sigma_ap+sigma_ep))*Pp_total/Ppsat + (sigma_as/(sigma_as+sigma_es))*Ps_total/Pssat )
    #       / ( 1 + Pp_total/Ppsat + Ps_total/Pssat )
    # To avoid division by zero, add small epsilon if needed; but here we assume denominators > 0.
    inv_denom = 1.0 + Pp_total / Ppsat + Ps_total / Pssat
    n2 = N * ( (sigma_ap / (sigma_ap + sigma_ep)) * Pp_total / Ppsat +
               (sigma_as / (sigma_as + sigma_es)) * Ps_total / Pssat ) / inv_denom
    
    # Net gain coefficients for pump and signal
    g_p = gamma_p * ( (sigma_ep + sigma_ap) * n2 - sigma_ap * N )
    g_s = gamma_s * ( (sigma_es + sigma_as) * n2 - sigma_as * N )
    
    # Derivative vector
    dPp_fwd_dz = (g_p - alpha_p) * Pp_fwd
    dPp_bwd_dz = - (g_p - alpha_p) * Pp_bwd
    dPs_fwd_dz = (g_s - alpha_s) * Ps_fwd
    dPs_bwd_dz = - (g_s - alpha_s) * Ps_bwd
    
    return np.array([dPp_fwd_dz, dPp_bwd_dz, dPs_fwd_dz, dPs_bwd_dz])