import numpy as np

def extract_output_power_and_inversion(sol, R2, N, sigma_ap, sigma_ep, sigma_as, sigma_es, tau):
    """
    From the BVP solution, compute the output power and the population inversion profile.

    Parameters:
    - sol: scipy.integrate.solve_bvp solution object
    - R2: reflectivity of the output mirror (at z=L)
    - N: total dopant concentration (m^-3)
    - sigma_ap, sigma_ep: absorption and emission cross-sections for pump (m^2)
    - sigma_as, sigma_es: absorption and emission cross-sections for signal (m^2)
    - tau: upper-state lifetime (s)

    Returns:
    - Pout: output power (W)
    - nz: population inversion profile (array of same length as sol.t)
    """
    # Extract power profiles
    Pp_forward = sol.y[0]  # forward pump
    Pp_backward = sol.y[1]  # backward pump
    Ps_forward = sol.y[2]  # forward signal
    Ps_backward = sol.y[3]  # backward signal

    # Total cavity powers at each z
    Pp_cavity = Pp_forward + Pp_backward
    Ps_cavity = Ps_forward + Ps_backward

    # Output power: forward signal at output end (z=L) transmitted through mirror
    Pout = (1 - R2) * Ps_forward[-1]

    # Population inversion density (number density of excited ions)
    # nz = N * (sigma_ap * Pp_cavity + sigma_as * Ps_cavity) /
    #        (1/tau + (sigma_ap+sigma_ep)*Pp_cavity + (sigma_as+sigma_es)*Ps_cavity)
    numerator = sigma_ap * Pp_cavity + sigma_as * Ps_cavity
    denominator = 1.0 / tau + (sigma_ap + sigma_ep) * Pp_cavity + (sigma_as + sigma_es) * Ps_cavity
    nz = N * numerator / denominator

    return Pout, nz

# Example usage (assuming sol and parameters are already defined):
# Pout, nz = extract_output_power_and_inversion(sol, R2, N, sigma_ap, sigma_ep, sigma_as, sigma_es, tau)