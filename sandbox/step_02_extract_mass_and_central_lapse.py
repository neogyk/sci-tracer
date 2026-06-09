import numpy as np

def compute_mass_lapse(sol):
    """
    Extract total mass and central lapse from a TOV integration solution.

    Parameters
    ----------
    sol : scipy.integrate.OdeSolution
        Solution object from solve_ivp with fields:
            sol.t : array of radial coordinates
            sol.y[0] : pressure array
            sol.y[1] : enclosed mass array
            sol.y[2] : metric potential array

    Returns
    -------
    tuple (float, float)
        (total mass M, central lapse alpha_c) in
        geometric units (G=c=Msun=1).
    """
    r = sol.t
    P = sol.y[0]
    m = sol.y[1]
    phi = sol.y[2]

    # Find the first index where pressure drops below threshold
    idx = np.where(P <= 1e-15)[0]
    if len(idx) == 0:
        # If no surface found, take the last point
        idx = np.array([len(P) - 1])
    idx = idx[0]

    # Radius and mass at surface
    R = r[idx]
    M = m[idx]

    # Metric potential at surface and centre
    phi_s = phi[idx]
    phi_0 = phi[0]

    # Central lapse from outer boundary condition:
    # alpha_c = exp(phi_0 - phi_s + 0.5 * ln(1 - 2M/R))
    lapse = np.exp(phi_0 - phi_s + 0.5 * np.log(1.0 - 2.0 * M / R))

    return float(M), float(lapse)