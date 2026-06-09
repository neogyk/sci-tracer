def compute_charge_density(r_grid, bound_states, nmax):
    """
    Sum over occupied states (n,l) squared wavefunctions times occupation
    (2 per l for each n) to get radial charge density.

    Parameters:
    r_grid (np.ndarray): Radial grid points.
    bound_states (list of tuple): List of (l, energy) for bound states.
    nmax (int): Maximum principal quantum number (not used in computation).

    Returns:
    np.ndarray: Array of radial charge density (4π r^2 times electron density).
    """
    import numpy as np
    charge = np.zeros(len(r_grid))
    # Each state has spin degeneracy 2
    for l, energy in bound_states:
        u = compute_Schrod(r_grid, energy, l)
        charge += 2 * u**2
    return charge