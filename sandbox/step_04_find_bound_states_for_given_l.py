import scipy.optimize as optimize

def find_bound_states(r_grid, l, energy_grid):
    """
    Use shooting method and scipy.optimize.brentq to locate energies where u(b)=0.
    Search over energy_grid. Return list of (l, E) for up to 10 bound states.

    Parameters:
    r_grid : array-like
        Radial grid points (assumed decreasing from large r to 0 or vice versa).
    l : int
        Angular momentum quantum number.
    energy_grid : array-like
        Energy values to search over (should span the bound state region).

    Returns:
    list of tuples (l, energy)
    """
    bound = []

    def shoot(energy):
        u = compute_Schrod(r_grid, energy, l)
        return u[-1]  # u at outermost r (should be 0 for bound state)

    for i in range(len(energy_grid) - 1):
        e1 = energy_grid[i]
        e2 = energy_grid[i + 1]
        f1 = shoot(e1)
        f2 = shoot(e2)
        if f1 * f2 < 0:
            try:
                root = optimize.brentq(shoot, e1, e2, xtol=1e-6)
                bound.append((l, root))
                if len(bound) >= 10:
                    break
            except ValueError:
                # Brentq may fail if function values are not of opposite sign (shouldn't happen)
                pass
    return bound