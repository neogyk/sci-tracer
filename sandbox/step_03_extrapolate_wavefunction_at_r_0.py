import numpy as np

def extrapolate_wf(r_grid: np.ndarray, u: np.ndarray, l: int) -> float:
    """
    Extrapolate the radial wavefunction u(r) / r^l to r = 0 using linear
    extrapolation from the first two grid points.

    Parameters:
        r_grid: array of radial coordinates (should be strictly increasing)
        u: radial wavefunction values u(r) (same length as r_grid)
        l: angular momentum quantum number

    Returns:
        Extrapolated value of u(r) / r^l at r = 0.
    """
    # u / r^l for the first two points (r > 0)
    u_scaled = u / (r_grid ** l)
    r0, r1 = r_grid[0], r_grid[1]
    u_scaled0, u_scaled1 = u_scaled[0], u_scaled[1]
    # Linear extrapolation to r = 0:
    # y = y0 + (y1 - y0) * (x - x0) / (x1 - x0), with x = 0
    val_at_0 = u_scaled0 + (u_scaled1 - u_scaled0) * (0.0 - r0) / (r1 - r0)
    return val_at_0