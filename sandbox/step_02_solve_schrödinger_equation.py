import numpy as np
from scipy import integrate

def compute_Schrod(r_grid, energy, l):
    step = r_grid[0] - r_grid[1]
    f_vals = np.array([f_Schrod(r, energy, l) for r in r_grid])
    u = numerov(f_vals, 0.0, -1e-7, step)
    # Normalize using Simpson's rule from large r to 0
    norm = integrate.simpson(u**2, r_grid)
    u /= np.sqrt(norm)
    return u