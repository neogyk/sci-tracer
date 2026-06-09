import numpy as np

def f_Schrod(energy, l, r_grid):
    Z = 1
    f_r = l*(l+1) / r_grid**2 - 2*Z / r_grid - energy
    return f_r