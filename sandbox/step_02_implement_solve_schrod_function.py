import numpy as np
from scipy.integrate import simpson

def Solve_Schrod(x, En, u_b, up_b, step):
    u = Numerov(x, En, u_b, up_b, step)
    norm = simpson(u**2, x)
    u_norm = u / np.sqrt(norm)
    return u_norm