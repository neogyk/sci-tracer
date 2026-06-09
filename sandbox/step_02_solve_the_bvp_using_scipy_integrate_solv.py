from scipy.integrate import solve_bvp
import numpy as np

def solve_bvp_fiber(L, R1, R2, Ppl, Ppr, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s):
    """
    Solve the two-level Yb-doped fiber laser BVP using solve_bvp.
    
    Parameters:
        L (float): Fiber length
        R1, R2 (float): Mirror reflectivities at z=0 and z=L
        Ppl, Ppr (float): Left/right pump powers
        Pssat, Ppsat (float): Signal and pump saturation powers
        N (float): Yb ion density
        sigma_ap, sigma_ep: Absorption/emission cross-sections for pump
        sigma_as, sigma_es: Absorption/emission cross-sections for signal
        gamma_p, gamma_s: Overlap factors
        alpha_p, alpha_s: Loss coefficients
        
    Returns:
        sol: BVP solution object from scipy.integrate.solve_bvp
    """
    # Initial mesh
    N_nodes = 100
    z_init = np.linspace(0, L, N_nodes)
    # Initial guess: forward pump constant, others zero
    y_init = np.zeros((4, N_nodes))
    y_init[0, :] = Ppl
    
    # Define ODE and BC wrappers to pass parameters
    def ode_func(z, y):
        return f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s)
    
    def bc_func(ya, yb):
        return bc(ya, yb, R1, R2, Ppl, Ppr)
    
    # Solve BVP
    sol = solve_bvp(ode_func, bc_func, z_init, y_init, max_nodes=1000, tol=1e-6)
    
    if not sol.success:
        raise RuntimeError('BVP solver did not converge: ' + sol.message)
    
    return sol