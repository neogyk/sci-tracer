import numpy as np

# Assuming the following variables are already defined:
# N: number of elements (integer)
# h: element size (float), h = 1/N
# x: numpy array of node coordinates, shape (N+1,)
# elements: list of tuples (i, j) where j = i+1
# a, kappa: float constants
# gauss_points: list of 3 Gauss points (xi values in [-1,1])
# gauss_weights: list of corresponding weights

A = np.zeros((N+1, N+1))
b = np.zeros(N+1)

for (i, j) in elements:
    h_e = h  # since uniform mesh
    # element Peclet number
    Pe = a * h_e / (2 * kappa)
    # SUPG stabilization parameter (optimal for linear elements)
    tau = h_e / (2 * abs(a)) * (np.cosh(Pe) / np.sinh(Pe) - 1.0 / Pe)
    
    # Local element matrix (2x2) and load vector (2x1)
    Ke = np.zeros((2, 2))
    fe = np.zeros(2)
    
    # Gauss quadrature over the reference element [-1,1]
    for g in range(3):
        xi = gauss_points[g]
        wgt = gauss_weights[g]
        
        # Physical coordinate corresponding to xi
        xg = 0.5 * (x[i] + x[j]) + 0.5 * h_e * xi  # equivalent mapping
        
        # Linear shape functions and their derivatives in physical coordinates
        N1 = (x[j] - xg) / h_e
        N2 = (xg - x[i]) / h_e
        dN1 = -1.0 / h_e
        dN2 = 1.0 / h_e
        
        f_val = 12.0 * xg**2  # source term
        
        # Jacobian of the mapping from xi to x: dx = (h_e/2) dxi
        jacobian = h_e / 2.0
        
        # Loop over test functions (rows) and trial functions (columns)
        for test in range(2):
            if test == 0:
                w = N1
                dw = dN1
            else:
                w = N2
                dw = dN2
            
            for trial in range(2):
                if trial == 0:
                    u = N1
                    du = dN1
                else:
                    u = N2
                    du = dN2
                
                # Integrand: standard advection-diffusion + SUPG stabilization
                # κ dw du + w a du + τ a^2 dw du
                integrand = kappa * dw * du + w * a * du + tau * a**2 * dw * du
                Ke[test, trial] += wgt * jacobian * integrand
            
            # Load vector: f w + τ a dw f
            load_integrand = f_val * w + tau * a * dw * f_val
            fe[test] += wgt * jacobian * load_integrand
    
    # Assemble local contributions into global system
    A[i, i] += Ke[0, 0]
    A[i, j] += Ke[0, 1]
    A[j, i] += Ke[1, 0]
    A[j, j] += Ke[1, 1]
    b[i] += fe[0]
    b[j] += fe[1]