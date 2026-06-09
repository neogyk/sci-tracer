import numpy as np

def solve_hartree(r_grid, charge_density):
    # Solve 1/r d^2(rV)/dr^2 = -4π ρ (radial Poisson)
    # We want V_H(r) such that U(r)=rV_H(r) satisfies U'' = -4π r ρ
    # ρ is charge density per unit volume, but here charge_density is 4π r^2 ρ (from above)
    # So U'' = -charge_density
    n = len(r_grid)
    step = r_grid[0] - r_grid[1]  # assuming uniform grid and decreasing order (large r to small)
    U = np.zeros(n)
    # Numerov integration from r=0 outward
    # At r=0, U=0, U'=0 (for a finite potential)
    U[0] = 0.0
    U[1] = 0.0  # approximate
    # Use integration scheme: U(i+1) = 2U(i) - U(i-1) - step**2 * charge_density(i)
    for i in range(1, n-1):
        U[i+1] = 2*U[i] - U[i-1] - step**2 * charge_density[i]
    # Boundary condition: at large r, V_H ~ (Z - N)/r, so U should go to constant
    return U