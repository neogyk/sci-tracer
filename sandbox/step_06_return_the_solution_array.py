import numpy as np
from typing import List

def solve(N: int) -> np.ndarray:
    """
    Solve the 1D advection-diffusion equation with SUPG stabilization and
    Nitsche's weak boundary conditions.

    Parameters
    ----------
    N : int
        Number of uniform elements.

    Returns
    -------
    sol : np.ndarray
        Nodal solution array of size N+1.
    """
    # Problem parameters
    a = 200.0          # advection speed
    kappa = 1.0        # diffusion coefficient
    s_kappa = 1.0      # scaling parameter for Nitsche penalty
    C = 50.0           # Nitsche penalty constant
    gauss_points = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_weights = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    # Mesh generation
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N+1)                     # nodal coordinates
    elements = [(i, i+1) for i in range(N)]             # element connectivity

    # Global system
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # Assembly loop over elements
    for (el, (i, j)) in enumerate(elements):
        x_left = x[i]
        x_right = x[j]
        h_e = h

        # Element Peclet number and SUPG parameter
        Pe = a * h_e / (2.0 * kappa)
        tau = h_e / (2.0 * abs(a)) * (np.cosh(Pe) / np.sinh(Pe) - 1.0 / Pe)

        # Local element matrices
        A_local = np.zeros((2, 2))
        b_local = np.zeros(2)

        # Gauss quadrature over reference element [-1, 1]
        for gp, w in zip(gauss_points, gauss_weights):
            # Physical coordinate
            xg = 0.5 * (x_left + x_right) + 0.5 * h_e * gp

            # Shape function values and derivatives
            # omega1 corresponds to node j (right), omega2 to node i (left)
            # Here: N2 for left node, N1 for right node
            N_left = (x_right - xg) / h_e      # shape function at left node i
            N_right = (xg - x_left) / h_e      # shape function at right node j
            dN_left = -1.0 / h_e               # derivative of N_left
            dN_right = 1.0 / h_e               # derivative of N_right

            # Source term
            source = 12.0 * xg**2

            # Loop over test functions (shape functions)
            for test_idx, (N_test, dN_test) in enumerate(
                [(N_left, dN_left), (N_right, dN_right)]):
                # Standard advection-diffusion contribution
                # (a * dN_trial/dx, N_test) + (kappa * dN_trial/dx, dN_test/dx)
                # SUPG contribution: tau * a^2 * dN_trial/dx * dN_test/dx
                # Load: source * N_test + tau * a * source * dN_test/dx

                # Contributions from each trial function
                for trial_idx, (N_trial, dN_trial) in enumerate(
                    [(N_left, dN_left), (N_right, dN_right)]):

                    # Stiffness integrand
                    stiff_integrand = (a * dN_trial * N_test +
                                       kappa * dN_trial * dN_test +
                                       tau * a**2 * dN_trial * dN_test)
                    A_local[test_idx, trial_idx] += w * stiff_integrand * (h_e/2.0)  # Jacobian det = h_e/2

                # Load integrand
                load_integrand = source * N_test + tau * a * source * dN_test
                b_local[test_idx] += w * load_integrand * (h_e/2.0)

        # Assembly into global matrix
        for test_idx, global_row in enumerate([i, j]):
            for trial_idx, global_col in enumerate([i, j]):
                A[global_row, global_col] += A_local[test_idx, trial_idx]
            b[global_row] += b_local[test_idx]

    # Nitsche's method for weak Dirichlet boundary condition at x=1 (node N)
    V_kappa = C / h * (1.0 + abs(s_kappa))
    A[N, N] += V_kappa
    b[N] += V_kappa * 1.0   # Dirichlet value u(1)=1

    # Solve linear system
    sol = np.linalg.solve(A, b)
    return sol