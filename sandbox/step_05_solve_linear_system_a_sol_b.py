import numpy as np

def solve(N):
    # Parameters
    a = 200.0
    kappa = 1.0
    C = 50.0
    s_kappa = 1.0
    g = 1.0  # Dirichlet value at x=1

    # Gauss quadrature (3 points)
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])

    # Mesh
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # Element loop
    for el in range(N):
        i = el          # left node index
        j = el + 1      # right node index
        x_i = x[i]
        x_j = x[j]

        # Element Peclet number and SUPG parameter
        Pe = a * h / (2.0 * kappa)
        tau = h / (2.0 * abs(a)) * (np.cosh(Pe) / np.sinh(Pe) - 1.0 / Pe)

        # Local matrices
        K_loc = np.zeros((2,2))
        f_loc = np.zeros(2)

        for q in range(3):
            xi = gauss_points[q]
            w = gauss_weights[q]

            # Physical coordinate
            xg = 0.5 * (x_i + x_j) + 0.5 * h * xi   # mapping from [-1,1] to [x_i,x_j]

            # Shape functions and derivatives
            N1 = (x_j - xg) / h
            N2 = (xg - x_i) / h
            dN1 = -1.0 / h
            dN2 = 1.0 / h

            N_vec = np.array([N1, N2])
            dN_vec = np.array([dN1, dN2])

            # Source term: 12*x^2
            source = 12.0 * xg**2

            # Jacobian determinant for mapping: dx/dxi = h/2
            detJ = h / 2.0

            # Contributions to K_loc
            for i_local in range(2):
                for j_local in range(2):
                    K_loc[i_local, j_local] += w * detJ * (
                        kappa * dN_vec[i_local] * dN_vec[j_local]
                        + a * dN_vec[i_local] * N_vec[j_local]
                        + tau * a**2 * dN_vec[i_local] * dN_vec[j_local]
                    )

            # Contributions to f_loc
            for i_local in range(2):
                f_loc[i_local] += w * detJ * (
                    N_vec[i_local] * source
                    + tau * a * dN_vec[i_local] * source
                )

        # Assembly
        A[i, i] += K_loc[0, 0]
        A[i, j] += K_loc[0, 1]
        A[j, i] += K_loc[1, 0]
        A[j, j] += K_loc[1, 1]
        b[i] += f_loc[0]
        b[j] += f_loc[1]

    # Nitsche method at x=1 (node N)
    V_kappa = C / h * (1 + abs(s_kappa))
    A[N, N] += V_kappa
    b[N] += V_kappa * g

    # Solve linear system
    sol = np.linalg.solve(A, b)
    return sol