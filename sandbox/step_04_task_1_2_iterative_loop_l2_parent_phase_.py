import numpy as np

def davidson_iteration_loop(
    V: np.ndarray,
    A: np.ndarray,
    num_eigenvalues: int,
    threshold: float,
    max_iter: int = 1000
) -> np.ndarray:
    """
    Perform the core Davidson iteration loop until the requested number of
    eigenvalues converge.

    Parameters:
        V: Initial orthonormal basis (n x m, m >= num_eigenvalues+2)
        A: Symmetric matrix (n x n)
        num_eigenvalues: Number of smallest eigenvalues to compute
        threshold: Residual norm tolerance for convergence
        max_iter: Maximum number of iterations

    Returns:
        current_eigenvalues: 1D numpy array of length num_eigenvalues
                             containing the converged eigenvalues
    """
    n, m = V.shape
    diag_A = np.diag(A)

    converged: list[float] = []          # list of converged eigenvalue values
    iteration = 0

    while len(converged) < num_eigenvalues and iteration < max_iter:
        # 1. Project A onto current subspace
        H = V.T @ A @ V

        # 2. Solve eigenproblem of H (sorted ascending)
        theta, y = np.linalg.eigh(H)
        # theta and y are already sorted by eigh

        # 3. Compute Ritz vectors and residuals
        U = V @ y                              # (n, m)
        residuals = A @ U - theta * U          # (n, m)
        res_norms = np.linalg.norm(residuals, axis=0)   # (m,)

        # 4. Check convergence for each Ritz pair
        #    Only consider the smallest num_eigenvalues (though we check all)
        for i in range(m):
            if res_norms[i] < threshold:
                # Avoid duplicates (if the same eigenvalue appears again)
                if theta[i] not in converged:
                    converged.append(theta[i])
                    # Optional: lock eigenvector, but we keep it simple

        # Exit early if we already have enough converged eigenvalues
        if len(converged) >= num_eigenvalues:
            break

        # 5. Compute correction vectors for unconverged pairs
        new_vectors = []
        for i in range(m):
            if res_norms[i] >= threshold:
                # Preconditioner: (diag(A) - θ_i)^(-1)
                shift = theta[i]
                denom = diag_A - shift
                # Avoid division by zero (or near-zero)
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                t = -residuals[:, i] / denom

                # Orthogonalise against current V (Gram-Schmidt)
                t = t - V @ (V.T @ t)
                t_norm = np.linalg.norm(t)
                if t_norm > 1e-12:
                    new_vectors.append(t / t_norm)

        # 6. Expand the subspace with new orthonormal vectors
        if new_vectors:
            V = np.hstack((V, np.column_stack(new_vectors)))

        iteration += 1

        # Shrink subspace if it becomes too large (optional, but keep for stability)
        # max subspace size = min(2*num_eigenvalues, n) – here we do nothing

    # 7. Return the smallest converged eigenvalues (sorted)
    converged_sorted = sorted(converged)[:num_eigenvalues]
    current_eigenvalues = np.array(converged_sorted, dtype=float)

    return current_eigenvalues