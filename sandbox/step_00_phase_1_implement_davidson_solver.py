import numpy as np

def davidson_solver(
    matrixA: np.ndarray,
    num_eigenvalues: int,
    threshold: float,
    max_iter: int = 1000
) -> np.ndarray:
    """
    Compute the smallest eigenvalues of a symmetric matrix using the Davidson method.

    Parameters
    ----------
    matrixA : np.ndarray
        Symmetric matrix of shape (n, n).
    num_eigenvalues : int
        Number of smallest eigenvalues to compute (must be < n).
    threshold : float
        Convergence threshold for the residual norm.
    max_iter : int, optional
        Maximum number of Davidson iterations (default 1000).

    Returns
    -------
    np.ndarray
        Converged eigenvalues of length num_eigenvalues, sorted in ascending order.
    """
    n = matrixA.shape[0]
    if num_eigenvalues >= n:
        raise ValueError("num_eigenvalues must be less than the matrix dimension.")
    if not np.allclose(matrixA, matrixA.T):
        raise ValueError("matrixA must be symmetric.")

    np.random.seed(42)  # deterministic initial basis

    # Initial subspace: random orthonormal basis of size num_eigenvalues+2 (or n if smaller)
    basis_size = min(num_eigenvalues + 2, n)
    V, _ = np.linalg.qr(np.random.randn(n, basis_size))

    # Main iteration
    converged = []          # list of converged eigenvalues
    iteration = 0
    while len(converged) < num_eigenvalues and iteration < max_iter:
        iteration += 1

        # Project matrix onto subspace: H = V^T A V
        H = V.T @ matrixA @ V

        # Solve the projected eigenproblem
        thetas, ys = np.linalg.eigh(H)   # eigenvalues and eigenvectors (shape (m,), (m,m))
        # thetas are sorted in ascending order by eigh

        # Compute Ritz vectors and residuals
        U = V @ ys                         # shape (n, m)
        residuals = matrixA @ U - thetas * U   # shape (n, m)

        # Compute residual norms
        residual_norms = np.linalg.norm(residuals, axis=0)

        # Identify converged Ritz pairs (those with norm < threshold)
        for i, theta in enumerate(thetas):
            if residual_norms[i] < threshold and theta not in converged:
                converged.append(theta)

        # If enough converged, break early (but we still need to expand for unconverged)
        if len(converged) >= num_eigenvalues:
            converged = sorted(converged)[:num_eigenvalues]
            break

        # Compute correction vectors for unconverged eigenvalues
        diagA = np.diag(matrixA)
        new_vectors = []
        for i, theta in enumerate(thetas):
            if residual_norms[i] >= threshold and V.shape[1] < n:
                # Preconditioned correction: t = - (diag(A) - theta*I)^{-1} r
                # Avoid division by zero by adding small epsilon
                denom = diagA - theta
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)  # safeguard
                t = -residuals[:, i] / denom

                # Orthogonalize t against current V (Gram-Schmidt)
                t = t - V @ (V.T @ t)
                norm_t = np.linalg.norm(t)
                if norm_t > 1e-12:
                    new_vectors.append(t / norm_t)

        # Expand subspace with new orthonormal vectors
        if new_vectors:
            V = np.hstack((V, np.column_stack(new_vectors)))
        else:
            # No new vectors; break to avoid infinite loop
            break

    if len(converged) < num_eigenvalues:
        raise RuntimeError(
            f"Davidson method did not converge {num_eigenvalues} eigenvalues within "
            f"{max_iter} iterations."
        )

    # Return the smallest requested eigenvalues
    return np.array(sorted(converged)[:num_eigenvalues])