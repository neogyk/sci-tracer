import numpy as np
from typing import List, Optional

def davidson_eigenvalues(
    A: np.ndarray,
    num_eigenvalues: int,
    threshold: float
) -> np.ndarray:
    """
    Compute the smallest eigenvalues of a symmetric matrix A using
    the Davidson iterative method with deflation.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix of shape (n, n).
    num_eigenvalues : int
        Number of smallest eigenvalues to compute.
    threshold : float
        Convergence tolerance for the residual norm.

    Returns
    -------
    eigvals : ndarray
        Array of length num_eigenvalues containing the computed lowest eigenvalues.
    """
    n = A.shape[0]
    if num_eigenvalues <= 0:
        return np.array([])
    if num_eigenvalues > n:
        num_eigenvalues = n

    # Storage for converged eigenvalues and vectors
    eigvals: List[float] = []
    eigvecs: List[np.ndarray] = []   # orthonormal converged eigenvectors

    for k in range(num_eigenvalues):
        # ---------- Find the (k+1)-th eigenvalue ----------
        # Initial vector: random, orthogonal to all converged eigenvectors
        v0 = np.random.randn(n)
        # Orthogonalize against converged eigenvectors
        for ev in eigvecs:
            v0 = v0 - np.dot(v0, ev) * ev
        v0 = v0 / np.linalg.norm(v0)

        # Davidson subspace for the current eigenvalue
        V = v0.reshape(n, 1)

        # Iteration
        while True:
            # Rayleigh-Ritz on current subspace
            H = V.T @ A @ V
            H_eigvals, H_eigvecs = np.linalg.eigh(H)

            # Smallest Ritz pair
            theta = H_eigvals[0]
            y = H_eigvecs[:, 0]
            u = V @ y

            # Residual
            r = A @ u - theta * u
            resnorm = np.linalg.norm(r)

            if resnorm < threshold:
                # Converged
                eigvals.append(theta)
                eigvecs.append(u)
                break

            # Preconditioner: diag(A) - theta * I
            diagA = np.diag(A)
            M = diagA - theta
            # Avoid division by zero
            M = np.where(np.abs(M) < 1e-12, 1e-12, M)
            t = r / M

            # Orthogonalize correction against subspace and converged eigenvectors
            for ev in eigvecs:
                t = t - np.dot(t, ev) * ev
            for i in range(V.shape[1]):
                t = t - np.dot(t, V[:, i]) * V[:, i]

            t_norm = np.linalg.norm(t)
            if t_norm < 1e-12:
                # Subspace cannot be expanded further; accept current approximation
                eigvals.append(theta)
                eigvecs.append(u)
                break

            t = t / t_norm
            V = np.hstack([V, t.reshape(n, 1)])

            # Optional restart if subspace size exceeds a limit
            if V.shape[1] > min(n, 20):
                # Restart with the current Ritz vector u
                V = u.reshape(n, 1)
                # also orthonormalize (should already be unit)
                V = V / np.linalg.norm(V)

    return np.array(eigvals[:num_eigenvalues])