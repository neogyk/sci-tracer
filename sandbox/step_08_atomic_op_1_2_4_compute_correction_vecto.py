import numpy as np

def compute_correction_vectors(matrixA, residuals, theta, threshold):
    """
    Compute preconditioned correction vectors for all unconverged residuals.

    Parameters
    ----------
    matrixA : np.ndarray, shape (n, n)
        Symmetric matrix.
    residuals : np.ndarray, shape (n, m)
        Residual vectors for each Ritz pair (columns).
    theta : np.ndarray, shape (m,)
        Ritz values corresponding to residuals.
    threshold : float
        Convergence tolerance on residual norm.

    Returns
    -------
    t_vectors : np.ndarray, shape (n, k)
        Correction vectors for unconverged residuals (columns).
    unconverged_indices : np.ndarray, shape (k,)
        Indices of unconverged residuals.
    """
    n = matrixA.shape[0]
    diagA = np.diag(matrixA)
    # Compute residual norms
    norms = np.linalg.norm(residuals, axis=0)
    unconverged = np.where(norms >= threshold)[0]
    k = len(unconverged)
    if k == 0:
        return np.zeros((n, 0)), unconverged

    t_vectors = np.zeros((n, k))
    for j, i in enumerate(unconverged):
        denom = diagA - theta[i]
        # Avoid division by zero by adding a small epsilon where denom is near zero
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        t_vectors[:, j] = -residuals[:, i] / denom
    return t_vectors, unconverged