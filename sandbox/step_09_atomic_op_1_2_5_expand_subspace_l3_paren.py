import numpy as np

def expand_subspace(V, correction_vectors):
    """
    Orthonormalise correction vectors and append them to the basis V.

    Parameters
    ----------
    V : ndarray, shape (n, m)
        Current orthonormal basis (columns are basis vectors).
    correction_vectors : ndarray, shape (n, k)
        Each column is a correction vector to be orthogonalised and added.

    Returns
    -------
    V_new : ndarray, shape (n, m + number_added)
        Updated orthonormal basis with new columns appended.
    """
    for i in range(correction_vectors.shape[1]):
        t = correction_vectors[:, i]
        # Orthogonalise against current basis via Gram-Schmidt
        t = t - V @ (V.T @ t)
        t_norm = np.linalg.norm(t)
        if t_norm > 1e-12:  # avoid adding a negligible vector
            t = t / t_norm
            V = np.hstack((V, t.reshape(-1, 1)))
    return V