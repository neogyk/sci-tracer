import numpy as np

def syspermute(X: np.ndarray, perm: list[int], dim: list[int]) -> np.ndarray:
    """
    Permute the subsystems of a density matrix X.
    
    Parameters:
        X (np.ndarray): Input density matrix of shape (total_dim, total_dim)
                        where total_dim = prod(dim).
        perm (list[int]): New order of subsystems (list of indices).
        dim (list[int]): List of dimensions for each subsystem.
    
    Returns:
        np.ndarray: Permuted density matrix of shape (total_dim, total_dim).
    """
    total_dim = np.prod(dim)
    assert X.shape == (total_dim, total_dim)
    n = len(dim)
    # Reshape to (d0, d1, ..., d_{n-1}, d0, d1, ..., d_{n-1})
    # Actually we have two indices: row and column, each decomposed into subsystems.
    shape = list(dim) + list(dim)
    X_reshaped = X.reshape(shape)
    # Build the new axes order:
    # row subsystems: indices 0..n-1, column subsystems: indices n..2n-1
    # After permutation of row and column we need to bring them together.
    # We want new order: row indices according to perm, then column indices according to perm.
    # So axes: [perm[0], perm[1], ..., perm[n-1], n+perm[0], n+perm[1], ..., n+perm[n-1]]
    axes = perm + [n + p for p in perm]
    X_perm = np.transpose(X_reshaped, axes)
    # Reshape back to matrix form
    return X_perm.reshape(total_dim, total_dim)


def partial_trace(X: np.ndarray, sys: list[int], dim: list[int]) -> np.ndarray:
    """
    Compute the partial trace of a density matrix X over the subsystems specified in sys.
    
    Parameters:
        X (np.ndarray): Input density matrix of shape (total_dim, total_dim).
        sys (list[int]): List of subsystem indices to trace out.
        dim (list[int]): List of dimensions for each subsystem.
    
    Returns:
        np.ndarray: Reduced density matrix of the remaining subsystems.
    """
    total_dim = np.prod(dim)
    assert X.shape == (total_dim, total_dim)
    # Determine subsystems to keep
    n = len(dim)
    keep = [i for i in range(n) if i not in sys]
    # New order: subsystems to trace out first, then keep
    order = sys + keep
    X_perm = syspermute(X, order, dim)
    d_trace = int(np.prod([dim[i] for i in sys]))
    d_keep = int(np.prod([dim[i] for i in keep]))
    # Reshape to (d_trace, d_keep, d_trace, d_keep)
    X_reshaped = X_perm.reshape(d_trace, d_keep, d_trace, d_keep)
    # Trace over the first and third axes (subsystems to discard)
    result = np.trace(X_reshaped, axis1=0, axis2=2)
    return result