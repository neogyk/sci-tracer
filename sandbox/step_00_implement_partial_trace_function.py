import numpy as np

def syspermute(X, order, dim):
    """
    Permute the order of subsystems of a density matrix.
    X: square matrix of size prod(dim)
    order: list of new ordering of subsystem indices (0..n-1)
    dim: list of subsystem dimensions
    """
    n = len(dim)
    # Reshape to tensor with row and column indices
    shape = dim + dim  # first n for row, last n for column
    X_tensor = X.reshape(shape)
    # Compute transposition order: apply same permutation to row and column indices
    new_order = order + [i + n for i in order]
    X_perm = np.transpose(X_tensor, new_order)
    # Reshape back to square matrix
    new_dim = [dim[i] for i in order]
    new_dim_prod = int(np.prod(new_dim))
    return X_perm.reshape(new_dim_prod, new_dim_prod)

def partial_trace(X, sys, dim):
    """
    Partial trace over subsystems specified in sys.
    X: square matrix (density matrix) of total size prod(dim)
    sys: list of subsystem indices to trace out (0-indexed)
    dim: list of subsystem dimensions
    """
    dims = list(dim)
    total_dim = int(np.prod(dims))
    assert X.shape == (total_dim, total_dim), "Matrix dimension mismatch"
    n = len(dims)
    keep = [i for i in range(n) if i not in sys]
    trace = [i for i in range(n) if i in sys]
    order = keep + trace
    dims_perm = [dims[i] for i in order]
    X_perm = syspermute(X, order, dims)
    dim_keep = int(np.prod(dims_perm[:len(keep)]))
    dim_trace = int(np.prod(dims_perm[len(keep):]))
    X_reshaped = X_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    result = np.trace(X_reshaped, axis1=1, axis2=3)
    return result