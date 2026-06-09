import numpy as np

def branch(weight: np.ndarray) -> np.ndarray:
    """
    Perform systematic resampling for a set of weights.

    Parameters
    ----------
    weight : np.ndarray
        1D array of nonnegative weights of shape (nconfig,). At least one > 0.

    Returns
    -------
    np.ndarray
        1D array of indices of shape (nconfig,) representing resampled indices.
    """
    n = len(weight)
    # Normalize weights to sum to 1
    cumsum = np.cumsum(weight / weight.sum())
    # Generate low-discrepancy offset in [0, 1/n)
    u = np.random.uniform(0, 1.0 / n)
    # Prepare output array
    new_indices = np.empty(n, dtype=int)
    j = 0
    for k in range(n):
        target = u + k / n
        while cumsum[j] < target:
            j += 1
        new_indices[k] = j
    return new_indices