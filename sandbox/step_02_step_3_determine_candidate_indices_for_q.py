import numpy as np
from typing import List, Tuple

def candidate_indices(
    Q: np.ndarray,
    B: np.ndarray,
    tolerance: float,
    max_index: int = 10
) -> List[Tuple[int, int, int]]:
    """
    Generate candidate (h, k, l) indices for a given Q vector.
    The reciprocal lattice vector G = B @ (h, k, l) must have magnitude
    within tolerance of |Q|.

    Parameters
    ----------
    Q : np.ndarray
        Momentum transfer vector (3,).
    B : np.ndarray
        Crystal B matrix (3,3) that maps fractional indices to Cartesian.
    tolerance : float
        Absolute tolerance for the difference in magnitudes.
    max_index : int, optional
        Maximum absolute value for h, k, l (default 10).

    Returns
    -------
    List[Tuple[int, int, int]]
        List of (h, k, l) triples satisfying the magnitude condition.
    """
    target_magnitude = np.linalg.norm(Q)
    candidates = []
    # Iterate over all integer triples within the specified range
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                # Skip the origin (0,0,0) as it has zero magnitude
                if h == 0 and k == 0 and l == 0:
                    continue
                # Compute reciprocal lattice vector in Cartesian coordinates
                hkl = np.array([h, k, l], dtype=float)
                G = B @ hkl
                # Check magnitude match
                diff = abs(np.linalg.norm(G) - target_magnitude)
                if diff <= tolerance:
                    candidates.append((h, k, l))
    return candidates

# Example usage (commented out):
# B = np.load('B_matrix.npy')
# Q1 = np.array([...])
# Q2 = np.array([...])
# tolerance = 0.01
# H1 = candidate_indices(Q1, B, tolerance)
# H2 = candidate_indices(Q2, B, tolerance)