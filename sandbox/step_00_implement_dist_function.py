import numpy as np

def dist(r1: np.ndarray, r2: np.ndarray, L: float) -> float:
    """
    Compute the minimum image distance between two points in a periodic box.
    
    Parameters:
        r1, r2: numpy arrays of shape (3,) representing positions.
        L: float, box length (positive).
    
    Returns:
        Minimum image distance as a float.
    """
    delta = r1 - r2
    delta = delta - L * np.round(delta / L)
    return np.linalg.norm(delta)