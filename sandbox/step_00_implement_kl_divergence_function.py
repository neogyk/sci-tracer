import numpy as np

def KL_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p)
    q = np.asarray(q)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))