import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence D(p||q) = sum_i p_i * log2(p_i / q_i).
    Assumes p and q are 1-D numpy arrays of the same length, valid probability distributions,
    and q_i > 0 for all i.
    """
    return np.sum(p * np.log2(p / q))