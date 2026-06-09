import numpy as np
from typing import Tuple

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute D(p || q) = sum_i p_i * log2(p_i / q_i).
    Assumes p and q are valid probability distributions (non-negative, sum to 1)
    and q_i > 0 for all i.
    """
    return np.sum(p * np.log2(p / q))

def mutual_information(channel: np.ndarray, prior: np.ndarray) -> float:
    """
    Compute I(X;Y) = sum_x prior[x] * D(p(Y|X=x) || p(Y))
    where p(Y) = channel @ prior.
    channel shape: (n_outputs, n_inputs)
    """
    p_y = channel @ prior
    n_inputs = prior.shape[0]
    info = 0.0
    for i in range(n_inputs):
        info += prior[i] * kl_divergence(channel[:, i], p_y)
    return info

def blahut_arimoto(channel: np.ndarray, e: float = 1e-6) -> float:
    """
    Compute the channel capacity using the Blahut-Arimoto algorithm.
    channel: transition matrix, shape (outputs, inputs), column-stochastic.
    e: convergence tolerance.
    Returns capacity in bits.
    """
    n_inputs = channel.shape[1]
    prior = np.ones(n_inputs) / n_inputs
    rate = mutual_information(channel, prior)
    while True:
        p_y = channel @ prior
        # Compute exp( D(p(Y|X=j) || p(Y)) ) for each input j
        exp_term = np.array([np.exp(kl_divergence(channel[:, j], p_y)) for j in range(n_inputs)])
        new_prior = prior * exp_term
        new_prior /= new_prior.sum()
        new_rate = mutual_information(channel, new_prior)
        if abs(new_rate - rate) < e:
            return new_rate
        prior = new_prior
        rate = new_rate

if __name__ == "__main__":
    # Test with Binary Symmetric Channel (BSC) with crossover prob 0.1
    bsc = np.array([[0.9, 0.1],
                    [0.1, 0.9]])
    cap = blahut_arimoto(bsc, 1e-6)
    # Expected capacity: 1 - H(0.1) = 1 - (0.1*log2(1/0.1) + 0.9*log2(1/0.9))
    expected = 1 - (0.1 * np.log2(1/0.1) + 0.9 * np.log2(1/0.9))
    print(f"Computed capacity: {cap}")
    print(f"Expected capacity: {expected}")
    assert abs(cap - expected) < 1e-5, f"Capacity mismatch: {cap} vs {expected}"
    print("Test passed.")