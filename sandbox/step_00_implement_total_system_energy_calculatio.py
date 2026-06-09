import numpy as np

def E_system(positions: np.ndarray, L: float, sigma: float, epsilon: float) -> float:
    total = 0.0
    N = len(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = dist(positions[i], positions[j], L)
            total += E_ij(r_ij, sigma, epsilon)
    return total