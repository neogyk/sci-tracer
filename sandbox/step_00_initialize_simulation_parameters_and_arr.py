import numpy as np
from typing import Tuple

def init_system(N: int, rho: float) -> np.ndarray:
    """
    Generate initial positions on an FCC lattice for N particles.
    Returns array of shape (N,3) with positions in [0, L)^3.
    """
    L = (N / rho) ** (1/3)
    # Number of unit cells along each dimension
    n_cells = int(np.ceil((N / 4) ** (1/3)))
    # FCC basis vectors
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]], dtype=float)
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    pos = np.array([ix + b[0], iy + b[1], iz + b[2]], dtype=float) / n_cells * L
                    positions.append(pos)
                    if len(positions) == N:
                        return np.array(positions)
    raise ValueError("Not enough lattice sites for given N")

def wrap(pos: np.ndarray, L: float) -> np.ndarray:
    """Apply periodic boundary conditions: wrap coordinates into [0, L)."""
    return pos - np.floor(pos / L) * L

def E_single(i: int, positions: np.ndarray, sigma: float, epsilon: float, L: float, r_c: float) -> float:
    """
    Compute energy of particle i with all other particles,
    using minimum image convention and cutoff.
    """
    total = 0.0
    pos_i = positions[i]
    N = len(positions)
    for j in range(N):
        if i == j:
            continue
        dr = positions[j] - pos_i
        # Minimum image
        dr = dr - L * np.round(dr / L)
        r = np.linalg.norm(dr)
        if r < r_c and r > 0:
            r_inv = sigma / r
            r6 = r_inv ** 6
            total += 4.0 * epsilon * (r6 ** 2 - r6)
    return total

def total_energy(positions: np.ndarray, sigma: float, epsilon: float, L: float, r_c: float) -> float:
    """Compute total potential energy of the system."""
    total = 0.0
    N = len(positions)
    for i in range(N):
        total += E_single(i, positions, sigma, epsilon, L, r_c)
    # Divide by 2 because each pair counted twice
    return total / 2.0

# ----------------------------------------------------------------------
# Simulation parameters (assumed provided as global variables)
# N, sigma, epsilon, r_c, rho, T, n_eq, n_prod, insertion_freq, move_magnitude
# ----------------------------------------------------------------------
# Example defaults for testing (you can override after code is placed)
N = 256
sigma = 1.0
epsilon = 1.0
r_c = 2.5
rho = 0.8
T = 1.5
n_eq = 1000
n_prod = 10000
insertion_freq = 100
move_magnitude = 0.1

# Initialization step
L = (N / rho) ** (1/3)
positions = init_system(N, rho)
E_array = np.zeros(n_prod)          # store total energy per production step
n_accp = 0                          # count accepted moves (will be incremented during MC)
E_total = total_energy(positions, sigma, epsilon, L, r_c)

print(f"Initialization complete: L = {L:.4f}, E_total = {E_total:.4f}")
print(f" positions shape: {positions.shape}")