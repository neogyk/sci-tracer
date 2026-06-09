import numpy as np
from scipy.sparse import diags, csr_matrix

def construct_matrix(N_p: int, dp: float, dt: float, r: float, sig: float) -> csr_matrix:
    a = 0.5 * dt * ((r - 0.5 * sig**2) / dp - sig**2 / dp**2)
    b = 1 + dt * (sig**2 / dp**2 + r)
    c = -0.5 * dt * ((r - 0.5 * sig**2) / dp + sig**2 / dp**2)
    diagonals = [a * np.ones(N_p-3), b * np.ones(N_p-2), c * np.ones(N_p-3)]
    D = diags(diagonals, offsets=[-1, 0, 1], shape=(N_p-2, N_p-2), format='csr')
    return D