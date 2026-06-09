import numpy as np
import cmath
import math
from typing import Tuple

# Physical parameters
a: float = 1.0          # lattice constant
t1: float = 1.0         # nearest-neighbor hopping
t2: float = 0.3         # next-nearest-neighbor hopping
delta: float = 0.1      # discretization step for BZ grid
N: int = 20             # number of points for m/phi sweep

def haldane_hamiltonian(kx: float, ky: float, a: float, t1: float, t2: float, phi: float, m: float) -> np.ndarray:
    """
    Returns the 2x2 Bloch Hamiltonian for the Haldane model.
    The honeycomb lattice with nearest-neighbor vectors d1, d2, d3
    and next-nearest-neighbor vectors (connecting same sublattice).
    """
    # Nearest-neighbor vectors (in units of a)
    d1 = np.array([0, a / math.sqrt(3)])
    d2 = np.array([a/2, -a/(2*math.sqrt(3))])
    d3 = np.array([-a/2, -a/(2*math.sqrt(3))])

    # Next-nearest-neighbor vectors (connecting A-A or B-B)
    a1 = np.array([math.sqrt(3)*a, 0])
    a2 = np.array([math.sqrt(3)*a/2, 3*a/2])
    a3 = np.array([-math.sqrt(3)*a/2, 3*a/2])

    # On-site sublattice potential (staggered)
    H = np.array([[m, 0], [0, -m]], dtype=complex)

    # Nearest-neighbor hopping
    f_k = np.exp(1j * np.dot(d1, [kx, ky])) + np.exp(1j * np.dot(d2, [kx, ky])) + np.exp(1j * np.dot(d3, [kx, ky]))
    H[0, 1] = -t1 * f_k
    H[1, 0] = -t1 * np.conj(f_k)

    # Next-nearest-neighbor hopping with complex phase phi
    # For sublattice A: hopping t2*exp(i*phi) along a1, a2, a3
    # For sublattice B: hopping t2*exp(-i*phi) along -a1, -a2, -a3
    # Matrix elements: H_A A = t2 * (exp(i*phi)*sum(...) + exp(-i*phi)*sum(conj))
    sum_A = np.exp(1j * np.dot(a1, [kx, ky])) + np.exp(1j * np.dot(a2, [kx, ky])) + np.exp(1j * np.dot(a3, [kx, ky]))
    sum_B = np.exp(-1j * np.dot(a1, [kx, ky])) + np.exp(-1j * np.dot(a2, [kx, ky])) + np.exp(-1j * np.dot(a3, [kx, ky]))
    H[0, 0] += t2 * (np.exp(1j*phi) * sum_A + np.exp(-1j*phi) * sum_B)
    H[1, 1] += t2 * (np.exp(-1j*phi) * sum_A + np.exp(1j*phi) * sum_B)

    return H

def compute_chern_number(delta: float, a: float, t1: float, t2: float, phi: float, m: float) -> float:
    """
    Compute the Chern number of the lower band using the discretized
    Brillouin zone method (Fukui, Hatsugai, Suzuki 2005).
    """
    # Brillouin zone ranges (reduced BZ for honeycomb lattice)
    kx0 = 2 * math.sqrt(3) * math.pi / (3 * a)
    ky0 = 4 * math.pi / (3 * a)

    # Number of grid points
    Nk = int(round(kx0 / delta))
    Nk_y = int(round(ky0 / delta))

    # Generate k-points
    kx_vals = np.linspace(0, kx0, Nk, endpoint=False)
    ky_vals = np.linspace(0, ky0, Nk_y, endpoint=False)

    # Compute eigenvectors for the lower band (index 0)
    eigvecs = np.zeros((Nk, Nk_y, 2), dtype=complex)
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = haldane_hamiltonian(kx, ky, a, t1, t2, phi, m)
            evals, evecs = np.linalg.eigh(H)
            eigvecs[i, j, :] = evecs[:, 0]  # lowest band

    # Compute U(1) link variables Ux and Uy
    Ux = np.zeros((Nk, Nk_y), dtype=complex)
    Uy = np.zeros((Nk, Nk_y), dtype=complex)
    for i in range(Nk):
        for j in range(Nk_y):
            ip = (i + 1) % Nk
            jp = (j + 1) % Nk_y
            # Ux: link in x-direction
            overlap = np.vdot(eigvecs[i, j, :], eigvecs[ip, j, :])
            Ux[i, j] = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
            # Uy: link in y-direction
            overlap = np.vdot(eigvecs[i, j, :], eigvecs[i, jp, :])
            Uy[i, j] = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0

    # Sum flux over all plaquettes
    chern = 0.0
    for i in range(Nk):
        for j in range(Nk_y):
            ip = (i + 1) % Nk
            jp = (j + 1) % Nk_y
            # Wilson loop (plaquette product)
            W = Ux[i, j] * Uy[ip, j] * np.conj(Ux[i, jp]) * np.conj(Uy[i, j])
            # Flux = log(W)
            flux = cmath.log(W)
            chern += flux

    # Chern number = total flux / (2π i)
    chern /= (2 * math.pi * 1j)
    return np.real(chern)

# Perform sweep over m/t2 ratio and phi
m_ratio_values = np.linspace(-3.0, 3.0, N)
phi_values = np.linspace(-np.pi, np.pi, N)
m_values = m_ratio_values * t2

results = np.zeros((N, N), dtype=float)

for i in range(N):
    for j in range(N):
        results[i, j] = compute_chern_number(delta, a, t1, t2, phi_values[j], m_values[i])

# The variable 'results' now contains the Chern numbers for each (m, phi) pair.
# You may optionally save or plot them.