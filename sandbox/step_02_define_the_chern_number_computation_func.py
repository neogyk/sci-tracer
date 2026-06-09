import numpy as np
import cmath
from math import pi

def compute_chern_number(delta, a, t1, t2, phi, m):
    """Compute Chern number for the Haldane model using discretized BZ.

    Parameters:
        delta (float): Discretization step size for k-space.
        a (float): Lattice constant.
        t1 (float): Nearest-neighbor hopping.
        t2 (float): Next-nearest-neighbor hopping.
        phi (float): Phase for next-nearest-neighbor hopping.
        m (float): Mass term (onsite energy difference).

    Assumes haldane_hamiltonian(kx, ky, a, t1, t2, phi, m) is defined.
    Returns:
        float: Chern number (should be integer).
    """
    # Define BZ extents
    kx0 = 2 * np.sqrt(3) * pi / (3 * a)
    ky0 = 4 * pi / (3 * a)
    Nk = int(round(kx0 / delta))
    Nk_y = int(round(ky0 / delta))
    # Force even? Not necessary, but ensure integer
    kx_vals = np.linspace(0, kx0, Nk, endpoint=False)
    ky_vals = np.linspace(0, ky0, Nk_y, endpoint=False)

    # Compute eigenvectors for the lowest band
    eigvecs = np.zeros((Nk, Nk_y, 2), dtype=complex)
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = haldane_hamiltonian(kx, ky, a, t1, t2, phi, m)
            evals, evecs = np.linalg.eigh(H)
            eigvecs[i, j, :] = evecs[:, 0]  # Lowest energy band

    # Compute U(1) link variables
    Ux = np.zeros((Nk, Nk_y), dtype=complex)
    Uy = np.zeros((Nk, Nk_y), dtype=complex)
    for i in range(Nk):
        for j in range(Nk_y):
            ip = (i + 1) % Nk
            jp = (j + 1) % Nk_y
            # Ux for step in kx direction
            overlap_x = np.vdot(eigvecs[i, j, :], eigvecs[ip, j, :])
            Ux[i, j] = overlap_x / np.abs(overlap_x)
            # Uy for step in ky direction
            overlap_y = np.vdot(eigvecs[i, j, :], eigvecs[i, jp, :])
            Uy[i, j] = overlap_y / np.abs(overlap_y)

    # Compute flux through each plaquette and sum
    chern = 0.0
    for i in range(Nk):
        for j in range(Nk_y):
            ip = (i + 1) % Nk
            jp = (j + 1) % Nk_y
            # Plaquette product: Ux(i,j) * Uy(ip,j) * Ux(i,jp)^{-1} * Uy(i,j)^{-1}
            product = Ux[i, j] * Uy[ip, j] * np.conj(Ux[i, jp]) * np.conj(Uy[i, j])
            F = cmath.log(product)  # imaginary part is the flux
            chern += F
    chern /= (2 * pi * 1j)
    return np.real(chern)