import numpy as np

def init_AB(N: int, L: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create complex tridiagonal matrices A and B of size (N-1)x(N-1)
    for the Crank-Nicolson method for the 1D Schrödinger equation.

    Parameters:
        N (int): Number of spatial grid points (N > 1).
        L (float): Length of spatial domain (L > 0).
        h (float): Time step size (h > 0).

    Returns:
        A, B (np.ndarray): Complex (N-1)x(N-1) tridiagonal matrices.
    """
    # Physical constants
    hbar = 1.0545718e-34  # reduced Planck constant (J·s)
    m = 9.109e-31         # electron mass (kg)

    # Spatial step size
    a = L / N

    # Constant factor
    C = 1j * hbar / (4 * m * a**2)

    # Diagonal and off-diagonal entries
    diagA = 1 + 2 * h * C
    offA  = -h * C
    diagB = 1 - 2 * h * C
    offB  = h * C

    # Build matrices
    A = (np.eye(N-1) * diagA +
         np.eye(N-1, k=1) * offA +
         np.eye(N-1, k=-1) * offA)
    B = (np.eye(N-1) * diagB +
         np.eye(N-1, k=1) * offB +
         np.eye(N-1, k=-1) * offB)

    return A, B