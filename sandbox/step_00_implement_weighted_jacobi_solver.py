import numpy as np
from typing import Tuple

def WJ(A: np.ndarray, b: np.ndarray, eps: float, x_true: np.ndarray, x0: np.ndarray, omega: float) -> Tuple[float, float]:
    """
    Weighted Jacobi iteration for solving linear systems.

    Parameters:
        A : coefficient matrix (n x n)
        b : right-hand side vector (n,)
        eps : convergence tolerance on iteration step size
        x_true : exact solution (for error computation)
        x0 : initial guess
        omega : relaxation parameter

    Returns:
        residual : float, final residual norm ||A x - b||
        error    : float, final error norm ||x - x_true||
    """
    x = x0.copy()
    D = np.diag(A)  # diagonal entries
    while True:
        # Weighted Jacobi update: x_new = (1-omega)*x + omega*D^{-1}*(b - R*x)
        # Equivalent to: x_new = (b - (A @ x - D * x)) * (omega / D)
        x_new = (b - (A @ x - D * x)) * (omega / D)
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
    residual = np.linalg.norm(A @ x - b)
    error = np.linalg.norm(x - x_true)
    return residual, error