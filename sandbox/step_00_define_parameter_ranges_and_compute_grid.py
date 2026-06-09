import numpy as np
from math import pi

def compute_chern_number_grid(delta: float, a: float, t1: float, t2: float, N: int):
    """
    Compute Chern number grid over m/t2 and phi parameters.
    
    Parameters:
        delta (float): staggered sublattice potential
        a (float): lattice constant
        t1 (float): nearest-neighbor hopping
        t2 (float): next-nearest-neighbor hopping
        N (int): number of points per axis
        
    Returns:
        results (np.ndarray): 2D array of Chern numbers
        m_values (np.ndarray): array of m/t2 values
        phi_values (np.ndarray): array of phi values
    """
    m_values = np.linspace(-6, 6, N)
    phi_values = np.linspace(-pi, pi, N)
    results = np.zeros((N, N))
    for i, m_ratio in enumerate(m_values):
        m = m_ratio * t2
        for j, phi in enumerate(phi_values):
            results[i, j] = compute_chern_number(delta, a, t1, t2, phi, m)
    return results, m_values, phi_values