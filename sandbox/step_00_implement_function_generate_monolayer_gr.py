import numpy as np
from typing import List

def generate_monolayer_graphene(s: float, a: float, z: float, n: int) -> np.ndarray:
    """
    Generate atomic coordinates for a monolayer graphene sheet.

    Parameters:
        s (float): Sliding distance in the y (armchair) direction.
        a (float): Carbon-carbon bond length.
        z (float): z-coordinate of the sheet.
        n (int): Number of unit cells in each direction (grid from -n to n).

    Returns:
        np.ndarray: Array of shape (N, 3) containing (x, y, z) coordinates of atoms.
    """
    sqrt3 = np.sqrt(3.0)
    # Lattice vectors
    a1 = np.array([a * sqrt3, 0.0])
    a2 = np.array([a * sqrt3 / 2.0, 1.5 * a])

    # Basis atoms in Cartesian coordinates
    basis = np.array([
        [0.0, 0.0],
        [a * sqrt3 / 2.0, a / 2.0]
    ])

    atoms: List[float] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            cell_origin = i * a1 + j * a2
            for b in basis:
                pos = cell_origin + b
                pos[1] += s  # apply sliding in y-direction
                atoms.append([pos[0], pos[1], z])

    return np.array(atoms)