from typing import Tuple
import numpy as np

def derive_coordinates(theta: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Derive coordinates for points A' and B' after a rotation by angle theta.

    Args:
    theta (float): The angle in radians for rotation.

    Returns:
    Tuple[Tuple[float, float], Tuple[float, float]]: Coordinates of A' and B' in the form ((x_a_prime, y_a_prime), (x_b_prime, y_b_prime)).
    """
    # Set initial coordinates for A and B
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Compute A' and B' using rotation
    A_prime = np.dot(rotation_matrix, A)
    B_prime = np.dot(rotation_matrix, B)
    
    return (A_prime[0], A_prime[1]), (B_prime[0], B_prime[1])