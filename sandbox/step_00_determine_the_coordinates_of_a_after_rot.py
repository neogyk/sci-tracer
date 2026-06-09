import numpy as np
from typing import Tuple

def calculate_A_prime(A: Tuple[float, float], B: Tuple[float, float], theta: float) -> Tuple[float, float]:
    """
    Calculate the coordinates of point A' after rotating point A around point B by an angle theta.

    :param A: Tuple representing coordinates of point A (x, y)
    :param B: Tuple representing coordinates of point B (x, y)
    :param theta: Angle in radians for rotation
    :return: Coordinates of A' after rotation
    """
    # Translating point A to the origin by subtracting B's coordinates
    A_translated = (A[0] - B[0], A[1] - B[1])
    
    # Rotation matrix for counterclockwise rotation
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Applying rotation matrix to translated coordinates
    A_translated_rotated = rotation_matrix @ A_translated
    
    # Translating back to original position by adding B's coordinates
    A_prime = (A_translated_rotated[0] + B[0], A_translated_rotated[1] + B[1])
    
    return A_prime

# Example usage
A = (1, 0)  # Example coordinates for A
B = (0, 0)  # Example coordinates for B
theta = np.pi / 4  # Example angle in radians (45 degrees)

A_prime = calculate_A_prime(A, B, theta)
print(f"Coordinates of A' after rotation: {A_prime}")