import numpy as np
from typing import Tuple

def rotate_triangle(
    A: Tuple[float, float], 
    B: Tuple[float, float], 
    C: Tuple[float, float], 
    angle: float
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Rotate the triangle with vertices A, B, and C by a given angle around the origin.

    Parameters:
    A (Tuple[float, float]): x and y coordinates of vertex A.
    B (Tuple[float, float]): x and y coordinates of vertex B.
    C (Tuple[float, float]): x and y coordinates of vertex C.
    angle (float): The rotation angle in radians.

    Returns:
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]: New coordinates of vertices A', B', and C' after rotation.
    """
    # Define the rotation matrix for a given angle
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Rotate each vertex using the rotation matrix
    A_rotated = tuple(np.dot(rotation_matrix, A))
    B_rotated = tuple(np.dot(rotation_matrix, B))
    C_rotated = tuple(np.dot(rotation_matrix, C))
    
    return A_rotated, B_rotated, C_rotated

# Example usage
# Assuming initial vertex positions at (x, y) for demonstration
A = (0, 0)  # Placeholders, replace with actual coordinates
B = (1, 0)
C = (0, 1)
angle_radians = np.pi / 2  # 90-degree rotation

A_prime, B_prime, C_prime = rotate_triangle(A, B, C, angle_radians)
print("Rotated vertices (A', B', C'):", A_prime, B_prime, C_prime)