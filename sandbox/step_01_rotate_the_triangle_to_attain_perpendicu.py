import numpy as np
from typing import Tuple

def rotate_triangle(vertices: np.ndarray, angle: float) -> np.ndarray:
    """Rotate triangle vertices about the origin using a transformation matrix.

    Args:
        vertices (np.ndarray): A numpy array of shape (3, 2) containing the coordinates of triangle vertices A, B, and C.
        angle (float): The rotation angle in radians.

    Returns:
        np.ndarray: A numpy array of shape (3, 2) containing the new coordinates of vertices A', B', and C' after rotation.
    """
    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Apply the rotation matrix to the vertices
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    
    return rotated_vertices

# Example usage
# Initial coordinates for triangle vertices A, B, C
vertices: np.ndarray = np.array([
    [13, 0],  # Assuming A's coordinates as (13, 0)
    [0, 14],  # Assuming B's coordinates as (0, 14)
    [-7.5, np.sqrt(15**2 - 7.5**2)]  # Calculating C's coordinates using Pythagorean theorem
])

# Rotate the triangle by 90 degrees (π/2 radians)
angle: float = np.pi / 2
rotated_vertices: np.ndarray = rotate_triangle(vertices, angle)

# New coordinates of vertices A', B', and C'
print(rotated_vertices)