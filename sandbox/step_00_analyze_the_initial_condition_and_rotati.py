import numpy as np

def rotate_point_around_pivot(A: np.ndarray, B: np.ndarray, theta: float) -> np.ndarray:
    """
    Computes the coordinates of point A' by rotating point A around point B by an angle theta.

    Parameters:
    A (np.ndarray): Coordinates of point A.
    B (np.ndarray): Coordinates of point B.
    theta (float): Rotation angle in radians.

    Returns:
    np.ndarray: Coordinates of point A' after rotation.
    """
    # Translation matrix to move B to origin
    translate_to_origin = np.array([
        [1, 0, -B[0]],
        [0, 1, -B[1]],
        [0, 0, 1]
    ])

    # Rotation matrix about origin
    rotate = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Translate back to the original position of B
    translate_back = np.array([
        [1, 0, B[0]],
        [0, 1, B[1]],
        [0, 0, 1]
    ])

    # Full transformation matrix
    transformation_matrix = translate_back @ rotate @ translate_to_origin

    # Convert A to homogeneous coordinate for matrix multiplication
    A_homogeneous = np.array([A[0], A[1], 1])

    # Compute A' in homogeneous coordinates
    A_prime_homogeneous = transformation_matrix @ A_homogeneous

    # Convert back to Cartesian coordinates
    A_prime = A_prime_homogeneous[:2]

    return A_prime

# Given data
A = np.array([0, 0])  # Point A at origin
B = np.array([1, 0])  # Point B on x-axis
theta = np.pi / 4     # Example rotation angle, in radians

# Compute A' coordinates
A_prime = rotate_point_around_pivot(A, B, theta)
print(f"Coordinates of A' after rotating A: {A_prime}")