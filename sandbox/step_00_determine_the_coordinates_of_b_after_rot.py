import numpy as np
from typing import Tuple

def rotate_point_around_origin(point: Tuple[float, float], angle: float) -> Tuple[float, float]:
    """
    Rotate a point around the origin by a given angle.
    
    :param point: The (x, y) coordinates of the point.
    :param angle: The angle in radians. Positive for counterclockwise, negative for clockwise.
    :return: The (x', y') coordinates after rotation.
    """
    x, y = point
    # Rotation matrix for clockwise rotation
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    x_prime = cos_angle * x + sin_angle * y
    y_prime = -sin_angle * x + cos_angle * y
    return x_prime, y_prime

def rotate_B_around_A_prime(A: Tuple[float, float], B: Tuple[float, float], theta: float) -> Tuple[float, float]:
    """
    Determine the coordinates of B' after rotating B around A' by an angle -theta.
    
    :param A: The initial coordinates of A (x_A, y_A).
    :param B: The initial coordinates of B (x_B, y_B).
    :param theta: The angle -theta in radians.
    :return: The coordinates of B' after rotation around A'.
    """
    # Compute A' coordinates after rotating A around B by theta
    A_prime = rotate_point_around_origin((A[0] - B[0], A[1] - B[1]), theta)
    A_prime = (A_prime[0] + B[0], A_prime[1] + B[1])  # Translate back relative to original position

    # B' is B rotated around A_prime by -theta
    B_to_A_prime = (B[0] - A_prime[0], B[1] - A_prime[1])
    B_prime = rotate_point_around_origin(B_to_A_prime, -theta)
    B_prime = (B_prime[0] + A_prime[0], B_prime[1] + A_prime[1])  # Translate back

    return B_prime