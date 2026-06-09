import numpy as np
from typing import Tuple

def rotate_point(origin: Tuple[float, float], point: Tuple[float, float], theta: float) -> Tuple[float, float]:
    """
    Rotate a point around another point by an angle theta (in radians).
    
    :param origin: The center of rotation (x, y).
    :param point: The point to rotate (x, y).
    :param theta: The angle of rotation in radians.
    :return: The new position of the point after rotation.
    """
    ox, oy = origin
    px, py = point
    
    # Translate point back to origin
    px -= ox
    py -= oy
    
    # Perform rotation
    qx = px * np.cos(theta) - py * np.sin(theta)
    qy = px * np.sin(theta) + py * np.cos(theta)
    
    # Translate point back
    qx += ox
    qy += oy
    
    return qx, qy

def coordinates_B_prime(A_prime: Tuple[float, float], B: Tuple[float, float], theta: float) -> Tuple[float, float]:
    """
    Calculate the coordinates of B' after rotating B around A'.
    
    :param A_prime: The coordinates of point A' (x', y').
    :param B: The coordinates of point B (x, y).
    :param theta: The angle of rotation in radians.
    :return: The new position of B after rotation, which is B'.
    """
    # Rotate B around A'
    B_prime = rotate_point(A_prime, B, theta)
    
    return B_prime

# Example usage
# Let's assume A' is at (1, 0), B is at (1, 0), and theta is np.pi / 4 (45 degrees)
A_prime = (1.0, 0.0)
B = (1.0, 0.0)
theta = np.pi / 4

# Calculate B'
B_prime = coordinates_B_prime(A_prime, B, theta)
print(B_prime)