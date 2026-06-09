from __future__ import annotations

import math
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


def rotate_about_origin(p: Point, theta: float) -> Point:
    """
    Rotate point p about the origin (0, 0) counterclockwise by angle theta (radians).
    """
    cos_t: float = math.cos(theta)
    sin_t: float = math.sin(theta)
    return Point(
        x=p.x * cos_t - p.y * sin_t,
        y=p.x * sin_t + p.y * cos_t,
    )


def compute_A_prime(theta: float) -> Point:
    """
    Given A = (1, 0) and B = (0, 0), compute A' after rotating A about B by theta (counterclockwise).
    Preconditions: 0 < theta < pi/2 (acute angle).
    Postconditions: A' = (cos(theta), sin(theta)) and distance A'B = 1.
    """
    if not (0.0 < theta < math.pi / 2):
        raise ValueError("Theta must be acute: 0 < theta < pi/2.")

    A: Point = Point(1.0, 0.0)
    B: Point = Point(0.0, 0.0)

    A_prime: Point = rotate_about_origin(A, theta)

    # Verify distance A'B == 1 within numerical tolerance
    dist_squared: float = (A_prime.x - B.x) ** 2 + (A_prime.y - B.y) ** 2
    if not math.isclose(dist_squared, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ArithmeticError("Rotation did not preserve distance as expected.")

    return A_prime


if __name__ == "__main__":
    # Example usage: rotate A by 30 degrees
    theta_example: float = math.radians(30.0)
    A_prime_example: Point = compute_A_prime(theta_example)
    print(f"A' = ({A_prime_example.x:.12f}, {A_prime_example.y:.12f})")