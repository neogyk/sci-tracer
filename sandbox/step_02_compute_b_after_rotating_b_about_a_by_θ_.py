from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)


def rotation_matrix_ccw(angle: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Counterclockwise rotation matrix R(angle).
    """
    c: float = math.cos(angle)
    s: float = math.sin(angle)
    return ((c, -s), (s, c))


def rotation_matrix_cw(angle: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Clockwise rotation matrix R(-angle) = [[cos a, sin a], [-sin a, cos a]].
    """
    c: float = math.cos(angle)
    s: float = math.sin(angle)
    return ((c, s), (-s, c))


def apply_matrix_to_vector(M: Tuple[Tuple[float, float], Tuple[float, float]], v: Point2D) -> Point2D:
    """
    Multiply 2x2 matrix M by vector v.
    """
    return Point2D(M[0][0] * v.x + M[0][1] * v.y, M[1][0] * v.x + M[1][1] * v.y)


def compute_A_prime(theta: float) -> Point2D:
    """
    A' after rotating A=(1,0) about B=(0,0) by theta counterclockwise.
    A' = (cos theta, sin theta).
    """
    return Point2D(math.cos(theta), math.sin(theta))


def compute_B_prime(theta: float) -> Point2D:
    """
    Compute B' after rotating B=(0,0) about A' by theta clockwise.

    Method:
    - A' = (cos theta, sin theta)
    - v = B - A' = (-cos theta, -sin theta)
    - R(-theta) v = (-1, 0)
    - B' = A' + (-1, 0) = (cos theta - 1, sin theta)
    """
    A_p: Point2D = compute_A_prime(theta)
    B: Point2D = Point2D(0.0, 0.0)

    # Vector from A' to B
    v: Point2D = B - A_p

    # Rotate v by -theta (i.e., clockwise by theta)
    R_cw: Tuple[Tuple[float, float], Tuple[float, float]] = rotation_matrix_cw(theta)
    rv: Point2D = apply_matrix_to_vector(R_cw, v)

    # Translate back by A' to get B'
    B_p: Point2D = A_p + rv
    return B_p


# Optional: direct closed-form computation of B' using the derived identity
def compute_B_prime_closed_form(theta: float) -> Point2D:
    """
    Closed-form result from the rotation identity:
    B' = (cos theta - 1, sin theta)
    """
    return Point2D(math.cos(theta) - 1.0, math.sin(theta))


if __name__ == "__main__":
    # Example usage and internal consistency check
    th: float = 0.7
    b_prime_matrix: Point2D = compute_B_prime(th)
    b_prime_closed: Point2D = compute_B_prime_closed_form(th)

    # Validate the two computations agree numerically
    assert math.isclose(b_prime_matrix.x, b_prime_closed.x, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(b_prime_matrix.y, b_prime_closed.y, rel_tol=1e-12, abs_tol=1e-12)