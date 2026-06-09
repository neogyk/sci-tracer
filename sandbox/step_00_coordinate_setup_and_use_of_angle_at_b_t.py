from __future__ import annotations
from dataclasses import dataclass
from math import isclose, sqrt
from typing import Tuple


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def derive_c_from_b(ab_length: float) -> Tuple[Point, Point, Point]:
    """
    Given AB = h1 (vertical), AE horizontal with A=(0,0), E=(20,0), and angle at B is 45°
    between BA (downward) and BC with |BC| = 14*sqrt(2), compute coordinates of B and C.

    Placement:
    - A = (0, 0)
    - B = (0, h1)
    - BC makes a 45° angle to the downward vertical, heading to the right (down-right).
      Thus Δx = 14, Δy = -14 since |BC| = 14*sqrt(2).
    - Therefore C = B + (14, -14) = (14, h1 - 14).

    Constraints enforced:
    - h1 > 14 to keep C above AE (y > 0) per precondition.

    Returns:
    - A, B, C as Points.
    """
    if ab_length <= 14:
        raise ValueError("Precondition violated: h1=AB must exceed 14 to keep C above AE.")

    a = Point(0.0, 0.0)
    b = Point(0.0, float(ab_length))
    c = Point(14.0, float(ab_length) - 14.0)

    # Optional verification that angle ∠ABC with BA is 45° using dot product
    # BA = A - B, BC = C - B
    ba_x, ba_y = a.x - b.x, a.y - b.y  # (0, -h1)
    bc_x, bc_y = c.x - b.x, c.y - b.y  # (14, -14)
    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = sqrt(ba_x * ba_x + ba_y * ba_y)
    mag_bc = sqrt(bc_x * bc_x + bc_y * bc_y)
    cos_theta = dot / (mag_ba * mag_bc)
    assert isclose(cos_theta, 1.0 / sqrt(2.0), rel_tol=1e-12, abs_tol=1e-12), "Angle at B is not 45°"

    return a, b, c


if __name__ == "__main__":
    # Example usage with h1 = 17 (satisfies h1 > 14)
    A, B, C = derive_c_from_b(17.0)
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")