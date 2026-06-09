from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def distance(p: Point, q: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p.x - q.x, p.y - q.y)


def establish_coordinates() -> Tuple[Point, Point]:
    """
    Establish coordinates with AB = 1, placing B at the origin and A on the positive x-axis.
    Returns (A, B).
    """
    B: Point = Point(0.0, 0.0)
    A: Point = Point(1.0, 0.0)

    # Verify AB = 1 per precondition
    assert math.isclose(distance(A, B), 1.0, rel_tol=1e-12, abs_tol=1e-12), "AB must be 1."
    return A, B


if __name__ == "__main__":
    A_point, B_point = establish_coordinates()
    print(f"A = {A_point}, B = {B_point}, AB = {distance(A_point, B_point):.12g}")