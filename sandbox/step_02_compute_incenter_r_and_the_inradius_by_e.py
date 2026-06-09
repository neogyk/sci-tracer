from __future__ import annotations
from fractions import Fraction
from typing import Tuple

Vector3 = Tuple[Fraction, Fraction, Fraction]
Plane = Tuple[Fraction, Fraction, Fraction, Fraction]  # ax + by + cz + d = 0


def vec_sub(u: Vector3, v: Vector3) -> Vector3:
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def cross(u: Vector3, v: Vector3) -> Vector3:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def dot(u: Vector3, v: Vector3) -> Fraction:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def plane_from_points(p: Vector3, q: Vector3, r: Vector3) -> Plane:
    # Plane through p, q, r: normal n = (q-p) x (r-p), equation n·(x-p) = 0 -> n·x + d = 0 with d = -n·p
    n: Vector3 = cross(vec_sub(q, p), vec_sub(r, p))
    a, b, c = n
    d0: Fraction = -(a * p[0] + b * p[1] + c * p[2])
    return (a, b, c, d0)


def distance_squared_point_plane(point: Vector3, plane: Plane) -> Fraction:
    a, b, c, d0 = plane
    x, y, z = point
    num: Fraction = a * x + b * y + c * z + d0
    denom: Fraction = a * a + b * b + c * c
    return (num * num) / denom


def main() -> None:
    F = Fraction
    # Given coordinates from the construction
    A: Vector3 = (F(-5), F(0), F(0))
    B: Vector3 = (F(5), F(0), F(0))
    C: Vector3 = (F(0), F(12), F(9))
    D: Vector3 = (F(0), F(12), F(-9))

    # Solve 3y/5 = (60 - 5y)/13 exactly with Fractions
    left_coeff: Fraction = F(3, 5)
    right_const: Fraction = F(60, 13)
    right_coeff: Fraction = F(-5, 13)
    y: Fraction = right_const / (left_coeff - right_coeff)
    r: Fraction = left_coeff * y

    # Build face planes and verify equal distances (optional verification)
    planes: tuple[Plane, ...] = (
        plane_from_points(A, B, C),  # ABC
        plane_from_points(A, C, D),  # ACD
        plane_from_points(A, B, D),  # ABD
        plane_from_points(B, C, D),  # BCD
    )

    R: Vector3 = (F(0), y, F(0))
    r2: Fraction = r * r
    for pl in planes:
        assert distance_squared_point_plane(R, pl) == r2, "Distances to all faces must be equal to r"

    # Output results
    print(f"Incenter R = (0, {y}, 0)")
    print(f"Inradius r = {r}")


if __name__ == "__main__":
    main()