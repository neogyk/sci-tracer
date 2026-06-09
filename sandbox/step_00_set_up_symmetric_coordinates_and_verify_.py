from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


def distance(p: Point3D, q: Point3D) -> float:
    dx: float = p.x - q.x
    dy: float = p.y - q.y
    dz: float = p.z - q.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def assert_close(name: str, value: float, expected: float, *, tol: float = 1e-9) -> None:
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{name} = {value:.12f}, expected {expected:.12f}")


def main() -> None:
    # Define points
    A: Point3D = Point3D(-5.0, 0.0, 0.0)
    B: Point3D = Point3D(5.0, 0.0, 0.0)
    C: Point3D = Point3D(0.0, 12.0, 9.0)
    D: Point3D = Point3D(0.0, 12.0, -9.0)

    # Compute pairwise distances
    dists: Dict[str, float] = {
        "AB": distance(A, B),
        "AC": distance(A, C),
        "AD": distance(A, D),
        "BC": distance(B, C),
        "BD": distance(B, D),
        "CD": distance(C, D),
    }

    # Expected values
    expected_equal: float = 5.0 * math.sqrt(10.0)

    # Verifications
    assert_close("AB", dists["AB"], 10.0)
    assert_close("CD", dists["CD"], 18.0)

    for edge in ("AC", "AD", "BC", "BD"):
        assert_close(edge, dists[edge], expected_equal)

    # If no assertion errors were raised, print confirmation and distances
    print("All edge length checks passed.")
    for k, v in dists.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()