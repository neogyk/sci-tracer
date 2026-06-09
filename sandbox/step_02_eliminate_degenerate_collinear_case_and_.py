from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass(frozen=True)
class Point:
    x: int
    y: int


def construct_points(h1: int, h2: int) -> Tuple[Point, Point, Point]:
    """
    Construct points B, C, D for the configuration:
    - A = (0, 0), E = (20, 0) are implicit and not needed here.
    - B = (0, h1)
    - C = (14, h1 - 14)  derived from BC = 14√2 at 45° down-right from B
    - D = (20, h2)
    """
    b = Point(0, h1)
    c = Point(14, h1 - 14)
    d = Point(20, h2)
    return b, c, d


def cross_bc_cd(h1: int, h2: int) -> int:
    """
    Compute the 2D cross product z-component of vectors:
      (B -> C) x (C -> D)
    Using:
      B = (0, h1)
      C = (14, h1 - 14)
      D = (20, h2)

    Algebra (verified):
      cross = -14 * (h1 - h2 - 20)

    Returns:
      int: the z-component (signed area factor). Zero implies collinearity of B, C, D.
    """
    # Direct formula to avoid recomputation of vectors.
    return -14 * (h1 - h2 - 20)


def discard_collinear_and_enforce_constraints() -> Dict[str, object]:
    """
    Implements the step:
      - From angle constraints we have h1 - h2 ∈ {8, 20}.
      - Compute cross(B→C, C→D) = -14(h1 - h2 - 20).
      - If h1 - h2 = 20 => cross = 0 ⇒ B, C, D collinear ⇒ ∠C = 180° ⇒ invalid (degenerate).
      - Keep h2 = h1 - 8.
      - Enforce that B, C, D are above AE (y > 0):
          B.y = h1 > 0
          C.y = h1 - 14 > 0 ⇒ h1 ≥ 15 (as integer)
          D.y = h2 = h1 - 8 > 0 ⇒ h1 ≥ 9 (redundant given h1 ≥ 15)

    Returns:
      Dict with the validated relation and bounds:
        {
          "valid_relation": "h2 = h1 - 8",
          "h1_min": 15,
          "notes": "...",
          "sanity_checks": { ... }
        }
    """
    # Sanity checks: verify cross-product behavior for the two candidate differences
    cross_when_diff_20 = cross_bc_cd(h1=100, h2=80)  # h1 - h2 = 20
    cross_when_diff_8 = cross_bc_cd(h1=100, h2=92)   # h1 - h2 = 8

    notes = (
        "Discarded h1 - h2 = 20 because cross(B→C, C→D) = 0 ⇒ B, C, D are collinear (∠C = 180°). "
        "Thus the nonconvex pentagon would degenerate. "
        "Keeping h2 = h1 - 8. For points above AE (y > 0), require C.y = h1 - 14 > 0 ⇒ h1 ≥ 15. "
        "Then D.y = h1 - 8 > 0 is automatically satisfied."
    )

    result: Dict[str, object] = {
        "valid_relation": "h2 = h1 - 8",
        "h1_min": 15,
        "notes": notes,
        "sanity_checks": {
            "cross_when_h1_minus_h2_equals_20": cross_when_diff_20,  # expected 0 (collinear)
            "cross_when_h1_minus_h2_equals_8": cross_when_diff_8,    # expected nonzero (valid non-collinear)
        },
    }
    return result


def is_configuration_valid(h1: int, h2: int) -> bool:
    """
    Check whether a given integer pair (h1, h2) satisfies the enforced constraints:
      - h2 = h1 - 8
      - h1 ≥ 15
      - B, C, D above AE (y > 0)
      - B, C, D not collinear

    Returns:
      True if valid, False otherwise.
    """
    if h2 != h1 - 8:
        return False
    if h1 < 15:
        return False

    b, c, d = construct_points(h1, h2)
    if not (b.y > 0 and c.y > 0 and d.y > 0):
        return False

    return cross_bc_cd(h1, h2) != 0


if __name__ == "__main__":
    # Produce the enforced constraints and run a quick validation example.
    constraints = discard_collinear_and_enforce_constraints()
    print(constraints)

    # Example validation with the minimal admissible h1
    example_h1 = constraints["h1_min"]  # 15
    example_h2 = example_h1 - 8         # 7
    print(
        {
            "example_h1": example_h1,
            "example_h2": example_h2,
            "valid_example": is_configuration_valid(example_h1, example_h2),
        }
    )