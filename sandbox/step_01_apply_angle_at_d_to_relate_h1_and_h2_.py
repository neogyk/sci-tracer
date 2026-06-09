from __future__ import annotations

from typing import Set, Tuple


def dc_vector(h1: int, h2: int) -> Tuple[int, int]:
    """
    Compute the vector DC given:
      C = (14, h1 - 14)
      D = (20, h2)

    Returns:
      (vx, vy) = (C.x - D.x, C.y - D.y) = (-6, h1 - h2 - 14)
    """
    vx: int = 14 - 20
    vy: int = (h1 - 14) - h2
    return vx, vy


def angle_with_vertical_is_45_at_D(h1: int, h2: int) -> bool:
    """
    Enforce that ∠D between DC and DE (vertical down) is 45°.
    For a vector v making 45° with the vertical, we must have |vx| = |vy|.

    With DC = (-6, h1 - h2 - 14), the condition becomes:
        |h1 - h2 - 14| = 6  ⇔  h1 - h2 ∈ {8, 20}.
    """
    vx, vy = dc_vector(h1, h2)
    return abs(vx) == abs(vy)


def possible_h1_minus_h2() -> Set[int]:
    """
    From the 45° condition at D:
        DC = (-6, h1 - h2 - 14) must satisfy |vx| = |vy|.
    Since |vx| = 6, we need |h1 - h2 - 14| = 6, hence:
        h1 - h2 ∈ {8, 20}.
    """
    return {8, 20}


# Example validations (can be removed if importing as a library)
if __name__ == "__main__":
    # Check the logical equivalence for sample values
    for diff in possible_h1_minus_h2():
        h1_test: int = 30
        h2_test: int = h1_test - diff
        assert angle_with_vertical_is_45_at_D(h1_test, h2_test)

    # Negative test: choose a difference not in {8, 20}
    h1_bad: int = 30
    h2_bad: int = h1_bad - 10
    assert not angle_with_vertical_is_45_at_D(h1_bad, h2_bad)