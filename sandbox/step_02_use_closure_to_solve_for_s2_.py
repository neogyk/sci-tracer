from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class S2Result:
    S2_numeric: float
    S2_closed_form_numeric: float
    S2_symbolic: str


def compute_S2() -> S2Result:
    """
    Use the closure relation ||sum e_i||^2 = 0 for an equiangular pentagon to solve for:
        S2 = sum s_i * s_{i+2}
    Given:
        S0 = sum s_i^2 = 308
        c1 = cos(72°) = (sqrt(5) - 1)/4
        C  = cos(36°) = (sqrt(5) + 1)/4
        S1 = sum s_i * s_{i+1} = 92 / c1  (from previous step via diagonals)

    Closure identity (pairwise dot products at angles 72° and 144°):
        0 = S0 + 2*c1*S1 + 2*cos(144°)*S2
        cos(144°) = -cos(36°) = -C
        => 0 = S0 + 2*c1*S1 - 2*C*S2
        => 2*C*S2 = S0 + 2*c1*S1
        => S2 = (S0 + 2*c1*S1) / (2*C)

    Returns numeric S2 and its closed form 246*(sqrt(5) - 1).
    """
    sqrt5: float = math.sqrt(5.0)
    c1: float = (sqrt5 - 1.0) / 4.0  # cos(72°)
    C: float = (sqrt5 + 1.0) / 4.0   # cos(36°)

    S0: float = 308.0
    S1: float = 92.0 / c1  # from earlier diagonal relation

    two_c1_S1: float = 2.0 * c1 * S1
    # Check intermediate step equals 184
    assert math.isclose(two_c1_S1, 184.0, rel_tol=1e-12, abs_tol=1e-12)

    rhs: float = S0 + two_c1_S1  # should be 492
    assert math.isclose(rhs, 492.0, rel_tol=1e-12, abs_tol=1e-12)

    S2: float = rhs / (2.0 * C)  # equals 246/C
    # Closed form using 1/C = sqrt(5) - 1
    S2_closed_form_numeric: float = 246.0 * (sqrt5 - 1.0)

    # Sanity check: both numerics match
    assert math.isclose(S2, S2_closed_form_numeric, rel_tol=1e-12, abs_tol=1e-12)

    return S2Result(
        S2_numeric=S2,
        S2_closed_form_numeric=S2_closed_form_numeric,
        S2_symbolic="246*(sqrt(5) - 1)",
    )


if __name__ == "__main__":
    result = compute_S2()
    print(f"S2 = {result.S2_symbolic} ≈ {result.S2_numeric:.12f}")