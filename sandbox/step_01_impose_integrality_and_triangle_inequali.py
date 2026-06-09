from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple


@dataclass(frozen=True)
class Bounds:
    m: int
    k: int
    BC: int
    u_min: int  # minimal integer u satisfying u > 200 - 8 m^2 and u >= 1
    u_max: int  # maximal integer u from tangency (u <= 199)
    v_min: int  # minimal integer v satisfying v > 225 - 9 m^2 and v >= 1
    v_max: int  # maximal integer v from tangency (v <= 224)


def residues_k_mod25() -> Set[int]:
    """
    Solve 17 * k^2 ≡ 0 (mod 25) over residues k ∈ {0,1,...,24}.
    Returns the set of residues k (mod 25) that satisfy the congruence.
    """
    return {k for k in range(25) if (17 * (k * k)) % 25 == 0}


def k_is_multiple_of_5() -> bool:
    """
    From 17 * k^2 ≡ 0 (mod 25) and gcd(17, 25) = 1, deduce k^2 ≡ 0 (mod 25),
    hence 5 | k. This function verifies the deduction computationally over residues.
    """
    residues: Set[int] = residues_k_mod25()
    # Valid residues are exactly multiples of 5 in modulo 25
    return residues == {0, 5, 10, 15, 20}


def feasible_m_values() -> List[int]:
    """
    With k = 5m and BC = 17k = 85m, the triangle inequality requires BC < AB + AC = 425,
    so 85m < 425 ⇒ m ∈ {1, 2, 3, 4}.
    """
    return [1, 2, 3, 4]


def bounds_for_m(m: int) -> Bounds:
    """
    Given m, compute:
    - k = 5m, BC = 17k = 85m
    - From R^2 > 0 and power relations:
        u > 200 - 8 m^2 and v > 225 - 9 m^2
    - Tangency and the fact B, C are not on ω give:
        u ≤ 199 and v ≤ 224
    Convert strict inequalities to minimal integer bounds.
    """
    if m <= 0:
        raise ValueError("m must be a positive integer")

    k: int = 5 * m
    BC: int = 17 * k  # = 85m

    # Strict lower bounds converted to integer minima:
    u_min: int = max(1, (200 - 8 * m * m) + 1)
    v_min: int = max(1, (225 - 9 * m * m) + 1)

    # Upper bounds from tangency constraints:
    u_max: int = 199
    v_max: int = 224

    return Bounds(m=m, k=k, BC=BC, u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)


def main() -> None:
    # Step 1: Modulo 25 deduction => k is a multiple of 5
    assert k_is_multiple_of_5(), "Modulo 25 deduction failed; expected k ≡ 0 (mod 5)."
    print("From 200u - 225v ≡ 17k^2 - 10625 (mod 25), we get 17k^2 ≡ 0 (mod 25).")
    print("Since gcd(17,25)=1, k^2 ≡ 0 (mod 25) ⇒ k ≡ 0 (mod 5). Let k = 5m.")

    # Step 2: Triangle inequality bound on m
    m_vals: List[int] = feasible_m_values()
    print(f"Triangle inequality: BC = 17k = 85m < 425 ⇒ m ∈ {m_vals}")

    # Step 3: Record lower/upper bounds on u and v for each m
    print("Bounds for (u, v) given R^2 > 0 and tangency (u ≤ 199, v ≤ 224):")
    for m in m_vals:
        b: Bounds = bounds_for_m(m)
        print(
            f"  m={b.m}: k={b.k}, BC={b.BC}; "
            f"u ∈ [{b.u_min}, {b.u_max}], v ∈ [{b.v_min}, {b.v_max}] "
            f"(from u > 200 − 8m^2, v > 225 − 9m^2)."
        )


if __name__ == "__main__":
    main()