from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModConstraint:
    modulus: int
    residue: int


@dataclass(frozen=True)
class FeasibilityConstraints:
    # Lower bound on n from inequalities r >= 7 and b >= 7
    min_n_by_inequalities: int
    # Congruence constraint on n (from integrality condition)
    congruence: ModConstraint
    # Smallest n satisfying both the inequalities and the congruence
    minimal_feasible_n: int


def minimal_n_from_inequalities(r_min: int = 7, b_min: int = 7) -> int:
    """
    Given r = (5n + 2)/8 and b = (3n - 2)/8, compute the smallest integer n such that
    r >= r_min and b >= b_min (ignoring integrality/congruence for the moment).
    """
    # Solve (5n + 2)/8 >= r_min  ->  5n + 2 >= 8*r_min  ->  n >= ceil((8*r_min - 2)/5)
    num_r: int = 8 * r_min - 2
    n_r: int = (num_r + 5 - 1) // 5  # ceil division for positive integers

    # Solve (3n - 2)/8 >= b_min  ->  3n - 2 >= 8*b_min  ->  n >= ceil((8*b_min + 2)/3)
    num_b: int = 8 * b_min + 2
    n_b: int = (num_b + 3 - 1) // 3  # ceil division

    return max(n_r, n_b)


def first_n_with_congruence_at_least(min_n: int, modulus: int, residue: int) -> int:
    """
    Find the smallest integer n >= min_n with n % modulus == residue.
    """
    rem: int = min_n % modulus
    delta: int = (residue - rem) % modulus
    return min_n + delta


def compute_feasibility_constraints() -> FeasibilityConstraints:
    """
    Implements the 'Impose feasibility constraints' step:
    - From r = (5n + 2)/8 and b = (3n - 2)/8 with r, b >= 7, derive n >= 20.
    - Combine with the integrality/congruence condition n ≡ 6 (mod 8).
    """
    # Inequality-derived lower bound
    min_n_ineq: int = minimal_n_from_inequalities(r_min=7, b_min=7)  # expected 20

    # Congruence from prior deduction (integrality): n ≡ 6 (mod 8)
    congruence = ModConstraint(modulus=8, residue=6)

    # Smallest n satisfying both
    minimal_feasible: int = first_n_with_congruence_at_least(
        min_n=min_n_ineq, modulus=congruence.modulus, residue=congruence.residue
    )

    return FeasibilityConstraints(
        min_n_by_inequalities=min_n_ineq,
        congruence=congruence,
        minimal_feasible_n=minimal_feasible,
    )


def enumerate_feasible_n(count: int) -> List[int]:
    """
    Enumerate the first 'count' feasible n values satisfying:
      - n >= 20 (from inequalities r, b >= 7), and
      - n ≡ 6 (mod 8) (from integrality).
    """
    constraints: FeasibilityConstraints = compute_feasibility_constraints()
    start: int = constraints.minimal_feasible_n
    step: int = constraints.congruence.modulus
    return [start + k * step for k in range(count)]


if __name__ == "__main__":
    constraints = compute_feasibility_constraints()
    print(f"n >= {constraints.min_n_by_inequalities} and n ≡ {constraints.congruence.residue} (mod {constraints.congruence.modulus})")
    print(f"Smallest feasible n: {constraints.minimal_feasible_n}")
    print(f"First five feasible n: {enumerate_feasible_n(5)}")