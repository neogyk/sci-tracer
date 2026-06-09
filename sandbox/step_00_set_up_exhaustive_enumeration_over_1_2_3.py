from itertools import product
from typing import List, Tuple


def to_residue(x: int) -> int:
    """
    Map integers 1, 2, 3 to Z3 residues 1, 2, 0 respectively.
    """
    return x % 3


def satisfies_constraints(t: Tuple[int, ...]) -> bool:
    """
    Check if a 7-tuple over {1,2,3} satisfies:
    1) Sum of residues ≡ 0 (mod 3)
    2) Sum of cyclic triple products r_i * r_{i+1} * r_{i+3} ≡ 0 (mod 3)
    """
    r: List[int] = [to_residue(x) for x in t]
    if sum(r) % 3 != 0:
        return False

    n: int = len(r)
    S: int = 0
    for i in range(n):
        S = (S + (r[i] * r[(i + 1) % n] * r[(i + 3) % n])) % 3
    return S == 0


def count_satisfying_tuples() -> int:
    """
    Enumerate all 7-tuples over {1,2,3} and count those satisfying both constraints.
    """
    count: int = 0
    for tup in product([1, 2, 3], repeat=7):
        if satisfies_constraints(tup):
            count += 1
    return count


def main() -> None:
    count: int = count_satisfying_tuples()
    print(count)
    assert count == 243


if __name__ == "__main__":
    main()