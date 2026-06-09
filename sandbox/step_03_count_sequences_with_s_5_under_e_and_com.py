from __future__ import annotations
from typing import Tuple, Iterable
from fractions import Fraction
from itertools import product


def is_visible(seq: Tuple[int, ...], t: int) -> bool:
    """
    A sticker placed at time t (1-based) is visible iff its face does not appear at a later time.
    """
    idx: int = t - 1  # convert to 0-based
    face: int = seq[idx]
    for j in range(idx + 1, len(seq)):
        if seq[j] == face:
            return False
    return True


def event_E(seq: Tuple[int, ...]) -> bool:
    """
    Conditioning event E: stickers 2 and 4 are visible (6 is always visible by being the last roll).
    """
    return is_visible(seq, 2) and is_visible(seq, 4)


def count_sequences_bruteforce() -> Tuple[int, int]:
    """
    Brute-force verification counts:
    - denom: number of sequences satisfying E
    - numer: number of sequences with S=5 (exactly one duplicate pair) under E
    """
    denom: int = 0
    numer: int = 0
    for seq in product(range(1, 7), repeat=6):
        if not event_E(seq):
            continue
        denom += 1
        # S = 5 means exactly one duplicate pair -> exactly 5 distinct faces
        if len(set(seq)) == 5:
            numer += 1
    return denom, numer


def analytic_counts() -> Tuple[int, int]:
    """
    Analytic counts per the combinatorial argument:
    - |E| = 6 (f1) * 6 (f2) * 5 (f3 != f2) * 5 (f4 != f2) * 4 (f5 != f2, f4) * 4 (f6 != f2, f4) = 14400
    - Allowed duplicate index pairs under E: 9 pairs
    - For each pair: choose repeated face (6 ways) and assign the remaining 4 positions injectively from 5 faces: P(5,4)=120
      Hence |S=5 ∧ E| = 9 * 6 * 120 = 6480
    """
    denom: int = 6 * 6 * 5 * 5 * 4 * 4  # 14400
    numer: int = 9 * 6 * 120           # 6480
    return denom, numer


def main() -> None:
    # Analytic result
    denom_a, numer_a = analytic_counts()
    p: Fraction = Fraction(numer_a, denom_a)  # = 9/20
    m_plus_n: int = p.numerator + p.denominator  # 9 + 20 = 29

    # Optional brute-force check (can be commented out if not desired)
    denom_b, numer_b = count_sequences_bruteforce()
    assert denom_b == denom_a and numer_b == numer_a, "Brute-force check failed"

    # Output results
    print(f"|E| = {denom_a}")
    print(f"|S=5 ∧ E| = {numer_a}")
    print(f"p = {p} = {float(p):.5f}")
    print(f"m + n = {m_plus_n}")


if __name__ == "__main__":
    main()