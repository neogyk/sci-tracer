from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Iterable, Tuple

DieFace = int
RollSeq = Tuple[DieFace, DieFace, DieFace, DieFace, DieFace, DieFace]

N_OUTCOMES: int = 6
TOTAL_SEQS: int = N_OUTCOMES ** 6  # 6^6 = 46656


def count_event_E_by_product() -> int:
    """
    Count sequences satisfying event E using the product rule.
    Event E: f2 is different from f3..f6 and f4 is different from f5,f6.
    Unconstrained positions: f1 free, f2 free, f3 != f2, f4 != f2, f5 != f2,f4, f6 != f2,f4.
    Returns the count |E|.
    """
    f1_choices: int = 6
    f2_choices: int = 6
    f3_choices: int = 5  # != f2
    f4_choices: int = 5  # != f2
    f5_choices: int = 4  # != f2,f4
    f6_choices: int = 4  # != f2,f4
    return f1_choices * f2_choices * f3_choices * f4_choices * f5_choices * f6_choices


def is_in_E(seq: RollSeq) -> bool:
    """Check E: f2 != any of f3..f6 and f4 != f5,f6."""
    _, f2, f3, f4, f5, f6 = seq
    return (f2 != f3 and f2 != f4 and f2 != f5 and f2 != f6) and (f4 != f5 and f4 != f6)


def count_event_E_by_bruteforce() -> int:
    """Brute-force count |E| by iterating all 6^6 sequences (for verification)."""
    faces: Iterable[int] = range(1, N_OUTCOMES + 1)
    cnt: int = 0
    for seq in product(faces, repeat=6):
        if is_in_E(seq):
            cnt += 1
    return cnt


def probability_of_E() -> Fraction:
    """Return exact probability P(E) as a Fraction."""
    return Fraction(count_event_E_by_product(), TOTAL_SEQS)


if __name__ == "__main__":
    count_product: int = count_event_E_by_product()
    count_bruteforce: int = count_event_E_by_bruteforce()
    prob: Fraction = probability_of_E()

    print({
        "|E| (product)": count_product,
        "|E| (bruteforce)": count_bruteforce,
        "Total": TOTAL_SEQS,
        "P(E) (exact)": f"{prob.numerator}/{prob.denominator}",
        "P(E) (float)": float(prob),
    })