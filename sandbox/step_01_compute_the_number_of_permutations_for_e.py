from typing import Dict, List, Sequence, Tuple
import math


def cycle_type_label(counts: Dict[int, int]) -> str:
    """
    Create a human-readable label like '2^2·1^2' for a given cycle-count mapping.
    Keys are cycle lengths; values are multiplicities.
    """
    parts: List[str] = []
    for i in sorted(counts.keys()):
        a_i: int = counts[i]
        if a_i <= 0:
            continue
        parts.append(f"{i}" if a_i == 1 else f"{i}^{a_i}")
    return "·".join(parts) if parts else "1"


def count_permutations_for_type(n: int, counts: Dict[int, int]) -> int:
    """
    Compute the number of permutations of n with a_i i-cycles according to:
      count = n! / ∏_i (i^{a_i} a_i!)
    """
    denom: int = 1
    for i, a_i in counts.items():
        if a_i < 0:
            raise ValueError("Cycle multiplicities must be nonnegative.")
        if a_i == 0:
            continue
        denom *= (i ** a_i) * math.factorial(a_i)
    if math.factorial(n) % denom != 0:
        raise ArithmeticError("Denominator does not divide n!; invalid cycle type for given n.")
    return math.factorial(n) // denom


def counts_from_cycle_list(cycles: Sequence[int]) -> Dict[int, int]:
    """
    Given a list of cycle lengths (partition of n), return mapping length -> multiplicity.
    """
    counts: Dict[int, int] = {}
    for L in cycles:
        counts[L] = counts.get(L, 0) + 1
    return counts


def compute_counts_for_valid_types(n: int, valid_types: Sequence[Sequence[int]]) -> Dict[str, int]:
    """
    For each valid cycle-type (as a list of cycle lengths that sum to n),
    compute the number of permutations using the cycle index formula and
    return a mapping from human-readable labels to counts.
    """
    result: Dict[str, int] = {}
    for t in valid_types:
        counts = counts_from_cycle_list(t)
        label = cycle_type_label(counts)
        result[label] = count_permutations_for_type(n, counts)
    return result


def main() -> None:
    n: int = 6
    # Valid cycle types V as lists of cycle lengths (only lengths dividing 6)
    V: List[List[int]] = [
        [1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 2],
        [3, 1, 1, 1],
        [3, 3],
        [3, 2, 1],
        [6],
    ]

    counts_by_label: Dict[str, int] = compute_counts_for_valid_types(n, V)

    # Present results in the same order as V
    ordered_results: List[Tuple[str, int]] = []
    for t in V:
        label = cycle_type_label(counts_from_cycle_list(t))
        ordered_results.append((label, counts_by_label[label]))

    # Print results
    for label, count in ordered_results:
        print(f"{label}: {count}")

    # Optional: verify expected postcondition values
    expected_ordered: List[int] = [1, 15, 45, 15, 40, 40, 120, 120]
    actual_ordered: List[int] = [c for _, c in ordered_results]
    assert actual_ordered == expected_ordered, f"Mismatch: got {actual_ordered}, expected {expected_ordered}"


if __name__ == "__main__":
    main()