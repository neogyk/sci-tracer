from __future__ import annotations

from typing import Sequence


def total_permutations(counts: Sequence[int]) -> int:
    """
    Sum the counts corresponding to valid cycle types.

    Args:
        counts: A sequence of non-negative integers representing counts per cycle type.

    Returns:
        The total number of permutations satisfying the constraint.
    """
    return sum(counts)


def main() -> None:
    counts: tuple[int, ...] = (1, 15, 45, 15, 40, 40, 120, 120)
    total: int = total_permutations(counts)
    print(total)  # Expected: 396


if __name__ == "__main__":
    main()