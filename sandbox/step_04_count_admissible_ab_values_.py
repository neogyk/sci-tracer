from typing import Final


def count_congruence_in_range(low: int, high: int, residue: int, modulus: int) -> int:
    """Count integers n in [low, high] satisfying n % modulus == residue.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound.
        residue: Desired residue class.
        modulus: Modulus (must be positive).

    Returns:
        The count of integers in the specified range with the given congruence.
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    residue = residue % modulus
    rem_low: int = low % modulus
    delta: int = (residue - rem_low) % modulus
    first: int = low + delta
    if first > high:
        return 0
    return ((high - first) // modulus) + 1


def main() -> None:
    low: Final[int] = 15
    high: Final[int] = 2025
    residue: Final[int] = 1
    modulus: Final[int] = 4

    count: int = count_congruence_in_range(low, high, residue, modulus)
    print(count)  # Expected: 503


if __name__ == "__main__":
    main()