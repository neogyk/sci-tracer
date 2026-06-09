from typing import Final


def compute_valid_count(composites_count: int, excluded_prime_squares: int) -> int:
    """
    Compute the final count of valid n by subtracting excluded prime squares from
    the total number of composite m in the range.

    Args:
        composites_count: Total number of composite integers m in [2, 101].
        excluded_prime_squares: Count of excluded m that are prime squares ≤ 101.

    Returns:
        The number of n ≤ 100 representable as a + b + ab with a ≠ b.
    """
    return composites_count - excluded_prime_squares


if __name__ == "__main__":
    composites: Final[int] = 74
    prime_squares: Final[int] = 4
    result: int = compute_valid_count(composites, prime_squares)
    print(result)