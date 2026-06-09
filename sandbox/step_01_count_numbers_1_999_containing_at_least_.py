from typing import Final

TOTAL: Final[int] = 999
WITHOUT_NINE: Final[int] = 9 ** 3 - 1  # Count of numbers in 1..999 without digit 9
WITH_NINE: Final[int] = TOTAL - WITHOUT_NINE  # Count with at least one digit 9


def brute_force_count_with_nine(start: int, end: int) -> int:
    """Return the count of integers in [start, end] containing at least one digit '9'."""
    count: int = 0
    for n in range(start, end + 1):
        if '9' in str(n):
            count += 1
    return count


def main() -> None:
    # Expected result per combinatorial counting
    print(WITH_NINE)
    # Sanity check via brute force
    assert WITH_NINE == brute_force_count_with_nine(1, 999)


if __name__ == "__main__":
    main()