from typing import Final


def main() -> None:
    START: Final[int] = 2
    END: Final[int] = 101

    interval_size: int = END - START + 1  # |{2, ..., 101}| = 100
    prime_count_up_to_101: int = 26      # π(101) = 26 (given)

    composites: int = interval_size - prime_count_up_to_101
    print(composites)  # Expected: 74


if __name__ == "__main__":
    main()