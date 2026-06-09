from typing import List


def value_of_digits_in_base(digits: List[int], base: int) -> int:
    """
    Evaluate the integer value of a list of digits in the given base.
    Digits are provided most-significant to least-significant.
    """
    if base < 2:
        raise ValueError("Base must be at least 2.")
    value: int = 0
    for digit in digits:
        if not (0 <= digit < base):
            raise ValueError(f"Digit {digit} is out of range for base {base}.")
        value = value * base + digit
    return value


def single_digit_solutions_without_nine() -> List[int]:
    """
    Single-digit n (without digit 9) such that interpreting n in base (max_digit(n)+1) equals n.
    These are exactly 1 through 8.
    """
    return list(range(1, 9))


def verify_single_digit_solutions() -> bool:
    """
    Verify that for each d in 1..8, the numeral 'd' in base (d+1) evaluates to d.
    """
    for d in range(1, 9):
        base: int = d + 1
        if value_of_digits_in_base([d], base) != d:
            return False
    return True


def main() -> None:
    sols: List[int] = single_digit_solutions_without_nine()
    assert verify_single_digit_solutions(), "Verification failed for single-digit solutions."
    print("Single-digit solutions without a 9:", sols)
    print("Count:", len(sols))


if __name__ == "__main__":
    main()